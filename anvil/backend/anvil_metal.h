// anvil_metal.h — host side of programs compiled by `anvil metal` (Anvil's Metal backend), in
// Objective-C++. Kernels are encoded into command buffers as the program runs, and the host only
// waits for the GPU (anvil_sync) when it must read memory the GPU writes: a print, a branch on a
// computed value, a run-time call. Every tensor is an MTLBuffer in shared memory, so after a sync
// the host reads and writes it directly through its .contents pointer.
#pragma once
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "anvil_host.h"

static id<MTLDevice> anvil_dev;
static id<MTLCommandQueue> anvil_queue;
static id<MTLLibrary> anvil_lib;
static id<MTLCommandBuffer> anvil_cb;              // being encoded
static id<MTLComputeCommandEncoder> anvil_enc;
static id<MTLCommandBuffer> anvil_last;            // the last one committed
static int anvil_pending;                          // dispatches in anvil_cb
static id<MTLBuffer> anvil_fault_buf;              // {code, value} of the first failed bounds check
static const char *anvil_fault_messages[256];

static void anvil_die(const char *what, NSError *err) {
    fflush(stdout);
    fprintf(stderr, "anvil: %s%s%s\n", what, err ? ": " : "", err ? [[err localizedDescription] UTF8String] : "");
    exit(1);
}

static void anvil_metal_init(const char *msl) {
    anvil_dev = MTLCreateSystemDefaultDevice();
    if (!anvil_dev) anvil_die("no Metal device", nil);
    anvil_queue = [anvil_dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;            // no fast-math shortcuts
    NSError *err = nil;
    anvil_lib = [anvil_dev newLibraryWithSource:[NSString stringWithUTF8String:msl] options:opt error:&err];
    if (!anvil_lib) anvil_die("the Metal kernels did not compile", err);
    anvil_fault_buf = [anvil_dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(anvil_fault_buf.contents, 0, 16);
}

static id<MTLComputePipelineState> anvil_pipeline(const char *name) {
    id<MTLFunction> f = [anvil_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    NSError *err = nil;
    id<MTLComputePipelineState> p = [anvil_dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) anvil_die("a Metal kernel could not be prepared", err);
    return p;
}

static id<MTLBuffer> anvil_buffer(size_t bytes) {
    if (bytes < 16) bytes = 16;
    id<MTLBuffer> b = [anvil_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) { fprintf(stderr, "anvil: out of GPU memory (%zu bytes)\n", bytes); exit(1); }
    memset(b.contents, 0, bytes);
    return b;
}

static id<MTLComputeCommandEncoder> anvil_encoder(void) {
    if (!anvil_cb) anvil_cb = [anvil_queue commandBuffer];
    if (!anvil_enc) anvil_enc = [anvil_cb computeCommandEncoder];   // serial: each dispatch sees the ones before it
    return anvil_enc;
}

static void anvil_end_encoder(void) {
    if (anvil_enc) { [anvil_enc endEncoding]; anvil_enc = nil; }
}

// for Metal Performance Shaders (matrix products), which encode into the command buffer themselves
static id<MTLCommandBuffer> anvil_mps_buffer(void) {
    anvil_end_encoder();
    if (!anvil_cb) anvil_cb = [anvil_queue commandBuffer];
    return anvil_cb;
}

static void anvil_commit(void) {
    anvil_end_encoder();
    if (!anvil_cb) return;
    [anvil_cb commit];
    anvil_last = anvil_cb;
    anvil_cb = nil;
    anvil_pending = 0;
}

static void anvil_counted(void) {
    if (++anvil_pending >= 64) anvil_commit();               // let the GPU start while the host encodes more
}

// one thread per element, in threadgroups of up to 256
static void anvil_dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, long n) {
    NSUInteger group = p.maxTotalThreadsPerThreadgroup;
    if (group > 256) group = 256;
    if ((long)group > n) group = n > 0 ? (NSUInteger)n : 1;
    [e dispatchThreads:MTLSizeMake((NSUInteger)(n > 0 ? n : 1), 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    anvil_counted();
}

static void anvil_sync(void) {
    anvil_commit();
    if (anvil_last) {
        [anvil_last waitUntilCompleted];
        if (anvil_last.status == MTLCommandBufferStatusError) anvil_die("the GPU failed", anvil_last.error);
        anvil_last = nil;
    }
}

static void anvil_check_faults(void) {
    anvil_sync();
    int *f = (int *)anvil_fault_buf.contents;
    if (f[0]) {
        fflush(stdout);
        fprintf(stderr, "anvil: runtime error: %s (got %d)\n", anvil_fault_messages[f[0]], f[1]);
        exit(1);
    }
}

// a failed bounds check in a scalar computed on the host
static void anvil_fail(int code, long long value) { anvil_oob(anvil_fault_messages[code], value); }

static inline uint32_t anvil_bits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
static inline uint32_t anvil_bits(int v) { return (uint32_t)v; }

// ---------------------------------------------------------------- matrix products (Metal Performance Shaders)
static MPSMatrixDescriptor *anvil_matrix(long rows, long cols, long ld) {
    return [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                                                rowBytes:(NSUInteger)(4 * ld) dataType:MPSDataTypeFloat32];
}

static MPSMatrixMultiplication *anvil_product(BOOL ta, BOOL tb, long m, long n, long k, double beta) {
    return [[MPSMatrixMultiplication alloc] initWithDevice:anvil_dev transposeLeft:ta transposeRight:tb resultRows:(NSUInteger)m
                                             resultColumns:(NSUInteger)n interiorColumns:(NSUInteger)k alpha:1.0 beta:beta];
}

static void anvil_mps(MPSMatrixMultiplication *mm, id<MTLBuffer> a, long ao, MPSMatrixDescriptor *da, id<MTLBuffer> b, long bo,
                    MPSMatrixDescriptor *db, id<MTLBuffer> c, long co, MPSMatrixDescriptor *dc) {
    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:a offset:(NSUInteger)(4 * ao) descriptor:da];
    MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:b offset:(NSUInteger)(4 * bo) descriptor:db];
    MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:c offset:(NSUInteger)(4 * co) descriptor:dc];
    [mm encodeToCommandBuffer:anvil_mps_buffer() leftMatrix:A rightMatrix:B resultMatrix:C];
    anvil_counted();
}

