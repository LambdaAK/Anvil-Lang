// ============================================================================
//  anvil runtime — AArch64 / Apple (Mach-O). Appended to every compiled program.
//  Depends only on libSystem (printf, putchar, fflush, exit, clock) and libz.
// ============================================================================
    .section __TEXT,__text,regular,pure_instructions

// ---------------------------------------------------------------------------
// void _anvil_rt_init(void) — remember the start time for clock(); start worker threads
    .p2align 2
_anvil_rt_init:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov w0, #8                          // CLOCK_UPTIME_RAW: stops while the Mac sleeps, like Python's perf_counter
    bl _clock_gettime_nsec_np
    adrp x9, _anvil_t0@PAGE
    str x0, [x9, _anvil_t0@PAGEOFF]
    // thread count: $ANVIL_THREADS, else the number of performance cores (at most 8)
    adrp x0, Lrt_env_threads@PAGE
    add x0, x0, Lrt_env_threads@PAGEOFF
    bl _getenv
    cbz x0, Lini_sysctl
    bl _atoi
    sxtw x19, w0
    b Lini_clamp
Lini_sysctl:
    str xzr, [sp, #32]                  // value
    mov x9, #8
    str x9, [sp, #40]                   // length
    adrp x0, Lrt_sysctl_pcores@PAGE
    add x0, x0, Lrt_sysctl_pcores@PAGEOFF
    add x1, sp, #32
    add x2, sp, #40
    mov x3, #0
    mov x4, #0
    bl _sysctlbyname
    ldr w19, [sp, #32]
    cmp x19, #8
    mov x9, #8
    csel x19, x19, x9, le
Lini_clamp:
    cmp x19, #1
    mov x9, #1
    csel x19, x19, x9, ge
    cmp x19, #64
    mov x9, #64
    csel x19, x19, x9, le
    adrp x9, _anvil_nthreads@PAGE
    str x19, [x9, _anvil_nthreads@PAGEOFF]
    mov x20, #1
Lini_spawn:
    cmp x20, x19
    b.ge Lini_done
    add x0, sp, #32                     // pthread_t (discarded)
    mov x1, #0
    adrp x2, _anvil_rt_worker@PAGE
    add x2, x2, _anvil_rt_worker@PAGEOFF
    mov x3, #0
    bl _pthread_create
    add x20, x20, #1
    b Lini_spawn
Lini_done:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// ---------------------------------------------------------------------------
// Parallel kernels.  A parallel kernel is called as kernel(start, end) over its outermost
// loop. _anvil_rt_parallel splits [0, total) into chunks of `grain`.
//
// Protocol: _anvil_job_word = (generation << 32) | next_chunk. To post a job the caller first
// closes the old one (new generation, chunk index 0xffffffff), then writes the job fields,
// then opens it (next generation, chunk 0) with a release store. Threads claim chunks with
// compare-and-swap on the whole word, so a claim only succeeds against the word it read,
// and the fields can only change while the word is closed. Each finished chunk increments
// _anvil_job_done; the caller waits until it reaches the chunk count.
// Idle workers sleep in `wfe` (woken by `sev` or by the store to the word they monitor).
//
// void _anvil_rt_parallel(void (*kernel)(i64, i64), i64 total, i64 grain)
    .p2align 2
_anvil_rt_parallel:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    adrp x9, _anvil_nthreads@PAGE
    ldr x9, [x9, _anvil_nthreads@PAGEOFF]
    cmp x9, #1
    b.le Lpar_serial
    cmp x1, x2
    b.le Lpar_serial
    add x10, x1, x2
    sub x10, x10, #1
    udiv x10, x10, x2                   // chunks
    // 1. close the previous job: no claim against an older word can succeed after this
    adrp x11, _anvil_job_word@PAGE
    add x11, x11, _anvil_job_word@PAGEOFF
    ldr x12, [x11]
    lsr x12, x12, #32
    add x12, x12, #1
    lsl x13, x12, #32
    orr x13, x13, #0xffffffff           // (g+1, "no chunks left")
    stlr x13, [x11]
    // 2. write the job
    adrp x9, _anvil_job@PAGE
    add x9, x9, _anvil_job@PAGEOFF
    stp x0, x1, [x9]                    // fn, total
    stp x2, x10, [x9, #16]              // grain, chunks
    adrp x14, _anvil_job_done@PAGE
    add x14, x14, _anvil_job_done@PAGEOFF
    str xzr, [x14]
    // 3. open it: (g+2, chunk 0); the release store makes the fields and done visible
    add x12, x12, #1
    lsl x13, x12, #32
    stlr x13, [x11]
    sev
    mov x0, x12
    bl Lrt_run_chunks
    adrp x9, _anvil_job@PAGE
    add x9, x9, _anvil_job@PAGEOFF
    ldr x10, [x9, #24]                  // chunks
    adrp x11, _anvil_job_done@PAGE
    add x11, x11, _anvil_job_done@PAGEOFF
Lpar_wait:
    ldaxr x12, [x11]
    cmp x12, x10
    b.hs Lpar_out
    wfe
    b Lpar_wait
Lpar_out:
    clrex
    ldp x29, x30, [sp], #32
    ret
Lpar_serial:
    mov x9, x0
    mov x0, #0
    blr x9
    ldp x29, x30, [sp], #32
    ret

// Lrt_run_chunks(x0 = generation): claim and run chunks of that job until none are left.
// Kernels clobber registers, so all state is reloaded from memory / the stack.
    .p2align 2
Lrt_run_chunks:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x0, [sp, #16]
Lrc_loop:
    adrp x9, _anvil_job_word@PAGE
    add x9, x9, _anvil_job_word@PAGEOFF
    ldar x11, [x9]                      // acquire
    lsr x12, x11, #32
    ldr x13, [sp, #16]
    cmp x12, x13
    b.ne Lrc_done                       // a different job: not ours to help with
    mov w14, w11                        // chunk index (low 32 bits)
    adrp x10, _anvil_job@PAGE
    add x10, x10, _anvil_job@PAGEOFF
    ldr x15, [x10, #24]                 // chunks
    cmp x14, x15
    b.hs Lrc_done
    add x16, x11, #1
    mov x17, x11
    casal x17, x16, [x9]                // claim chunk x14 if the word is unchanged
    cmp x17, x11
    b.ne Lrc_loop
    ldp x12, x13, [x10]                 // fn, total
    ldr x15, [x10, #16]                 // grain
    mul x0, x14, x15
    add x1, x0, x15
    cmp x1, x13
    csel x1, x1, x13, lo
    blr x12
    adrp x9, _anvil_job_done@PAGE
    add x9, x9, _anvil_job_done@PAGEOFF
    mov x10, #1
    ldaddl x10, x11, [x9]               // release: this chunk's writes are visible
    sev
    b Lrc_loop
Lrc_done:
    ldp x29, x30, [sp], #32
    ret

// void *_anvil_rt_worker(void *) — wait for a new generation, help with that job, repeat.
// Waiting: spin on `wfe` for ~300 µs (back-to-back parallel kernels dispatch instantly), then
// sleep with usleep, 50 µs at first and twice as long each time up to 4 ms, so a worker idle
// in a long-running process (a server calling anvil.function) costs almost no CPU. A job never
// waits for a sleeping worker: the thread that posts it runs chunks too, and the worker joins
// when it wakes.
    .p2align 2
_anvil_rt_worker:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    str xzr, [sp, #16]                  // last generation seen
    mrs x9, cntfrq_el0
    mov x10, #3333
    udiv x9, x9, x10
    str x9, [sp, #32]                   // spin budget in timer ticks (~300 µs)
Lw_wait:
    mrs x9, cntvct_el0
    str x9, [sp, #24]                   // start of this wait
    mov x9, #50
    str x9, [sp, #40]                   // the first sleep: 50 µs
Lw_spin:
    adrp x9, _anvil_job_word@PAGE
    add x9, x9, _anvil_job_word@PAGEOFF
    ldaxr x10, [x9]
    lsr x10, x10, #32
    ldr x11, [sp, #16]
    cmp x10, x11
    b.ne Lw_go
    mrs x12, cntvct_el0
    ldr x13, [sp, #24]
    sub x12, x12, x13
    ldr x13, [sp, #32]
    cmp x12, x13
    b.hi Lw_doze
    wfe
    b Lw_spin
Lw_doze:
    clrex
    ldr x0, [sp, #40]
    bl _usleep
    ldr x9, [sp, #40]
    lsl x9, x9, #1
    mov x10, #4000
    cmp x9, x10
    csel x9, x9, x10, lo
    str x9, [sp, #40]                   // the next sleep: twice as long, at most 4 ms
    adrp x9, _anvil_job_word@PAGE
    add x9, x9, _anvil_job_word@PAGEOFF
    ldar x10, [x9]
    lsr x10, x10, #32
    ldr x11, [sp, #16]
    cmp x10, x11
    b.eq Lw_doze
Lw_go:
    clrex
    str x10, [sp, #16]
    mov x0, x10
    bl Lrt_run_chunks
    b Lw_wait

// ---------------------------------------------------------------------------
// void _anvil_rt_clock(float *out) — seconds since start (not counting time the Mac slept)
    .p2align 2
_anvil_rt_clock:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]
    mov x19, x0
    mov w0, #8                          // CLOCK_UPTIME_RAW
    bl _clock_gettime_nsec_np
    adrp x9, _anvil_t0@PAGE
    ldr x9, [x9, _anvil_t0@PAGEOFF]
    sub x0, x0, x9
    ucvtf d0, x0
    movz x9, #0x41cd, lsl #48           // 1e9 as a double: 0x41CDCD6500000000
    movk x9, #0xcd65, lsl #32
    fmov d1, x9
    fdiv d0, d0, d1
    fcvt s0, d0
    str s0, [x19]
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_seed(uint32 s)
    .p2align 2
_anvil_rt_seed:
    adrp x9, _anvil_seed@PAGE
    str w0, [x9, _anvil_seed@PAGEOFF]
    adrp x9, _anvil_stream@PAGE
    str wzr, [x9, _anvil_stream@PAGEOFF]
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_iota(int32 *buf, int64 n) — buf[i] = i
    .p2align 2
_anvil_rt_iota:
    mov x9, #0
Liota_loop:
    cmp x9, x1
    b.ge Liota_done
    str w9, [x0, x9, lsl #2]
    add x9, x9, #1
    b Liota_loop
Liota_done:
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_shuffle(int32 *perm, int64 n) — Fisher–Yates with the counter-based hash
//   stream = _anvil_stream++ ; K = stream*0x85EBCA77 + 7*0xC2B2AE3D + 0x632BE5AB + seed
//   for k in 0 .. n-2:  i = n-1-k ; h = lowbias32²(k*0x9E3779B1 + K) ; j = h % (i+1) ; swap
    .p2align 2
_anvil_rt_shuffle:
    cmp x1, #1
    b.le Lsh_done
    adrp x9, _anvil_stream@PAGE
    add x9, x9, _anvil_stream@PAGEOFF
    ldr w10, [x9]
    add w11, w10, #1
    str w11, [x9]
    movz w12, #0xca77
    movk w12, #0x85eb, lsl #16
    mul w10, w10, w12
    movz w12, #0xa956                   // 7*0xC2B2AE3D + 0x632BE5AB mod 2^32 = 0xB60EA956
    movk w12, #0xb60e, lsl #16
    add w10, w10, w12
    adrp x9, _anvil_seed@PAGE
    ldr w12, [x9, _anvil_seed@PAGEOFF]
    add w10, w10, w12                   // K
    movz w13, #0x79b1
    movk w13, #0x9e37, lsl #16          // golden ratio
    movz w14, #0x352d
    movk w14, #0x7feb, lsl #16
    movz w15, #0xa68b
    movk w15, #0x846c, lsl #16
    mov x2, #0                          // k
    sub x3, x1, #1                      // n - 1
Lsh_loop:
    cmp x2, x3
    b.ge Lsh_done
    madd w4, w2, w13, w10
    eor w4, w4, w4, lsr #16
    mul w4, w4, w14
    eor w4, w4, w4, lsr #15
    mul w4, w4, w15
    eor w4, w4, w4, lsr #16
    eor w4, w4, w4, lsr #16
    mul w4, w4, w14
    eor w4, w4, w4, lsr #15
    mul w4, w4, w15
    eor w4, w4, w4, lsr #16
    sub x5, x3, x2                      // i
    add x6, x5, #1
    udiv x7, x4, x6
    msub x7, x7, x6, x4                 // j = h % (i+1)
    ldr w8, [x0, x5, lsl #2]
    ldr w9, [x0, x7, lsl #2]
    str w9, [x0, x5, lsl #2]
    str w8, [x0, x7, lsl #2]
    add x2, x2, #1
    b Lsh_loop
Lsh_done:
    ret

// ---------------------------------------------------------------------------
// noreturn _anvil_rt_fail(const char *msg)
// noreturn _anvil_rt_oob(const char *msg, int64 value)
    .p2align 2
_anvil_rt_fail:
    mov x1, #0
    adrp x2, Lrt_fail_fmt@PAGE
    add x2, x2, Lrt_fail_fmt@PAGEOFF
    b Lrt_die
    .p2align 2
_anvil_rt_oob:
    adrp x2, Lrt_oob_fmt@PAGE
    add x2, x2, Lrt_oob_fmt@PAGEOFF
Lrt_die:
    mov x9, sp                          // keep the stack 16-byte aligned regardless of caller
    and x9, x9, #-16
    mov sp, x9
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x0, x1, [sp, #16]
    str x2, [sp, #32]
    mov x0, #0
    bl _fflush
    adrp x8, ___stderrp@GOTPAGE
    ldr x8, [x8, ___stderrp@GOTPAGEOFF]
    ldr x0, [x8]
    ldr x1, [sp, #32]
    ldp x9, x10, [sp, #16]
    sub sp, sp, #16
    stp x9, x10, [sp]
    bl _fprintf
    mov w0, #1
    bl _exit

// ---------------------------------------------------------------------------
// void _anvil_rt_check(const float *p, int64 n, const int64 *rec, int64 mode)
// `anvil run --check`: called after each kernel for each f32 tensor it wrote. Stops at the first
// nan (mode 1), or the first nan or infinity (mode 2), in p[0:n], and says where it came from.
// rec: {name, dims, rank, kernel description, number of inputs, (data, numel, name) per input}
    .p2align 2
_anvil_rt_check:
    mov x9, #0
    and x10, x1, #-4                    // whole vectors first
    cmp x3, #2
    b.ge Lchk_vinf
Lchk_vnan:
    cmp x9, x10
    b.ge Lchk_tail
    add x12, x0, x9, lsl #2
    ldr q0, [x12]
    fcmeq v1.4s, v0.4s, v0.4s           // all ones where not nan
    uminv s1, v1.4s
    fmov w11, s1
    cbz w11, Lchk_tail                  // a nan in these four: find it one at a time
    add x9, x9, #4
    b Lchk_vnan
Lchk_vinf:
    cmp x9, x10
    b.ge Lchk_tail
    add x12, x0, x9, lsl #2
    ldr q0, [x12]
    fsub v1.4s, v0.4s, v0.4s            // 0 where finite, nan where not
    fcmeq v1.4s, v1.4s, #0.0
    uminv s1, v1.4s
    fmov w11, s1
    cbz w11, Lchk_tail
    add x9, x9, #4
    b Lchk_vinf
Lchk_tail:
    cmp x9, x1
    b.ge Lchk_ok
    ldr s0, [x0, x9, lsl #2]
    fsub s1, s0, s0
    fcmp s1, s1
    b.vc Lchk_next                      // finite
    fcmp s0, s0
    b.vs Lchk_bad                       // nan
    cmp x3, #2
    b.ge Lchk_bad                       // an infinity, and infinities count
Lchk_next:
    add x9, x9, #1
    b Lchk_tail
Lchk_ok:
    ret
Lchk_bad:
    stp x29, x30, [sp, #-144]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]             // [sp, #64, #128): the element's index, one per dimension
    mov x19, x2                         // the record
    mov x20, x9                         // the flat index
    adrp x22, Lchk_s_nan@PAGE           // what was found
    add x22, x22, Lchk_s_nan@PAGEOFF
    fcmp s0, s0
    b.vs 1f
    adrp x22, Lchk_s_inf@PAGE
    add x22, x22, Lchk_s_inf@PAGEOFF
    fcmp s0, #0.0
    b.gt 1f
    adrp x22, Lchk_s_ninf@PAGE
    add x22, x22, Lchk_s_ninf@PAGEOFF
1:  mov x0, #0
    bl _fflush
    adrp x8, ___stderrp@GOTPAGE
    ldr x8, [x8, ___stderrp@GOTPAGEOFF]
    ldr x21, [x8]                       // stderr
    mov x0, x21
    adrp x1, Lchk_f_head@PAGE
    add x1, x1, Lchk_f_head@PAGEOFF
    ldr x9, [x19]                       // the tensor's name and shape
    sub sp, sp, #16
    stp x22, x9, [sp]
    bl _fprintf
    add sp, sp, #16
    // the flat index as one index per dimension, last dimension first
    ldr x23, [x19, #16]                 // rank
    ldr x24, [x19, #8]                  // dims
    cbz x23, Lchk_desc
    mov x9, x20
    mov x10, x23
2:  sub x10, x10, #1
    ldr x11, [x24, x10, lsl #3]
    udiv x12, x9, x11
    msub x13, x12, x11, x9
    add x14, sp, #64
    str x13, [x14, x10, lsl #3]
    mov x9, x12
    cbnz x10, 2b
    mov x0, x21
    adrp x1, Lchk_f_open@PAGE
    add x1, x1, Lchk_f_open@PAGEOFF
    bl _fprintf
    mov x20, #0
3:  cmp x20, x23
    b.ge 4f
    mov x0, x21
    adrp x1, Lchk_f_index@PAGE
    add x1, x1, Lchk_f_index@PAGEOFF
    cbz x20, 5f
    adrp x1, Lchk_f_index2@PAGE
    add x1, x1, Lchk_f_index2@PAGEOFF
5:  add x14, sp, #64
    ldr x9, [x14, x20, lsl #3]
    sub sp, sp, #16
    str x9, [sp]
    bl _fprintf
    add sp, sp, #16
    add x20, x20, #1
    b 3b
4:  mov x0, x21
    adrp x1, Lchk_f_close@PAGE
    add x1, x1, Lchk_f_close@PAGEOFF
    bl _fprintf
Lchk_desc:
    mov x0, x21
    adrp x1, Lchk_f_desc@PAGE
    add x1, x1, Lchk_f_desc@PAGEOFF
    ldr x9, [x19, #24]
    sub sp, sp, #16
    str x9, [sp]
    bl _fprintf
    add sp, sp, #16
    // the kernel's inputs: which of them already held a nan or an infinity?
    ldr x23, [x19, #32]                 // number of inputs
    add x24, x19, #40
    mov x20, #0                         // inputs that did
6:  cbz x23, Lchk_end
    ldr x9, [x24]                       // data
    ldr x10, [x24, #8]                  // numel
    mov x11, #0                         // nans
    mov x12, #0                         // infinities
    mov x13, #0
7:  cmp x13, x10
    b.ge 8f
    ldr s0, [x9, x13, lsl #2]
    fsub s1, s0, s0
    fcmp s1, s1
    b.vc 9f
    fcmp s0, s0
    cinc x11, x11, vs
    cinc x12, x12, vc
9:  add x13, x13, #1
    b 7b
8:  orr x13, x11, x12
    cbz x13, 10f
    add x20, x20, #1
    mov x0, x21
    adrp x1, Lchk_f_input@PAGE
    add x1, x1, Lchk_f_input@PAGEOFF
    ldr x9, [x24, #16]
    sub sp, sp, #32
    str x9, [sp]
    stp x11, x12, [sp, #8]
    bl _fprintf
    add sp, sp, #32
10: add x24, x24, #24
    sub x23, x23, #1
    b 6b
Lchk_end:
    mov x0, x21
    adrp x1, Lchk_f_made@PAGE
    add x1, x1, Lchk_f_made@PAGEOFF
    cbz x20, 11f
    adrp x1, Lchk_f_passed@PAGE
    add x1, x1, Lchk_f_passed@PAGEOFF
11: bl _fprintf
    mov w0, #1
    bl _exit

// ---------------------------------------------------------------------------
// void _anvil_rt_load_idx(const char *path, void *dst, int64 numel, int64 header,
//                       int64 dtype /*0 = f32, 1 = i32*/, const int64 *dims, int64 ndim)
// Reads an (optionally gzipped) IDX file of unsigned bytes, checks its header, and
// widens the payload in place into f32 or i32.
    .p2align 2
_anvil_rt_load_idx:
    stp x29, x30, [sp, #-128]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    // header buffer at [sp, #80 .. #128) (48 bytes = up to 11 dims)
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
    adrp x1, Lrt_rb@PAGE
    add x1, x1, Lrt_rb@PAGEOFF
    bl _gzopen
    cbz x0, Lidx_open_fail
    mov x26, x0
    // header
    mov x0, x26
    add x1, sp, #80
    mov x2, x22
    bl _gzread
    cmp x0, x22
    b.ne Lidx_bad
    ldrb w9, [sp, #80]
    cbnz w9, Lidx_bad
    ldrb w9, [sp, #81]
    cbnz w9, Lidx_bad
    ldrb w9, [sp, #83]
    cmp x9, x25
    b.ne Lidx_bad
    mov x10, #0
Lidx_dims:
    cmp x10, x25
    b.ge Lidx_dims_ok
    add x11, sp, #84
    ldr w12, [x11, x10, lsl #2]
    rev w12, w12
    ldr x13, [x24, x10, lsl #3]
    cmp x12, x13
    b.ne Lidx_bad
    add x10, x10, #1
    b Lidx_dims
Lidx_dims_ok:
    // payload goes to the last numel bytes of dst, then is widened forward
    add x22, x20, x21, lsl #1
    add x22, x22, x21                   // x22 = dst + 3*numel  (byte cursor)
    mov x23, x23                        // dtype stays in x23
    mov x24, x21                        // remaining
    mov x25, x22                        // write cursor
Lidx_read:
    cbz x24, Lidx_read_done
    mov x2, #0x40000000
    cmp x24, x2
    csel x2, x24, x2, lo
    mov x0, x26
    mov x1, x25
    bl _gzread
    cmp w0, #0
    b.le Lidx_bad
    sxtw x0, w0
    add x25, x25, x0
    sub x24, x24, x0
    b Lidx_read
Lidx_read_done:
    mov x0, x26
    bl _gzclose
    mov x10, #0
Lidx_widen:
    cmp x10, x21
    b.ge Lidx_done
    ldrb w11, [x22, x10]
    cbnz x23, Lidx_int
    ucvtf s0, w11
    str s0, [x20, x10, lsl #2]
    add x10, x10, #1
    b Lidx_widen
Lidx_int:
    str w11, [x20, x10, lsl #2]
    add x10, x10, #1
    b Lidx_widen
Lidx_done:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x29, x30, [sp], #128
    ret
Lidx_open_fail:
    adrp x0, Lrt_open_msg@PAGE
    add x0, x0, Lrt_open_msg@PAGEOFF
    mov x1, x19
    b Lidx_die
Lidx_bad:
    adrp x0, Lrt_bad_msg@PAGE
    add x0, x0, Lrt_bad_msg@PAGEOFF
    mov x1, x19
Lidx_die:
    // fprintf(stderr, msg, path); exit(1)
    stp x0, x1, [sp, #80]
    mov x0, #0
    bl _fflush
    adrp x8, ___stderrp@GOTPAGE
    ldr x8, [x8, ___stderrp@GOTPAGEOFF]
    ldr x0, [x8]
    ldp x1, x9, [sp, #80]
    sub sp, sp, #16
    str x9, [sp]
    bl _fprintf
    mov w0, #1
    bl _exit

// ---------------------------------------------------------------------------
// void _anvil_rt_load_bytes(const char *path, int32_t *out, int64 n)
// A file of exactly n bytes, each widened to an i32 (read into the buffer, then spread out from
// the end so that no byte is overwritten before it is read).
    .p2align 2
_anvil_rt_load_bytes:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    adrp x1, Lrt_rb@PAGE
    add x1, x1, Lrt_rb@PAGEOFF
    bl _fopen
    cbz x0, Lbytes_open_fail
    mov x22, x0
    mov x0, x20
    mov x1, #1
    add x2, x21, #1                     // one more than expected: a longer file is caught too
    mov x3, x22
    bl _fread
    mov x9, x0
    mov x0, x22
    stp x9, xzr, [sp, #-16]!
    bl _fclose
    ldp x9, xzr, [sp], #16
    cmp x9, x21
    b.ne Lbytes_bad
    mov x9, x21
Lbytes_widen:
    cbz x9, Lbytes_done
    sub x9, x9, #1
    ldrb w10, [x20, x9]
    str w10, [x20, x9, lsl #2]
    b Lbytes_widen
Lbytes_done:
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret
Lbytes_open_fail:
    adrp x0, Lrt_open_msg@PAGE
    add x0, x0, Lrt_open_msg@PAGEOFF
    mov x1, x19
    b Lidx_die
Lbytes_bad:
    adrp x0, Lrt_bad_msg@PAGE
    add x0, x0, Lrt_bad_msg@PAGEOFF
    mov x1, x19
    b Lidx_die

// ---------------------------------------------------------------------------
// void _anvil_rt_load_npy(const char *path, void *dst, int64 n, int64 offset, int64 kind, int64 size)
// The n values of a NumPy file, from byte `offset`, converted to 32 bits: kind 0 f32, 1 f64, 2 f16
// (to f32); 3 32-bit, 4 64-bit, 5 unsigned 8-bit, 6 signed 8-bit, 7 unsigned 16-bit, 8 signed
// 16-bit integers (to i32). size: bytes per value in the file. The file must hold exactly n.
    .p2align 2
_anvil_rt_load_npy:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0                         // path
    mov x20, x1                         // dst
    mov x21, x2                         // n
    mov x23, x3                         // offset
    mov x24, x4                         // kind
    mul x22, x2, x5                     // bytes of data
    adrp x1, Lrt_rb@PAGE
    add x1, x1, Lrt_rb@PAGEOFF
    bl _fopen
    cbz x0, Lnpy_open_fail
    mov x25, x0
    mov x1, x23
    mov w2, #0                          // SEEK_SET
    bl _fseek
    add x0, x22, #1
    bl _malloc
    mov x26, x0
    mov x1, #1
    add x2, x22, #1                     // one more than expected: a longer file is caught too
    mov x3, x25
    bl _fread
    mov x23, x0
    mov x0, x25
    bl _fclose
    cmp x23, x22
    b.ne Lnpy_bad
    mov x9, #0
Lnpy_loop:
    cmp x9, x21
    b.ge Lnpy_done
    cmp x24, #1
    b.eq Lnpy_f64
    cmp x24, #2
    b.eq Lnpy_f16
    cmp x24, #4
    b.eq Lnpy_i64
    cmp x24, #5
    b.eq Lnpy_u8
    cmp x24, #6
    b.eq Lnpy_i8
    cmp x24, #7
    b.eq Lnpy_u16
    cmp x24, #8
    b.eq Lnpy_i16
    ldr w10, [x26, x9, lsl #2]          // f32 and 32-bit integers: as they are
    b Lnpy_store
Lnpy_f64:
    ldr d0, [x26, x9, lsl #3]
    fcvt s0, d0
    fmov w10, s0
    b Lnpy_store
Lnpy_f16:
    ldr h0, [x26, x9, lsl #1]
    fcvt s0, h0
    fmov w10, s0
    b Lnpy_store
Lnpy_i64:
    ldr x10, [x26, x9, lsl #3]
    b Lnpy_store
Lnpy_u8:
    ldrb w10, [x26, x9]
    b Lnpy_store
Lnpy_i8:
    ldrsb w10, [x26, x9]
    b Lnpy_store
Lnpy_u16:
    ldrh w10, [x26, x9, lsl #1]
    b Lnpy_store
Lnpy_i16:
    ldrsh w10, [x26, x9, lsl #1]
Lnpy_store:
    str w10, [x20, x9, lsl #2]
    add x9, x9, #1
    b Lnpy_loop
Lnpy_done:
    mov x0, x26
    bl _free
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
Lnpy_open_fail:
    adrp x0, Lrt_open_msg@PAGE
    add x0, x0, Lrt_open_msg@PAGEOFF
    mov x1, x19
    b Lidx_die
Lnpy_bad:
    adrp x0, Lrt_bad_msg@PAGE
    add x0, x0, Lrt_bad_msg@PAGEOFF
    mov x1, x19
    b Lidx_die

// ---------------------------------------------------------------------------
// void _anvil_rt_save_npy(const char *path, const uint8_t *header, int64 hlen, const void *data, int64 bytes)
// A NumPy file: its header (made by the compiler) and the data. A file that cannot be written is
// reported, and the program goes on.
    .p2align 2
_anvil_rt_save_npy:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    adrp x1, Lrt_wb@PAGE
    add x1, x1, Lrt_wb@PAGEOFF
    bl _fopen
    cbz x0, Lsnpy_fail
    mov x24, x0
    mov x0, x20
    mov x1, #1
    mov x2, x21
    mov x3, x24
    bl _fwrite
    mov x0, x22
    mov x1, #1
    mov x2, x23
    mov x3, x24
    bl _fwrite
    mov x0, x24
    bl _fclose
    b Lsnpy_end
Lsnpy_fail:
    mov x0, #0
    bl _fflush
    adrp x8, ___stderrp@GOTPAGE
    ldr x8, [x8, ___stderrp@GOTPAGEOFF]
    ldr x0, [x8]
    adrp x1, Lsnpy_msg@PAGE
    add x1, x1, Lsnpy_msg@PAGEOFF
    sub sp, sp, #16
    str x19, [sp]
    bl _fprintf
    add sp, sp, #16
Lsnpy_end:
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_print_chars(const int32_t *codes, int64 n): the low byte of each, to stdout
    .p2align 2
_anvil_rt_print_chars:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x19, x0
    mov x20, x1
    adrp x8, ___stdoutp@GOTPAGE
    ldr x8, [x8, ___stdoutp@GOTPAGEOFF]
    ldr x21, [x8]
    mov x22, #0
Lchars_loop:
    cmp x22, x20
    b.ge Lchars_done
    ldr w0, [x19, x22, lsl #2]
    and w0, w0, #0xff
    mov x1, x21
    bl _fputc
    add x22, x22, #1
    b Lchars_loop
Lchars_done:
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_load_csv(const char *path, float *out, int64 n, int64 skip_lines)
// The first n numbers in a text file after its first skip_lines lines. The compiler has checked
// that the file holds a table of numbers of this size (and which lines are a header).
    .p2align 2
_anvil_rt_load_csv:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    str x25, [sp, #64]
    mov x19, x0                         // path
    mov x20, x1                         // out
    mov x21, x2                         // numbers wanted
    mov x22, x3                         // lines to skip
    adrp x1, Lrt_rb@PAGE
    add x1, x1, Lrt_rb@PAGEOFF
    bl _fopen
    cbz x0, Lcsv_open_fail
    mov x23, x0                         // FILE *
    mov x1, #0
    mov w2, #2                          // SEEK_END
    bl _fseek
    mov x0, x23
    bl _ftell
    mov x24, x0                         // file size
    mov x0, x23
    mov x1, #0
    mov w2, #0                          // SEEK_SET
    bl _fseek
    add x0, x24, #1
    bl _malloc
    cbz x0, Lcsv_bad
    mov x25, x0                         // the whole file, NUL-terminated
    mov x1, #1
    mov x2, x24
    mov x3, x23
    bl _fread
    strb wzr, [x25, x0]
    mov x0, x23
    bl _fclose
    mov x24, x25                        // scan position
Lcsv_skip:
    cbz x22, Lcsv_scan_start
    ldrb w9, [x24]
    cbz w9, Lcsv_bad
    add x24, x24, #1
    cmp w9, #10
    b.ne Lcsv_skip
    sub x22, x22, #1
    b Lcsv_skip
Lcsv_scan_start:
    mov x23, #0                         // numbers read
Lcsv_scan:
    cmp x23, x21
    b.ge Lcsv_done
    ldrb w9, [x24]
    cbz w9, Lcsv_bad                    // the file has fewer numbers than it did at compile time
    mov x10, x24                        // a number: [+-] digit … or [+-] . digit …
    cmp w9, #'+'
    b.eq Lcsv_sign
    cmp w9, #'-'
    b.ne Lcsv_point
Lcsv_sign:
    add x10, x10, #1
    ldrb w9, [x10]
Lcsv_point:
    cmp w9, #'.'
    b.ne Lcsv_digit
    ldrb w9, [x10, #1]
Lcsv_digit:
    sub w9, w9, #'0'
    cmp w9, #9
    b.hi Lcsv_next
    mov x0, x24
    add x1, sp, #72
    bl _strtof
    str s0, [x20, x23, lsl #2]
    add x23, x23, #1
    ldr x24, [sp, #72]                  // continue after the number
    b Lcsv_scan
Lcsv_next:
    add x24, x24, #1
    b Lcsv_scan
Lcsv_done:
    mov x0, x25
    bl _free
    ldr x25, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
Lcsv_open_fail:
    adrp x0, Lrt_open_msg@PAGE
    add x0, x0, Lrt_open_msg@PAGEOFF
    mov x1, x19
    b Lidx_die                          // (it only writes above our frame, then exits)
Lcsv_bad:
    adrp x0, Lrt_bad_msg@PAGE
    add x0, x0, Lrt_bad_msg@PAGEOFF
    mov x1, x19
    b Lidx_die

// ---------------------------------------------------------------------------
// void _anvil_rt_print_tensor(const void *data, int64 dtype, int64 rank, const int64 *dims)
// NumPy-style: nested brackets; summarized with "..." when there are > 1000 elements.
    .p2align 2
_anvil_rt_print_tensor:
    stp x29, x30, [sp, #-176]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    // strides live at [sp, #64 .. #128) (rank <= 8)
    mov x19, x0                         // data
    mov x20, x1                         // dtype
    mov x21, x2                         // rank
    mov x22, x3                         // dims
    cbnz x21, Lpt_ranked
    mov x0, #0
    bl Lpt_elem
    b Lpt_out
Lpt_ranked:
    // strides and numel
    mov x9, #1
    sub x10, x21, #1
Lpt_strides:
    add x11, sp, #64
    str x9, [x11, x10, lsl #3]
    ldr x12, [x22, x10, lsl #3]
    mul x9, x9, x12
    subs x10, x10, #1
    b.ge Lpt_strides
    cmp x9, #1000
    cset x23, gt                        // summarize?
    cbz x9, Lpt_empty
    add x24, sp, #64                    // strides pointer
    mov x0, #0
    mov x1, #0
    bl Lpt_rec
    b Lpt_out
Lpt_empty:
    mov w0, #'['
    bl _putchar
    mov w0, #']'
    bl _putchar
Lpt_out:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x29, x30, [sp], #176
    ret

// Lpt_elem(x0 = element index): print one element (uses x19 data, x20 dtype)
    .p2align 2
Lpt_elem:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    cbnz x20, Lpt_elem_int
    ldr s0, [x19, x0, lsl #2]
    bl _anvil_rt_print_f32
    ldp x29, x30, [sp], #32
    ret
Lpt_elem_int:
    ldrsw x9, [x19, x0, lsl #2]
    sub sp, sp, #16
    str x9, [sp]
    adrp x0, Lrt_d@PAGE
    add x0, x0, Lrt_d@PAGEOFF
    bl _printf
    add sp, sp, #16
    ldp x29, x30, [sp], #32
    ret

// Lpt_rec(x0 = offset, x1 = dim): print the sub-tensor at offset along dims[dim:]
// uses x19 data, x20 dtype, x21 rank, x22 dims, x23 summarize, x24 strides
    .p2align 2
Lpt_rec:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x25, x26, [sp, #16]
    stp x27, x28, [sp, #32]
    str x19, [sp, #48]                  // (padding slot; x19 is preserved anyway)
    mov x25, x0                         // offset
    mov x26, x1                         // dim
    ldr x27, [x22, x26, lsl #3]         // n
    mov w0, #'['
    bl _putchar
    // visible = (summarize && n > 6) ? 7 : n   -> keep in [sp, #56]
    mov x9, x27
    cbz x23, Lpt_vis
    cmp x27, #6
    b.le Lpt_vis
    mov x9, #7
Lpt_vis:
    str x9, [sp, #56]
    mov x28, #0                         // k
Lpt_loop:
    ldr x9, [sp, #56]
    cmp x28, x9
    b.ge Lpt_close
    cbz x28, Lpt_item
    // separator
    sub x10, x21, #1
    cmp x26, x10
    b.ne Lpt_sep_nl
    mov w0, #','
    bl _putchar
    mov w0, #' '
    bl _putchar
    b Lpt_item
Lpt_sep_nl:
    mov w0, #','
    bl _putchar
    mov w0, #'\n'
    bl _putchar
    add x9, x26, #1
    str x9, [sp, #64]
Lpt_indent:
    ldr x9, [sp, #64]
    cbz x9, Lpt_item
    sub x9, x9, #1
    str x9, [sp, #64]
    mov w0, #' '
    bl _putchar
    b Lpt_indent
Lpt_item:
    // idx = k, or the summarized mapping 0,1,2,(...),n-3,n-2,n-1
    mov x10, x28
    ldr x9, [sp, #56]
    cmp x9, x27
    b.eq Lpt_have_idx                   // not summarized along this dim
    cmp x28, #3
    b.lt Lpt_have_idx
    b.gt Lpt_tail_idx
    // ellipsis
    adrp x0, Lrt_ellipsis@PAGE
    add x0, x0, Lrt_ellipsis@PAGEOFF
    bl _printf
    b Lpt_next
Lpt_tail_idx:
    sub x10, x27, #7
    add x10, x10, x28
Lpt_have_idx:
    sub x11, x21, #1
    cmp x26, x11
    b.ne Lpt_recurse
    add x0, x25, x10
    bl Lpt_elem
    b Lpt_next
Lpt_recurse:
    ldr x12, [x24, x26, lsl #3]
    madd x0, x10, x12, x25
    add x1, x26, #1
    bl Lpt_rec
Lpt_next:
    add x28, x28, #1
    b Lpt_loop
Lpt_close:
    mov w0, #']'
    bl _putchar
    ldp x25, x26, [sp, #16]
    ldp x27, x28, [sp, #32]
    ldp x29, x30, [sp], #80
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_show(const i32 *grid, i64 rows, i64 cols, const char **glyphs, i64 n)
// Draws each value v as glyphs[v] (glyphs[n] is "?" for out-of-range values).
    .p2align 2
_anvil_rt_show:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    str x25, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    adrp x8, ___stdoutp@GOTPAGE
    ldr x8, [x8, ___stdoutp@GOTPAGEOFF]
    ldr x25, [x8]                       // FILE *stdout
    mov x24, #0                         // element index
    mul x20, x20, x21                   // total elements
Lshow_loop:
    cmp x24, x20
    b.ge Lshow_done
    ldrsw x9, [x19, x24, lsl #2]
    cmp x9, #0
    csel x9, x23, x9, lt
    cmp x9, x23
    csel x9, x23, x9, gt
    ldr x0, [x22, x9, lsl #3]
    mov x1, x25
    bl _fputs
    add x24, x24, #1
    udiv x9, x24, x21
    msub x9, x9, x21, x24               // index % cols
    cbnz x9, Lshow_loop
    mov w0, #'\n'
    mov x1, x25
    bl _fputc
    b Lshow_loop
Lshow_done:
    ldr x25, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

// ---------------------------------------------------------------------------
// Checkpoints (save / load). The file holds "EINW", u32 version 1, u64 n, n u64 element counts,
// then every tensor's elements (4 bytes each). tab: n entries {void *data, u64 count}.

// void _anvil_rt_save(const char *path, const entry *tab, int64 n)
    .p2align 2
_anvil_rt_save:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    str x23, [sp, #48]
    mov x19, x0                         // path
    mov x20, x1                         // table
    mov x21, x2                         // n
    adrp x1, Lrt_wb@PAGE
    add x1, x1, Lrt_wb@PAGEOFF
    bl _fopen
    cbz x0, Lsave_fail
    mov x22, x0                         // FILE *
    movz x9, #0x4945                    // "EINW", version 1
    movk x9, #0x574e, lsl #16
    movk x9, #1, lsl #32
    str x9, [sp, #56]
    add x0, sp, #56
    mov x1, #8
    mov x2, #1
    mov x3, x22
    bl _fwrite
    str x21, [sp, #56]
    add x0, sp, #56
    mov x1, #8
    mov x2, #1
    mov x3, x22
    bl _fwrite
    mov x23, #0
Lsave_sizes:
    cmp x23, x21
    b.ge Lsave_data_start
    add x0, x20, x23, lsl #4
    add x0, x0, #8                      // &tab[i].count
    mov x1, #8
    mov x2, #1
    mov x3, x22
    bl _fwrite
    add x23, x23, #1
    b Lsave_sizes
Lsave_data_start:
    mov x23, #0
Lsave_data:
    cmp x23, x21
    b.ge Lsave_close
    add x9, x20, x23, lsl #4
    ldp x0, x2, [x9]                    // data, count
    mov x1, #4
    mov x3, x22
    bl _fwrite
    add x23, x23, #1
    b Lsave_data
Lsave_close:
    mov x0, x22
    bl _fclose
    b Lsave_ret
Lsave_fail:
    adrp x8, ___stderrp@GOTPAGE
    ldr x8, [x8, ___stderrp@GOTPAGEOFF]
    ldr x0, [x8]
    adrp x1, Lrt_cant_write@PAGE
    add x1, x1, Lrt_cant_write@PAGEOFF
    sub sp, sp, #16
    str x19, [sp]                       // the variadic argument: the path
    bl _fprintf
    add sp, sp, #16
Lsave_ret:
    ldr x23, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// int _anvil_rt_load(const char *path, const entry *tab, int64 n)
// 1 if the tensors were read; 0 if the file is missing or holds tensors of other sizes (and then
// nothing is changed).
    .p2align 2
_anvil_rt_load:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x20, x1                         // table
    mov x21, x2                         // n
    mov x19, #0                         // the result
    adrp x1, Lrt_rb@PAGE
    add x1, x1, Lrt_rb@PAGEOFF
    bl _fopen
    cbz x0, Lload_ret
    mov x22, x0                         // FILE *
    bl Lload_word
    movz x10, #0x4945
    movk x10, #0x574e, lsl #16
    movk x10, #1, lsl #32
    cmp x9, x10
    b.ne Lload_close
    bl Lload_word
    cmp x9, x21
    b.ne Lload_close
    mov x23, #0
    mov x24, #0                         // elements in all
Lload_sizes:
    cmp x23, x21
    b.ge Lload_check_size
    bl Lload_word
    add x10, x20, x23, lsl #4
    ldr x10, [x10, #8]
    cmp x9, x10
    b.ne Lload_close
    add x24, x24, x10
    add x23, x23, #1
    b Lload_sizes
Lload_check_size:                       // the file must be exactly that long
    mov x0, x22
    mov x1, #0
    mov w2, #2                          // SEEK_END
    bl _fseek
    mov x0, x22
    bl _ftell
    lsl x9, x21, #3
    add x9, x9, #16                     // where the data starts
    add x10, x9, x24, lsl #2
    cmp x0, x10
    b.ne Lload_close
    mov x0, x22
    mov x1, x9
    mov w2, #0                          // SEEK_SET
    bl _fseek
    mov x23, #0
Lload_data:
    cmp x23, x21
    b.ge Lload_done
    add x9, x20, x23, lsl #4
    ldp x0, x2, [x9]                    // data, count
    mov x24, x2
    mov x1, #4
    mov x3, x22
    bl _fread
    cmp x0, x24
    b.ne Lload_close
    add x23, x23, #1
    b Lload_data
Lload_done:
    mov x19, #1
Lload_close:
    mov x0, x22
    bl _fclose
Lload_ret:
    mov w0, w19
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret
Lload_word:                             // read one u64 into x9, or give up on the file
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    add x0, sp, #80                     // the caller's scratch slot (its frame starts 16 bytes up)
    mov x1, #8
    mov x2, #1
    mov x3, x22
    bl _fread
    ldp x29, x30, [sp], #16
    cmp x0, #1
    b.ne Lload_bad_word
    ldr x9, [sp, #64]
    ret
Lload_bad_word:
    mov x9, #-1                         // matches no header or size
    ret

// ---------------------------------------------------------------------------
// int _anvil_rt_input(float *out)
// Read a line from stdin and store the first decimal number in it (or nan) to *out, and return
// 0. At the end of the input, return 1: the program then ends, as if the script had.
    .p2align 2
_anvil_rt_input:
    stp x29, x30, [sp, #-304]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0                         // out
    mov x0, #0
    bl _fflush                          // show the prompt
    add x0, sp, #48                     // the line: sp+48 .. sp+304
    mov w1, #256
    adrp x8, ___stdinp@GOTPAGE
    ldr x8, [x8, ___stdinp@GOTPAGEOFF]
    ldr x2, [x8]
    bl _fgets
    cbz x0, Lin_eof
    add x20, sp, #48                    // where a number might start
Lin_scan:
    ldrb w9, [x20]
    cbz w9, Lin_none
    mov x10, x20                        // a number: [+-] digit … or [+-] . digit …
    cmp w9, #'+'
    b.eq Lin_sign
    cmp w9, #'-'
    b.ne Lin_point
Lin_sign:
    add x10, x10, #1
    ldrb w9, [x10]
Lin_point:
    cmp w9, #'.'
    b.ne Lin_digit
    ldrb w9, [x10, #1]
Lin_digit:
    sub w9, w9, #'0'
    cmp w9, #9
    b.hi Lin_next
    ldrb w9, [x10]                      // decimal only: "0x1f" reads as 0
    cmp w9, #'0'
    b.ne Lin_parse
    ldrb w9, [x10, #1]
    orr w9, w9, #0x20
    cmp w9, #'x'
    b.ne Lin_parse
    strb wzr, [x10, #1]
Lin_parse:
    mov x0, x20
    add x1, sp, #40                     // end pointer (unused)
    bl _strtof
    str s0, [x19]
    b Lin_ret
Lin_next:
    add x20, x20, #1
    b Lin_scan
Lin_none:
    movz w9, #0x7fc0, lsl #16           // nan
    str w9, [x19]
Lin_ret:
    mov w0, #0
Lin_out:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #304
    ret
Lin_eof:
    mov w0, #10
    bl _putchar                         // end the prompt's line
    mov w0, #1
    b Lin_out

// ---------------------------------------------------------------------------
// void _anvil_rt_nonzero(const int32_t *flags, int64 n, int32_t *out, int64 size)
// The indices of the nonzero flags, in order, into out[0:size]; the rest of out is -1.
    .p2align 2
_anvil_rt_nonzero:
    mov x9, #0                          // indices written
    mov x10, #0                         // flag index
Lnz_scan:
    cmp x10, x1
    b.ge Lnz_fill
    cmp x9, x3
    b.ge Lnz_fill
    ldr w11, [x0, x10, lsl #2]
    cbz w11, Lnz_next
    str w10, [x2, x9, lsl #2]
    add x9, x9, #1
Lnz_next:
    add x10, x10, #1
    b Lnz_scan
Lnz_fill:
    mov w11, #-1
Lnz_fill_loop:
    cmp x9, x3
    b.ge Lnz_done
    str w11, [x2, x9, lsl #2]
    add x9, x9, #1
    b Lnz_fill_loop
Lnz_done:
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_sleep(float seconds in s0)
    .p2align 2
_anvil_rt_sleep:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x0, #0
    bl _fflush                          // show everything printed so far before pausing
    movz w9, #0x4974, lsl #16           // 1e6 as f32 = 0x49742400
    movk w9, #0x2400
    fmov s1, w9
    fmul s0, s0, s1
    fcvtzu w0, s0
    bl _usleep
    ldp x29, x30, [sp], #16
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_print_f32(float v in s0): "%.4f", or "%.4e" when |v| < 1e-4 or |v| >= 1e6
    .p2align 2
_anvil_rt_print_f32:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    fcvt d0, s0
    fabs d1, d0
    adrp x0, Lrt_f4@PAGE
    add x0, x0, Lrt_f4@PAGEOFF
    fcmp d1, #0.0
    b.eq Lpf_print
    b.vs Lpf_print                      // NaN
    movz x9, #0x432d                    // 1e-4 = 0x3F1A36E2EB1C432D
    movk x9, #0xeb1c, lsl #16
    movk x9, #0x36e2, lsl #32
    movk x9, #0x3f1a, lsl #48
    fmov d2, x9
    fcmp d1, d2
    b.lt Lpf_sci
    movz x9, #0x412e, lsl #48           // 1e6 = 0x412E848000000000
    movk x9, #0x8480, lsl #32
    fmov d2, x9
    fcmp d1, d2
    b.lt Lpf_print
Lpf_sci:
    adrp x0, Lrt_e4@PAGE
    add x0, x0, Lrt_e4@PAGEOFF
Lpf_print:
    sub sp, sp, #16
    str d0, [sp]
    bl _printf
    add sp, sp, #16
    ldp x29, x30, [sp], #16
    ret

// ---------------------------------------------------------------------------
// void _anvil_rt_profile_report(const entry *tab, int64 n)   entry = {char *name, u64 ticks, u64 calls}
    .p2align 2
_anvil_rt_profile_report:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp d8, d9, [sp, #48]
    mov x19, x0
    mov x20, x1
    // total ticks -> d8, tick frequency -> d9
    mov x9, #0
    mov x10, #0
Lprof_sum:
    cmp x10, x20
    b.ge Lprof_sum_done
    mov x11, #24
    madd x12, x10, x11, x19
    ldr x13, [x12, #8]
    add x9, x9, x13
    add x10, x10, #1
    b Lprof_sum
Lprof_sum_done:
    ucvtf d8, x9
    mrs x9, cntfrq_el0
    ucvtf d9, x9
    adrp x0, Lrt_prof_head@PAGE
    add x0, x0, Lrt_prof_head@PAGEOFF
    bl _printf
    mov x21, #0
Lprof_loop:
    cmp x21, x20
    b.ge Lprof_done
    mov x11, #24
    madd x22, x21, x11, x19
    ldr x9, [x22, #8]                   // ticks
    ucvtf d0, x9
    fmov d1, #1.0
    fmax d2, d8, d1
    fdiv d1, d0, d2
    mov x10, #100
    ucvtf d2, x10
    fmul d1, d1, d2                     // percent
    fdiv d0, d0, d9
    mov x10, #1000
    ucvtf d2, x10
    fmul d0, d0, d2                     // milliseconds
    ldr x10, [x22, #16]                 // calls
    ldr x11, [x22]                      // name
    sub sp, sp, #32
    str d1, [sp]
    str d0, [sp, #8]
    str x10, [sp, #16]
    str x11, [sp, #24]
    adrp x0, Lrt_prof_row@PAGE
    add x0, x0, Lrt_prof_row@PAGEOFF
    bl _printf
    add sp, sp, #32
    add x21, x21, #1
    b Lprof_loop
Lprof_done:
    fdiv d0, d8, d9
    mov x10, #1000
    ucvtf d2, x10
    fmul d0, d0, d2
    sub sp, sp, #16
    str d0, [sp]
    adrp x0, Lrt_prof_total@PAGE
    add x0, x0, Lrt_prof_total@PAGEOFF
    bl _printf
    add sp, sp, #16
    ldp d8, d9, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// ---------------------------------------------------------------------------
    .section __TEXT,__cstring,cstring_literals
Lrt_prof_head: .asciz "\n  time%%        ms      calls   kernel\n"
Lrt_prof_row:  .asciz "%6.1f%%  %8.1f  %9lld   %s\n"
Lrt_prof_total: .asciz "          %8.1f ms in kernels\n"
Lrt_fail_fmt:  .asciz "anvil: runtime error: %s\n"
Lchk_s_nan:    .asciz "nan"
Lchk_s_inf:    .asciz "inf"
Lchk_s_ninf:   .asciz "-inf"
Lchk_f_head:   .asciz "\nanvil: --check: %s in %s"
Lchk_f_open:   .asciz " at ["
Lchk_f_index:  .asciz "%lld"
Lchk_f_index2: .asciz ", %lld"
Lchk_f_close:  .asciz "]"
Lchk_f_desc:   .asciz "\n%s\n"
Lchk_f_input:  .asciz "  its input %s already held %lld nan and %lld infinite values\n"
Lchk_f_made:   .asciz "  its inputs held no nan or infinity: this computation made it\n"
Lchk_f_passed: .asciz "  so the trouble started earlier: --check=inf stops at the first infinity\n"
Lsnpy_msg:     .asciz "anvil: cannot write %s\n"
Lrt_oob_fmt:   .asciz "anvil: runtime error: %s (got %lld)\n"
Lrt_open_msg:  .asciz "anvil: cannot open data file %s\n"
Lrt_bad_msg:   .asciz "anvil: data file %s does not have the shape this program was compiled for\n"
Lrt_rb:        .asciz "rb"
Lrt_wb:        .asciz "wb"
Lrt_cant_write: .asciz "anvil: cannot write %s\n"
Lrt_f4:        .asciz "%.4f"
Lrt_e4:        .asciz "%.4e"
Lrt_d:         .asciz "%d"
Lrt_ellipsis:  .asciz "..."

Lrt_env_threads: .asciz "ANVIL_THREADS"
Lrt_sysctl_pcores: .asciz "hw.perflevel0.physicalcpu"

    .section __DATA,__data
    .p2align 3
_anvil_seed:     .long 0
_anvil_stream:   .long 0
_anvil_t0:       .quad 0
_anvil_nthreads: .quad 1
    .p2align 7                          // separate 128-byte lines: no false sharing
_anvil_job:      .quad 0, 0, 0, 0         // fn, total, grain, chunks
    .p2align 7
_anvil_job_word: .quad 0                  // (generation << 32) | next chunk
    .p2align 7
_anvil_job_done: .quad 0
