"""The Metal backend: Anvil programs on the Apple GPU.

The output is one Objective-C++ file: the kernels, in Metal Shading Language (compiled when the
program starts), and a host program with the control flow. Kernels are generated as in the CUDA
backend (one thread per output element, reductions as loops in the thread, atomics for
scatter-adds, two-pass reductions when few outputs have long reductions), and so is most of the
host code. What differs is when the host waits for the GPU:

- Every tensor lives in an MTLBuffer in shared memory. Kernels are encoded into command buffers as
  the program runs; the host waits (anvil_sync) only before it reads or writes GPU memory: a print of
  a tensor, a branch on a computed value, a run-time call.
- Scalars that only the host computes, loop counters and what is computed from them alone (`step %
  100 == 0`, an optimizer's step count), live on the host: they are computed there, and copied into
  each kernel's arguments when it is encoded. So a training loop runs without a single wait, and an
  `if` on the step number does not stop the GPU.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from .. import ir
from ..ir import F32, I32, Load
from .aarch64 import c_escape, sanitize
from .cuda import CudaGen, KernelGen, ctype, flit, runtime_source, split_factor

MSL_PRELUDE = r"""#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;

// the counter-based random numbers every backend shares (float arithmetic is exact here)
inline uint32_t anvil_lowbias32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x;
}
inline uint32_t anvil_rand_bits(uint32_t idx, uint32_t stream, uint32_t salt, uint32_t seed) {
    uint32_t k = stream * 0x85EBCA77u + salt * 0xC2B2AE3Du + seed + 0x632BE5ABu;
    return anvil_lowbias32(anvil_lowbias32(idx * 0x9E3779B1u + k));
}
inline float anvil_rand(uint32_t idx, uint32_t stream, uint32_t salt, uint32_t seed, int open_low) {
    float hi = (float)(anvil_rand_bits(idx, stream, salt, seed) >> 8);
    if (open_low) hi += 1.0f;
    return hi * (1.0f / 16777216.0f);
}
inline int anvil_idiv(int a, int b) { if (b == 0) b = 1; int q = a / b, r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q; }
inline int anvil_imod(int a, int b) { if (b == 0) b = 1; int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
inline float anvil_fmod(float a, float b) { float r = fmod(a, b); return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r; }
inline float anvil_fmax(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : (a > b ? a : b); }
inline float anvil_fmin(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : (a < b ? a : b); }
inline int anvil_imax(int a, int b) { return a > b ? a : b; }
inline int anvil_imin(int a, int b) { return a < b ? a : b; }
inline float anvil_fsign(float a) { return a > 0 ? 1.0f : (a < 0 ? -1.0f : a); }
inline int anvil_isign(int a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }
inline float anvil_sigmoid(float a) { return 1.0f / (1.0f + exp(-a)); }
// Metal's tanh overflows to nan for large |a| (it computes with exp(2a)); past |a| = 10 tanh is ±1 in
// float anyway. A nan stays a nan, for --check.
inline float anvil_tanh(float a) { return (fabs(a) <= 10.0f || isnan(a)) ? tanh(a) : copysign(1.0f, a); }
inline int anvil_f2i(float a) { return (int)trunc(a); }
inline void anvil_atomic_add(device float *p, float v) { atomic_fetch_add_explicit((device atomic_float *)p, v, memory_order_relaxed); }
inline void anvil_atomic_add(device int *p, int v) { atomic_fetch_add_explicit((device atomic_int *)p, v, memory_order_relaxed); }
// a failed bounds check: the first one is kept, and the host reports it
inline void anvil_fail(device atomic_int *f, int code, long value) {
    int expected = 0;
    if (atomic_compare_exchange_weak_explicit(&f[0], &expected, code, memory_order_relaxed, memory_order_relaxed))
        atomic_store_explicit(&f[1], (int)value, memory_order_relaxed);
}
"""

MATH = {"expf": "exp", "logf": "log", "sqrtf": "sqrt", "tanhf": "anvil_tanh", "sinf": "sin", "cosf": "cos",
        "fabsf": "fabs", "floorf": "floor", "ceilf": "ceil", "rintf": "rint", "powf": "pow", "truncf": "trunc"}
MAX_BUFFERS = 31
MPS_MIN = 1 << 15          # M·N·K from which a product goes to Metal Performance Shaders


def fits(g) -> bool:
    """MPS reads each matrix as `rows` full rows of `ld` floats: they must lie inside the buffer, for
    every element of the batch."""
    def ok(buf, off, rows, ld, which):
        if buf is None:
            return True
        last = off + sum((lp.extent - 1) * getattr(lp, which) for lp in g.loops if getattr(lp, which) > 0)
        first = off + sum((lp.extent - 1) * getattr(lp, which) for lp in g.loops if getattr(lp, which) < 0)
        return first >= 0 and last + rows * ld <= max(4, buf.root.numel)
    a_rows = g.K if g.trans_a else g.M
    b_rows = g.N if g.trans_b else g.K
    return ok(g.A, g.a_off, a_rows, g.lda, "a") and ok(g.B, g.b_off, b_rows, g.ldb, "b") and ok(g.C, g.c_off, g.M, g.ldc, "c")


class MetalError(Exception):
    pass


def msl(text: str) -> str:
    """CUDA-flavored C from the shared kernel generator, as Metal Shading Language."""
    text = re.sub(r"\b(" + "|".join(MATH) + r")\(", lambda m: MATH[m.group(1)] + "(", text)
    text = text.replace("atomicAdd(", "anvil_atomic_add(").replace("anvil_fail(", "anvil_fail(anvil_fault, ")
    return text.replace("unsigned long long", "ulong")


class MetalKernelGen(KernelGen):
    """A kernel in MSL. Host scalars arrive in the argument array (after the stream and the seed)."""

    def __init__(self, gen: "MetalGen", k: ir.Kernel):
        super().__init__(gen, k)
        self.hosts: dict[int, ir.Buffer] = {}

    def host_value(self, b: ir.Buffer) -> str:
        r = b.root
        self.hosts[r.id] = r
        return f"h{r.id}"

    def scalar_ref(self, b: ir.Buffer) -> str:
        if self.gen.is_host(b):
            return f"(long){self.host_value(b)}"
        return super().scalar_ref(b)

    def build(self, e, ind) -> str:
        if isinstance(e, Load) and self.gen.is_host(e.buf):
            return self.host_value(e.buf)
        return super().build(e, ind)

    def wrap(self, name, comment, scratch, total, lines) -> str:
        params = ["constant uint *anvil_args [[buffer(0)]]", "device atomic_int *anvil_fault [[buffer(1)]]"]
        tensors = []                                   # (C type, name in the kernel, name on the host)
        if scratch is not None:
            tensors.append((ctype(self.k.red.dtype), scratch, scratch))
        for b in self.bufs.values():
            tensors.append((ctype(b.dtype), self.gen.bname(b), self.gen.mname(b)))
        # Metal binds at most MAX_BUFFERS buffers to a kernel. Past that (a sum of 32 unrolled losses
        # reads 32 tensors), the rest go through an argument buffer: a struct of their GPU addresses
        direct = tensors if len(params) + len(tensors) <= MAX_BUFFERS else tensors[:MAX_BUFFERS - len(params) - 1]
        more = tensors[len(direct):]
        self.args = [host for _, _, host in direct]
        self.more = [host for _, _, host in more]
        for t, kname, _ in direct:
            params.append(f"device {t} *{kname} [[buffer({len(params)})]]")
        struct = []
        pre = ["    uint32_t stream = anvil_args[0], seed = anvil_args[1];", "    (void)stream; (void)seed;"]
        if more:
            struct = [f"struct {name}_more {{"] + [f"    device {t} *{kname};" for t, kname, _ in more] + ["};"]
            params.append(f"constant {name}_more &anvil_more [[buffer({len(params)})]]")
            pre += [f"    device {t} *{kname} = anvil_more.{kname};" for t, kname, _ in more]
        self.host_order = list(self.hosts.values())
        for i, b in enumerate(self.host_order):
            v = f"as_type<float>(anvil_args[{2 + i}])" if b.dtype == F32 else f"(int)anvil_args[{2 + i}]"
            pre.append(f"    {ctype(b.dtype)} h{b.id} = {v};")
        head = struct + [comment, f"kernel void {name}({', '.join(params)}, uint tid_ [[thread_position_in_grid]]) {{"] + pre
        if self.parallel:
            head += ["    long tid = tid_;", f"    if (tid >= {total}L) return;", "    {"]
        else:
            head += ["    if (tid_ != 0) return;", f"    for (long tid = 0; tid < {total}L; tid++) {{"]
        return msl("\n".join(head + lines + ["    }", "}"]))


class HostKernelGen(KernelGen):
    """A scalar kernel computed on the host (from host scalars and constants): plain C."""

    def wrap(self, name, comment, scratch, total, lines) -> str:
        return "\n".join(["{ long tid = 0; (void)tid;"] + lines + ["}"])


class MetalGen(CudaGen):
    def __init__(self, prog: ir.Program, seed: int = 0):
        super().__init__(prog, seed)
        from .blas import prepare
        # matrix products go to Metal Performance Shaders (its tuned GPU kernels)
        self.gemms = prepare(prog, size_min=MPS_MIN, accept=fits) if os.environ.get("ANVIL_MPS", "1") != "0" else {}
        self.gemm_objects: list[str] = []
        self.gemm_inits: list[str] = []
        self.hosts = host_scalars(prog)
        self.host_kernels = {k.id for k in ir.all_kernels(prog) if is_host_kernel(k, self.hosts)}
        self.pipelines: list[str] = []

    def is_host(self, b: ir.Buffer) -> bool:
        return b.root.id in self.hosts

    def mname(self, b: ir.Buffer) -> str:
        return "M_" + self.bname(b)

    def parallel_safe(self, k: ir.Kernel) -> bool:
        return super().parallel_safe(k)

    # ------------------------------------------------------------------ host code
    def gpu_reads(self, bufs) -> bool:
        return any(isinstance(b, ir.Buffer) and not self.is_host(b) for b in bufs)

    def sync(self, ind, needed=True) -> list[str]:
        return [f"{ind}anvil_check_faults();"] if needed else []

    def stmt(self, st, ind) -> list[str]:
        if isinstance(st, ir.KernelStmt):
            k = st.kernel
            if k.id in self.host_kernels:
                return [ind + line for line in HostKernelGen(self, k).generate(f"k{k.id}").splitlines()]
            if k.id in self.gemms:
                return self.mps(self.gemms[k.id], k, ind)
            return self.launch(k, ind)
        if isinstance(st, ir.For):
            c = self.bname(st.counter)
            out = self.sync(ind, self.gpu_reads([st.start, st.stop]))
            out += [f"{ind}{{", f"{ind}    long stop_ = {self.scalar(st.stop)};",
                    f"{ind}    for ({c}[0] = (int)({self.scalar(st.start)}); {c}[0] < stop_; {c}[0] += {st.step}) {{"]
            return out + self.block(st.body, ind + "        ") + [f"{ind}    }}", f"{ind}}}"]
        if isinstance(st, ir.While):
            out = [f"{ind}while (1) {{"] + self.block(st.cond_block, ind + "    ")
            out += self.sync(ind + "    ", self.gpu_reads([st.cond]))
            out += [f"{ind}    if ({self.bname(st.cond)}[0] == 0) break;"]
            return out + self.block(st.body, ind + "    ") + [f"{ind}}}"]
        if isinstance(st, ir.If):
            out = self.sync(ind, self.gpu_reads([st.cond]))
            out += [f"{ind}if ({self.bname(st.cond)}[0] != 0) {{"] + self.block(st.then, ind + "    ")
            if st.orelse.stmts:
                out += [f"{ind}}} else {{"] + self.block(st.orelse, ind + "    ")
            return out + [f"{ind}}}"]
        if isinstance(st, ir.Print):
            gpu = self.gpu_reads([it.buf for it in st.items if it.buf is not None])
            lines = super().print_(st, ind)
            lines = [l for l in lines if l.strip() != "anvil_check_faults();"]
            return self.sync(ind, gpu) + lines
        if isinstance(st, ir.Check):
            gpu = self.gpu_reads([t.buf for t in st.value.terms if isinstance(t, ir.ScalarRef)])
            lines = [l for l in super().stmt(st, ind) if l.strip() != "anvil_check_faults();"]
            return self.sync(ind, gpu) + lines
        if isinstance(st, ir.RTCall):
            host_only = st.name in ("seed", "skip_rand")
            return self.sync(ind, not host_only) + [ind + line for line in self.rtcall(st)]
        return super().stmt(st, ind)

    def launch(self, k: ir.Kernel, ind) -> list[str]:
        if k.id not in self.kernel_names:
            parallel = self.parallel_safe(k)
            split = split_factor(k, parallel)
            passes = []
            modes = [("part", f"k{k.id}_part"), ("final", f"k{k.id}")] if split > 1 else [("full", f"k{k.id}")]
            for mode, name in modes:
                kg = MetalKernelGen(self, k)
                self.kernels.append(kg.generate(name, mode, split))
                self.pipelines.append(name)
                n = ir.prod(v.extent for v in k.domain) * (split if mode == "part" else 1)
                args = [("M_" + a if not a.startswith("M_") else a) for a in kg.args]
                more = [("M_" + a if not a.startswith("M_") else a) for a in kg.more]
                passes.append((name, n if kg.parallel else 1, args, kg.host_order, more))
            self.kernel_names[k.id] = passes
        out = [f"{ind}{{"]
        stream = "anvil_stream++" if k.uses_rand() else "0"
        out.append(f"{ind}    uint32_t s_ = {stream}; (void)s_;")
        for name, n, args, hosts, more in self.kernel_names[k.id]:
            vals = ["s_", "anvil_seed_value"] + [f"anvil_bits({self.bname(h)}[0])" for h in hosts]
            out.append(f"{ind}    {{ id<MTLComputeCommandEncoder> e_ = anvil_encoder();")
            out.append(f"{ind}      uint32_t a_[] = {{{', '.join(vals)}}};")
            out.append(f"{ind}      [e_ setComputePipelineState:P_{name}];")
            out.append(f"{ind}      [e_ setBytes:a_ length:sizeof a_ atIndex:0];")
            out.append(f"{ind}      [e_ setBuffer:anvil_fault_buf offset:0 atIndex:1];")
            for i, a in enumerate(args):
                out.append(f"{ind}      [e_ setBuffer:{a} offset:0 atIndex:{2 + i}];")
            if more:                                   # the argument buffer: GPU addresses, made resident
                out.append(f"{ind}      uint64_t m_[] = {{{', '.join(f'{a}.gpuAddress' for a in more)}}};")
                out.append(f"{ind}      [e_ setBytes:m_ length:sizeof m_ atIndex:{2 + len(args)}];")
                for a in more:
                    out.append(f"{ind}      [e_ useResource:{a} usage:MTLResourceUsageRead | MTLResourceUsageWrite];")
            out.append(f"{ind}      anvil_dispatch(e_, P_{name}, {n}L); }}")
        out.append(f"{ind}}}")
        return out

    def mps(self, g, k: ir.Kernel, ind) -> list[str]:
        """A matrix product (or a stack or sum of them) through MPSMatrixMultiplication."""
        n = f"g{k.id}"
        if n not in self.gemm_objects:
            self.gemm_objects.append(n)
            a_rows, a_cols = (g.K, g.M) if g.trans_a else (g.M, g.K)
            b_rows, b_cols = (g.N, g.K) if g.trans_b else (g.K, g.N)
            ta, tb = "YES" if g.trans_a else "NO", "YES" if g.trans_b else "NO"
            self.gemm_inits += [
                f"{n}_a = anvil_matrix({a_rows}, {a_cols}, {g.lda}); {n}_b = anvil_matrix({b_rows}, {b_cols}, {g.ldb}); "
                f"{n}_c = anvil_matrix({g.M}, {g.N}, {g.ldc});",
                f"{n}_first = anvil_product({ta}, {tb}, {g.M}, {g.N}, {g.K}, {g.beta}); "
                f"{n}_rest = anvil_product({ta}, {tb}, {g.M}, {g.N}, {g.K}, 1.0);"]
        out = [f"{ind}{{   // {k.label or 'matmul'}: {g.M}×{g.K} @ {g.K}×{g.N}" + (f" × {g.calls}" if g.loops else "")
               + " (Metal Performance Shaders)"]
        inner = ind + "    "
        for d, lp in enumerate(g.loops):
            out.append(f"{inner}for (long l{d} = 0; l{d} < {lp.extent}; l{d}++) {{")
            inner += "    "

        def offset(base, which):
            terms = [str(base)] + [f"l{d} * {getattr(lp, which)}L" for d, lp in enumerate(g.loops) if getattr(lp, which)]
            return " + ".join(terms)
        summed = [f"l{d} == 0" for d, lp in enumerate(g.loops) if lp.summed]
        mm = f"({' && '.join(summed)} ? {n}_first : {n}_rest)" if summed else f"{n}_first"
        out.append(f"{inner}anvil_mps({mm}, {self.mname(g.A)}, {offset(g.a_off, 'a')}, {n}_a, "
                   f"{self.mname(g.B)}, {offset(g.b_off, 'b')}, {n}_b, {self.mname(g.C)}, {offset(g.c_off, 'c')}, {n}_c);")
        for d in range(len(g.loops)):
            inner = inner[:-4]
            out.append(f"{inner}}}")
        out.append(f"{ind}}}")
        return out

    def scratch(self, k: ir.Kernel, numel: int) -> str:
        name = f"part{k.id}"
        if name not in self.scratches:
            self.scratches[name] = (ctype(k.red.dtype), numel)
        elif numel:
            self.scratches[name] = (ctype(k.red.dtype), max(numel, self.scratches[name][1]))
        return name

    def generate(self) -> str:
        body = self.block(self.prog.main, "        ")
        src = self.prog.source
        name = src.display_path() if src is not None else "<program>"
        kernels = MSL_PRELUDE + "\n" + "\n\n".join(self.kernels)
        if ")ANVILMSL\"" in kernels:
            raise MetalError("the kernel source contains the raw-string delimiter")
        out = [f"// Generated by anvil from {name} — Metal (the Apple GPU): the kernels in Metal Shading Language,",
               "// compiled when the program starts, and the host program in Objective-C++.",
               "// Build: clang++ -std=c++17 -fobjc-arc -O2 -x objective-c++ prog.mm -framework Metal -framework Foundation -lz",
               "", runtime_source("anvil_metal.h"), "",
               "// ---------------------------------------------------------------- kernels (Metal Shading Language)",
               f'static const char *anvil_msl = R"ANVILMSL({kernels})ANVILMSL";', "",
               "// ---------------------------------------------------------------- tensors"]
        bufs = sorted(self.used.values(), key=lambda b: b.id)
        for b in bufs:
            T = ctype(b.dtype)
            if self.is_host(b):
                init = flit(float(b.init[0])) if (b.init and b.dtype == F32) else (str(int(b.init[0])) if b.init else "0")
                out.append(f"static {T} {self.bname(b)}[1] = {{{init}}};   // {b.kind} {b.name}: on the host")
                continue
            out.append(f"static id<MTLBuffer> {self.mname(b)}; static {T} *{self.bname(b)};   // {b.kind} {b.name}{list(b.shape)}")
            if b.init is not None:
                vals = ", ".join((flit(float(v)) if b.dtype == F32 else str(int(v))) for v in b.init)
                out.append(f"static const {T} {self.bname(b)}_init[] = {{{vals}}};")
        for nm, (T, numel) in self.scratches.items():
            out.append(f"static id<MTLBuffer> M_{nm}; static {T} *{nm};   // partial results of a split reduction")
        for p in self.pipelines:
            out.append(f"static id<MTLComputePipelineState> P_{p};")
        for n in self.gemm_objects:
            out.append(f"static MPSMatrixDescriptor *{n}_a, *{n}_b, *{n}_c; static MPSMatrixMultiplication *{n}_first, *{n}_rest;")
        out += [""] + self.tables
        out += ["", "int main(void) {", "    @autoreleasepool {", "        anvil_t0 = anvil_now();",
                f"        anvil_seed_value = {self.seed & 0xFFFFFFFF}u;", "        anvil_metal_init(anvil_msl);"]
        for p in self.pipelines:
            out.append(f'        P_{p} = anvil_pipeline("{p}");')
        out += ["        " + line for line in self.gemm_inits]
        for b in bufs:
            if self.is_host(b):
                continue
            out.append(f"        {self.mname(b)} = anvil_buffer({max(1, b.numel) * 4}); "
                       f"{self.bname(b)} = ({ctype(b.dtype)} *){self.mname(b)}.contents;")
            if b.init is not None:
                out.append(f"        memcpy({self.bname(b)}, {self.bname(b)}_init, sizeof {self.bname(b)}_init);")
        for nm, (T, numel) in self.scratches.items():
            out.append(f"        M_{nm} = anvil_buffer({numel * 4}); {nm} = ({T} *)M_{nm}.contents;")
        for i, m in enumerate(self.faults, 1):
            out.append(f'        anvil_fault_messages[{i}] = "{c_escape(m)}";')
        out += ["    " + c for c in self.ckpts]
        out += body
        out += ["    anvil_end:", "        anvil_check_faults();", "        fflush(stdout);", "    }", "    return 0;", "}", ""]
        return "\n".join(out)


def host_scalars(prog: ir.Program) -> set:
    """The ids of the scalars only the host computes: loop counters, and 0-d results of kernels that
    read only such scalars and constants (found by iterating to a fixed point)."""
    writers: dict[int, list] = {}
    counters = set()
    for blk in ir.walk_blocks(prog.main):
        for st in blk.stmts:
            if isinstance(st, ir.For):
                counters.add(st.counter.root.id)
                writers.setdefault(st.counter.root.id, []).append("for")
            elif isinstance(st, ir.KernelStmt):
                for s in st.kernel.stores:
                    writers.setdefault(s.buf.root.id, []).append(st.kernel)
            elif isinstance(st, ir.RTCall):
                from ..optimize import RT_WRITES, rt_buffers
                for b in rt_buffers(st, RT_WRITES):
                    writers.setdefault(b.root.id, []).append("rt")
    hosts = {c for c in counters if all(w == "for" for w in writers.get(c, []))}
    kernels = [k for k in ir.all_kernels(prog) if scalar_kernel(k)]

    def outs(k):
        return {s.buf.root.id for s in k.stores}
    banned: set = set()
    changed = True
    while changed:
        changed = False
        for k in kernels:                         # grow: a scalar kernel whose inputs are on the host
            if not inputs_on_host(k, hosts | (outs(k) - banned)):     # (it may read what it writes: t = t + 1)
                continue
            for o in outs(k) - hosts - banned:
                if all(isinstance(w, ir.Kernel) and scalar_kernel(w) for w in writers.get(o, [])):
                    hosts.add(o)
                    changed = True
        for o in sorted(hosts - counters):        # shrink: every writer must run on the host, whole
            ws = writers.get(o, [])
            if not all(isinstance(w, ir.Kernel) and inputs_on_host(w, hosts) and outs(w) <= hosts for w in ws):
                hosts.discard(o)
                banned.add(o)
                changed = True
    return hosts


def scalar_kernel(k: ir.Kernel) -> bool:
    return (not k.domain and k.red is None and not k.uses_rand()
            and all(not s.buf.shape and not s.accumulate for s in k.stores))


def inputs_on_host(k: ir.Kernel, hosts: set) -> bool:
    from .cuda import kernel_loads
    for ld in kernel_loads(k):
        b = ld.buf.root
        if b.id not in hosts and b.kind != "const":
            return False
    for e in k.exprs():
        for x in ir.iter_expr(e):
            terms = x.offset.terms if isinstance(x, Load) else (x.affine.terms if isinstance(x, ir.Index) else {})
            for t in terms:
                if isinstance(t, ir.ScalarRef) and t.buf.root.id not in hosts:
                    return False
    for s in k.stores:
        for t in s.offset.terms:
            if isinstance(t, ir.ScalarRef) and t.buf.root.id not in hosts:
                return False
    return True


def is_host_kernel(k: ir.Kernel, hosts: set) -> bool:
    return scalar_kernel(k) and all(s.buf.root.id in hosts for s in k.stores) and inputs_on_host(k, hosts)


def generate(prog: ir.Program, seed: int = 0) -> str:
    return MetalGen(prog, seed).generate()


def build(source: str, exe: str):
    """Compile a generated Objective-C++ file (the Metal kernels compile when it runs)."""
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        raise MetalError("no clang++ found (install the Xcode command line tools)")
    with tempfile.TemporaryDirectory() as d:
        mm = os.path.join(d, "prog.mm")
        with open(mm, "w") as f:
            f.write(source)
        cmd = [cxx, "-std=c++17", "-fobjc-arc", "-O2", "-w", "-x", "objective-c++", mm, "-o", exe,
               "-framework", "Metal", "-framework", "MetalPerformanceShaders", "-framework", "Foundation", "-lz"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalError(f"clang++ failed:\n{r.stderr[-4000:]}")
