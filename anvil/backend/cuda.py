"""The CUDA backend: Anvil's kernel IR as one self-contained .cu file.

Each kernel `out[I] = ⊕_J body(I, J)` becomes a __global__ function that runs a grid-stride loop
over the flattened output domain, one element per thread; the reduction over J is a sequential
loop inside the thread. Accumulating stores (scatter-adds and gradient scatters) use atomicAdd.
A kernel whose result would depend on the order of its elements (it reads elements of a buffer it
also writes, or several elements write the same place without adding) runs on one thread.

Host code is plain C with the program's control flow. Every tensor lives in CUDA unified memory,
so the runtime (printing, data loading, the random number generator, input, checkpoints) works on
the same buffers, after a cudaDeviceSynchronize. The file compiles with nvcc, and also as C++ on a
machine without a GPU (-DANVIL_EMULATE), which is how the backend is tested against the interpreter.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

from .. import ir
from ..ir import F32, I32, Acc, Affine, Binary, Const, Gather, Index, LetRef, Load, Rand, ScalarRef, Select, Unary
from .aarch64 import c_escape, sanitize

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "anvil_cuda.h")


def runtime_source(header: str) -> str:
    """A runtime header with anvil_host.h written into it (generated files stand alone)."""
    with open(os.path.join(HERE, "anvil_host.h")) as f:
        host = f.read().replace("#pragma once\n", "")
    with open(os.path.join(HERE, header)) as f:
        return f.read().replace("#pragma once\n", "").replace('#include "anvil_host.h"', host)
THREADS = 256


def ctype(dtype) -> str:
    return "int" if dtype == I32 else "float"


def flit(v: float) -> str:
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    r = repr(float(v))
    return f"{r}f" if ("e" in r or "." in r) else f"{r}.0f"


class KernelGen:
    def __init__(self, gen: "CudaGen", k: ir.Kernel):
        self.gen = gen
        self.k = k
        self.lines: list[str] = []
        self.n = 0
        self.bufs: dict[int, ir.Buffer] = {}
        self.dom = {v: f"d{i}" for i, v in enumerate(k.domain)}
        self.red = {v: f"r{i}" for i, v in enumerate(k.red.vars)} if k.red is not None else {}
        self.lets: dict[int, str] = {}
        self.memo: dict = {}
        self.parallel = gen.parallel_safe(k)
        self.rand_idx = "tid"

    def tmp(self) -> str:
        self.n += 1
        return f"t{self.n}"

    def buf(self, b: ir.Buffer) -> str:
        r = b.root
        self.bufs[r.id] = r
        return self.gen.bname(r)

    def var(self, v) -> str:
        return self.dom.get(v) or self.red[v]

    def affine(self, a: Affine, ind: str) -> str:
        parts = [str(a.const)] if a.const else []
        for t, c in a.terms.items():
            if isinstance(t, ir.Var):
                x = self.var(t)
            elif isinstance(t, ScalarRef):
                x = self.scalar_ref(t.buf)
            elif isinstance(t, Gather):
                x = self.gather(t, ind)
            else:
                raise NotImplementedError(repr(t))
            parts.append(x if c == 1 else f"{c}L * {x}")
        return " + ".join(parts) if parts else "0"

    def scalar_ref(self, b: ir.Buffer) -> str:
        """A run-time integer scalar used in an index."""
        return f"(long){self.buf(b)}[0]"

    def gather(self, g: Gather, ind: str) -> str:
        """A gathered index, bounds-checked (a failure is reported by the host; the index is clamped)."""
        key = ("g", g.key())
        if key in self.memo:
            return self.memo[key]
        idx = self.affine(g.load.offset, ind)
        name = self.tmp()
        lo, hi = g.check_range()
        code = self.gen.fault(f"index out of bounds ({g.where or 'gather'}; dimension size {g.bound})")
        self.lines.append(f"{ind}long {name} = (long){self.buf(g.load.buf)}[{idx}];")
        self.lines.append(f"{ind}if ({name} < {lo}L || {name} >= {hi}L) {{ anvil_fail({code}, {name}); {name} = {lo}L; }}")
        self.memo[key] = name
        return name

    def ex(self, e, ind: str) -> str:
        """C for expression e; shared subexpressions become temporaries."""
        if isinstance(e, Const):
            return flit(e.value) if e.dtype == F32 else f"({int(e.value)})"
        if isinstance(e, Acc):
            return "acc"
        if isinstance(e, LetRef):
            return self.lets[id(e.let)]
        key = e.key()
        if key in self.memo:
            return self.memo[key]
        s = self.build(e, ind)
        if isinstance(e, (Load, Index, Rand, Unary, Binary, Select)):
            name = self.tmp()
            self.lines.append(f"{ind}{ctype(e.dtype)} {name} = {s};")
            s = name
        self.memo[key] = s
        return s

    def cast(self, e, dtype, ind) -> str:
        s = self.ex(e, ind)
        if e.dtype == dtype:
            return s
        return f"(float)({s})" if dtype == F32 else f"anvil_f2i({s})"

    def build(self, e, ind) -> str:
        if isinstance(e, Load):
            return f"{self.buf(e.buf)}[{self.affine(e.offset, ind)}]"
        if isinstance(e, Index):
            return f"(int)({self.affine(e.affine, ind)})"
        if isinstance(e, Rand):
            return f"anvil_rand((uint32_t){self.rand_idx}, stream, {e.salt}u, seed, {1 if e.open_low else 0})"
        if isinstance(e, Unary):
            return self.unary(e, ind)
        if isinstance(e, Binary):
            return self.binary(e, ind)
        if isinstance(e, Select):
            c = self.ex(e.c, ind)
            return f"(({c}) != 0 ? {self.cast(e.a, e.dtype, ind)} : {self.cast(e.b, e.dtype, ind)})"
        raise NotImplementedError(type(e).__name__)

    def unary(self, e: Unary, ind) -> str:
        op, isint = e.op, e.a.dtype == I32
        a = self.ex(e.a, ind)
        if op in ("detach",):
            return a
        if op == "neg":
            return f"(-({a}))"
        if op == "abs":
            return f"abs({a})" if isint else f"fabsf({a})"
        if op == "not":
            return f"(({a}) == 0 ? 1.0f : 0.0f)"
        if op == "f32":
            return f"(float)({a})"
        if op == "i32":
            return a if isint else f"anvil_f2i({a})"
        if op == "sign":
            return f"anvil_isign({a})" if isint else f"anvil_fsign({a})"
        if op in ("floor", "ceil", "round"):
            return a if isint else {"floor": "floorf", "ceil": "ceilf", "round": "rintf"}[op] + f"({a})"
        fa = f"(float)({a})" if isint else a
        if op == "rsqrt":
            return f"(1.0f / sqrtf({fa}))"
        if op == "recip":
            return f"(1.0f / ({fa}))"
        if op == "sigmoid":
            return f"anvil_sigmoid({fa})"
        return {"exp": "expf", "log": "logf", "sqrt": "sqrtf", "tanh": "tanhf", "sin": "sinf", "cos": "cosf"}[op] + f"({fa})"

    def binary(self, e: Binary, ind) -> str:
        op = e.op
        if op in ir.COMPARISONS or op in ("and", "or"):
            a, b = self.ex(e.a, ind), self.ex(e.b, ind)
            if op in ("and", "or"):
                return f"((({a}) != 0 {'&&' if op == 'and' else '||'} ({b}) != 0) ? 1.0f : 0.0f)"
            sym = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[op]
            if e.a.dtype != e.b.dtype:
                a, b = f"(float)({a})", f"(float)({b})"
            return f"(({a}) {sym} ({b}) ? 1.0f : 0.0f)"
        if e.dtype == I32:
            a, b = self.ex(e.a, ind), self.ex(e.b, ind)
            if op in ("add", "sub", "mul"):
                sym = {"add": "+", "sub": "-", "mul": "*"}[op]
                return f"(int)((uint32_t)({a}) {sym} (uint32_t)({b}))"       # wraps around, like int32
            return {"idiv": "anvil_idiv", "mod": "anvil_imod", "max": "anvil_imax", "min": "anvil_imin"}[op] + f"({a}, {b})"
        a, b = self.cast(e.a, F32, ind), self.cast(e.b, F32, ind)
        if op in ("add", "sub", "mul", "div"):
            return f"(({a}) {dict(add='+', sub='-', mul='*', div='/')[op]} ({b}))"
        return {"max": f"anvil_fmax({a}, {b})", "min": f"anvil_fmin({a}, {b})", "pow": f"powf({a}, {b})",
                "idiv": f"floorf(({a}) / ({b}))", "mod": f"anvil_fmod({a}, {b})"}[op]

    # ------------------------------------------------------------------ the kernel
    def generate(self, name: str, mode: str = "full", split: int = 1) -> str:
        """mode "full": the whole kernel. A split reduction is two kernels: "part" reduces every
        split-th element (thread p of each output starts at p) into scratch, and "final" combines
        the partial results and runs the kernel's statements."""
        k = self.k
        ind = "        "
        out_total = ir.prod(v.extent for v in k.domain)
        total = out_total * (split if mode == "part" else 1)
        body = [f"{ind}{index_t(total)} u = ({index_t(total)})tid;"]
        if mode == "part":
            body.append(f"{ind}{index_t(total)} otid = u / {split}u, p = u % {split}u;")
            self.rand_idx = "otid"
        src = "otid" if mode == "part" else "u"
        strides = ir.row_major_strides([v.extent for v in k.domain])
        for v, st in zip(k.domain, strides):
            body.append(f"{ind}long {self.dom[v]} = (long)(({src} / {st}u) % {v.extent}u);")
        scratch = None
        if mode != "full":
            scratch = self.gen.scratch(k, out_total * split)
        if k.red is not None:
            if mode == "full":
                self.reduction(ind)
            elif mode == "part":
                self.reduction(ind, split=split)
                self.lines.append(f"{ind}{scratch}[tid] = acc;")
            else:
                self.combine(ind, scratch, split)
        if mode != "part":
            self.statements(ind)
        sched = {"full": "parallel" if self.parallel else "one thread", "part": f"split reduction, part 1 of 2 ({split} threads per output)",
                 "final": "split reduction, part 2 of 2"}[mode]
        comment = (f"// {k.name} · {k.label or 'kernel'} · {sched} · "
                   + " × ".join(f"{v.name}<{v.extent}" for v in k.domain)
                   + (f"  reduce {' × '.join(f'{v.name}<{v.extent}' for v in k.red.vars)}" if k.red is not None else ""))
        return self.wrap(name, comment, scratch, total, body + self.lines)

    def wrap(self, name, comment, scratch, total, lines) -> str:
        """The kernel function around its body (one output element per `tid`)."""
        params = [f"{ctype(b.dtype)} *__restrict__ {self.gen.bname(b)}" for b in self.bufs.values()]
        if scratch is not None:
            params.insert(0, f"{ctype(self.k.red.dtype)} *__restrict__ {scratch}")
        head = [comment,
                f"__global__ void {name}(uint32_t stream, uint32_t seed{''.join(', ' + p for p in params)}) {{",
                f"    for (long tid = blockIdx.x * (long)blockDim.x + threadIdx.x; tid < {total}L; "
                f"tid += (long)gridDim.x * blockDim.x) {{"]
        return "\n".join(head + lines + ["    }", "}"])

    def statements(self, ind):
        for st in self.k.stmts:
            if isinstance(st, ir.Let):
                v = self.ex(st.value, ind)
                name_ = self.tmp()
                self.lines.append(f"{ind}{ctype(st.value.dtype)} {name_} = {v};")
                self.lets[id(st)] = name_
                continue
            val = self.cast(st.value, st.buf.dtype, ind)
            off = self.affine(st.offset, ind)
            target = f"{self.buf(st.buf)}[{off}]"
            if st.accumulate:
                self.lines.append(f"{ind}atomicAdd(&{target}, {val});" if self.parallel else f"{ind}{target} += {val};")
            else:
                self.lines.append(f"{ind}{target} = {val};")
            self.memo = {}                       # loads after a store read memory again

    def init(self):
        op, dt = self.k.red.op, self.k.red.body.dtype
        if op in ("argmax", "argmin"):
            return ("-INFINITY" if op == "argmax" else "INFINITY") if dt == F32 else \
                ("(-2147483647 - 1)" if op == "argmax" else "2147483647")
        return {"sum": "0", "prod": "1", "max": "-INFINITY" if dt == F32 else "(-2147483647 - 1)",
                "min": "INFINITY" if dt == F32 else "2147483647"}[op]

    def combine_into(self, ind, val):
        op, dt = self.k.red.op, self.k.red.body.dtype
        if op == "sum":
            return f"{ind}acc += {val};"
        if op == "prod":
            return f"{ind}acc *= {val};"
        return f"{ind}acc = {'anvil_f' if dt == F32 else 'anvil_i'}{op}(acc, {val});"

    def reduction(self, ind, split: int = 1):
        red = self.k.red
        op, dt = red.op, red.body.dtype
        T = ctype(dt)
        if op in ("argmax", "argmin"):
            self.lines.append(f"{ind}{T} best = {self.init()}; int acc = 0; long ridx = 0;")
        else:
            self.lines.append(f"{ind}{T} acc = {self.init()};")
        inner = ind
        if split > 1:                            # every split-th element of the flattened reduction
            n = ir.prod(v.extent for v in red.vars)
            self.lines.append(f"{inner}for ({index_t(n)} rr = p; rr < {n}u; rr += {split}u) {{")
            inner += "    "
            for v, st in zip(red.vars, ir.row_major_strides([v.extent for v in red.vars])):
                self.lines.append(f"{inner}long {self.red[v]} = (long)((rr / {st}u) % {v.extent}u);")
        else:
            for v in red.vars:
                self.lines.append(f"{inner}for (long {self.red[v]} = 0; {self.red[v]} < {v.extent}L; {self.red[v]}++) {{")
                inner += "    "
        saved = self.memo
        self.memo = dict(saved)                  # temporaries made inside the loop stay inside it
        val = self.ex(red.body, inner)
        if op in ("argmax", "argmin"):
            cmp = ">" if op == "argmax" else "<"
            self.lines.append(f"{inner}if ({val} {cmp} best) {{ best = {val}; acc = (int)ridx; }}")
            self.lines.append(f"{inner}ridx++;")
        else:
            self.lines.append(self.combine_into(inner, val))
        self.memo = saved
        for _ in (red.vars if split == 1 else [0]):
            inner = inner[:-4]
            self.lines.append(f"{inner}}}")

    def combine(self, ind, scratch, split):
        T = ctype(self.k.red.body.dtype)
        self.lines.append(f"{ind}{T} acc = {self.init()};")
        self.lines.append(f"{ind}for (int p = 0; p < {split}; p++) {{")
        self.lines.append(self.combine_into(ind + "    ", f"{scratch}[tid * {split} + p]"))
        self.lines.append(f"{ind}}}")


def index_t(n: int) -> str:
    """32-bit index arithmetic where it fits (64-bit division is slow on a GPU)."""
    return "uint32_t" if n < (1 << 32) else "unsigned long long"


SPLIT_MAX_OUTPUTS = 4096       # kernels with fewer outputs than this…
SPLIT_MIN_REDUCE = 2048        # …and a reduction at least this long are split across threads
SPLIT_THREADS = 1 << 16        # aim for about this many threads
SPLIT_MAX = 1024


def split_factor(k: ir.Kernel, parallel: bool) -> int:
    """Threads per output for a kernel whose reduction is long and whose outputs are few."""
    if k.red is None or not parallel or k.red.op in ("argmax", "argmin"):
        return 1
    out = ir.prod(v.extent for v in k.domain)
    n = ir.prod(v.extent for v in k.red.vars)
    if out > SPLIT_MAX_OUTPUTS or n < SPLIT_MIN_REDUCE:
        return 1
    p = 1
    while p * 2 <= SPLIT_MAX and out * p * 2 <= SPLIT_THREADS and n // (p * 2) >= 16:
        p *= 2
    return p


class CudaGen:
    def __init__(self, prog: ir.Program, seed: int = 0):
        self.prog = prog
        self.seed = seed
        self.kernels: list[str] = []
        self.kernel_names: dict[int, tuple] = {}
        self.faults: list[str] = []
        self.tables: list[str] = []
        self.ckpts: list[str] = []
        self.used: dict[int, ir.Buffer] = {}
        self.scratches: dict[str, tuple] = {}
        self.n = 0

    def bname(self, b: ir.Buffer) -> str:
        r = b.root
        self.used[r.id] = r
        return f"b{r.id}_{sanitize(r.name)}"

    def fault(self, message: str) -> int:
        if message in self.faults:
            return self.faults.index(message) + 1
        if len(self.faults) >= 254:
            message = "index out of bounds"
            if message in self.faults:
                return self.faults.index(message) + 1
        self.faults.append(message)
        return len(self.faults)

    def label(self, hint: str) -> str:
        self.n += 1
        return f"{hint}{self.n}"

    def parallel_safe(self, k: ir.Kernel) -> bool:
        """Can every output element run in its own thread, in any order?"""
        written = {s.buf.root for s in k.stores}
        for s in k.stores:
            if s.accumulate:
                continue                                   # atomicAdd
            if s.offset.gathers():
                return False                               # duplicate indices: the last write wins
            if not injective_over(s.offset, k.domain):
                return False                               # several elements write one place
        for ld in kernel_loads(k):
            if ld.buf.root in written:
                same = [s for s in k.stores if s.buf.root is ld.buf.root]
                if any(s.accumulate or s.offset != ld.offset for s in same):
                    return False                           # reads what other elements write
        return True

    # ------------------------------------------------------------------ host code
    def generate(self) -> str:
        body = self.block(self.prog.main, "    ")
        src = self.prog.source
        name = src.display_path() if src is not None else "<program>"
        out = [f"// Generated by anvil from {name} — CUDA (one thread per output element, unified memory).",
               "// Compile: nvcc -O3 -o prog prog.cu -lz        Without a GPU: c++ -O2 -std=c++17 -DANVIL_EMULATE -x c++ prog.cu -lz",
               "", runtime_source("anvil_cuda.h"), "", "// ---------------------------------------------------------------- tensors"]
        bufs = sorted(self.used.values(), key=lambda b: b.id)
        for b in bufs:
            out.append(f"static {ctype(b.dtype)} *{self.bname(b)};   // {b.kind} {b.name}{list(b.shape)}")
            if b.init is not None:
                vals = ", ".join((flit(float(v)) if b.dtype == F32 else str(int(v))) for v in b.init)
                out.append(f"static const {ctype(b.dtype)} {self.bname(b)}_init[] = {{{vals}}};")
        for name, (T, numel) in self.scratches.items():
            out.append(f"static {T} *{name};   // partial results of a split reduction")
        out += [""] + self.tables + ["", "// ---------------------------------------------------------------- kernels"]
        out += self.kernels
        out += ["", "int main(void) {", "    anvil_t0 = anvil_now();", f"    anvil_seed_value = {self.seed & 0xFFFFFFFF}u;"]
        for b in bufs:
            out.append(f"    {self.bname(b)} = ({ctype(b.dtype)} *)anvil_alloc({max(1, b.numel) * 4});")
            if b.init is not None:
                out.append(f"    memcpy({self.bname(b)}, {self.bname(b)}_init, sizeof {self.bname(b)}_init);")
        for name, (T, numel) in self.scratches.items():
            out.append(f"    {name} = ({T} *)anvil_alloc({numel * 4});")
        for i, m in enumerate(self.faults, 1):
            out.append(f'    anvil_fault_messages[{i}] = "{c_escape(m)}";')
        out += self.ckpts
        out += body
        out += ["anvil_end:", "    anvil_check_faults();", "    fflush(stdout);", "    return 0;", "}", ""]
        return "\n".join(out)

    def block(self, b: ir.Block, ind: str) -> list[str]:
        out = []
        for st in b.stmts:
            out += self.stmt(st, ind)
        return out

    def scalar(self, x) -> str:
        return str(x) if isinstance(x, int) else f"(long){self.bname(x)}[0]"

    def stmt(self, st, ind) -> list[str]:
        if isinstance(st, ir.KernelStmt):
            return self.launch(st.kernel, ind)
        if isinstance(st, ir.For):
            c = self.bname(st.counter)
            out = [f"{ind}anvil_check_faults();", f"{ind}{{",
                   f"{ind}    long stop_ = {self.scalar(st.stop)};",
                   f"{ind}    for ({c}[0] = (int)({self.scalar(st.start)}); {c}[0] < stop_; anvil_check_faults(), {c}[0] += {st.step}) {{"]
            return out + self.block(st.body, ind + "        ") + [f"{ind}    }}", f"{ind}}}"]
        if isinstance(st, ir.While):
            out = [f"{ind}while (1) {{"] + self.block(st.cond_block, ind + "    ")
            out += [f"{ind}    anvil_check_faults();", f"{ind}    if ({self.bname(st.cond)}[0] == 0) break;"]
            return out + self.block(st.body, ind + "    ") + [f"{ind}}}"]
        if isinstance(st, ir.If):
            out = [f"{ind}anvil_check_faults();", f"{ind}if ({self.bname(st.cond)}[0] != 0) {{"]
            out += self.block(st.then, ind + "    ")
            if st.orelse.stmts:
                out += [f"{ind}}} else {{"] + self.block(st.orelse, ind + "    ")
            return out + [f"{ind}}}"]
        if isinstance(st, ir.Break):
            return [f"{ind}break;"]
        if isinstance(st, ir.Continue):
            return [f"{ind}continue;"]
        if isinstance(st, ir.Print):
            return self.print_(st, ind)
        if isinstance(st, ir.Check):
            loc = f"{st.span.location()}: " if st.span is not None else ""
            v = st.value.const
            terms = " + ".join(f"{c}L * (long){self.bname(t.buf)}[0]" for t, c in st.value.terms.items()
                               if isinstance(t, ScalarRef))
            return [f"{ind}anvil_check_faults();",
                    f"{ind}{{ long v_ = {v}L{' + ' + terms if terms else ''}; "
                    f"if (v_ < {st.lo}L || v_ > {st.hi}L) anvil_oob(\"{c_escape(loc + st.message)}\", v_); }}"]
        if isinstance(st, ir.RTCall):
            return [f"{ind}anvil_check_faults();"] + [ind + line for line in self.rtcall(st)]
        raise NotImplementedError(type(st).__name__)

    def launch(self, k: ir.Kernel, ind) -> list[str]:
        if k.id not in self.kernel_names:
            parallel = self.parallel_safe(k)
            split = split_factor(k, parallel)
            passes = []
            modes = [("part", f"k{k.id}_part"), ("final", f"k{k.id}")] if split > 1 else [("full", f"k{k.id}")]
            for mode, name in modes:
                kg = KernelGen(self, k)
                self.kernels.append(kg.generate(name, mode, split))
                n = ir.prod(v.extent for v in k.domain) * (split if mode == "part" else 1)
                args = ([self.scratch(k, 0)] if mode != "full" else []) + [self.bname(b) for b in kg.bufs.values()]
                passes.append((name, n, kg.parallel, args))
            self.kernel_names[k.id] = passes
        passes = self.kernel_names[k.id]
        out = []
        if k.uses_rand():                        # every pass draws from the kernel's one stream
            out.append(f"{ind}{{ uint32_t s_ = anvil_stream++;")
            stream, inner = "s_", ind + "  "
        else:
            stream, inner = "0", ind
        for name, n, parallel, args in passes:
            blocks = f"ANVIL_BLOCKS({n}L, {THREADS})" if parallel else "1"
            threads = f"ANVIL_TPB({THREADS})" if parallel else "1"
            out.append(f"{inner}ANVIL_LAUNCH({name}, {blocks}, {threads}, {stream}, anvil_seed_value"
                       f"{''.join(', ' + a for a in args)});")
        if k.uses_rand():
            out.append(f"{ind}}}")
        return out

    def scratch(self, k: ir.Kernel, numel: int) -> str:
        """Partial results of a split reduction."""
        name = f"part{k.id}"
        if name not in self.scratches:
            self.scratches[name] = (ctype(k.red.dtype), numel)
        return name

    def print_(self, st: ir.Print, ind) -> list[str]:
        f = "stderr" if st.err else "stdout"
        out = [f"{ind}anvil_check_faults();"]
        if st.err:
            out.append(f"{ind}fflush(stdout);")
        for it in list(st.items) + [ir.PrintItem("text", st.end)]:
            if it.kind == "text":
                if it.text:
                    out.append(f'{ind}fputs("{c_escape(it.text)}", {f});')
            elif it.kind == "scalar":
                b = self.bname(it.buf)
                if it.fmt:
                    isint = it.buf.dtype == I32 or it.fmt.endswith("d")
                    v = f"(int){b}[0]" if isint else f"(double){b}[0]"
                    out.append(f'{ind}fprintf({f}, "{c_escape(it.fmt)}", {v});')
                elif it.buf.dtype == I32:
                    out.append(f'{ind}fprintf({f}, "%d", {b}[0]);')
                else:
                    out.append(f"{ind}anvil_print_f32({f}, {b}[0]);")
            elif it.kind == "tensor":
                b = it.buf
                dims = self.table("long", [str(d) for d in b.shape] or ["0"])
                out.append(f"{ind}anvil_print_tensor({self.bname(b)}, {1 if b.dtype == I32 else 0}, {len(b.shape)}, {dims});")
            elif it.kind == "pick":
                tab = self.strings(it.choices)
                out.append(f"{ind}anvil_print_pick({f}, {self.bname(it.buf)}[0], {tab}, {len(it.choices)});")
            elif it.kind == "chars":
                out.append(f"{ind}anvil_print_chars({self.bname(it.buf)}, {it.buf.numel});")
            else:
                raise NotImplementedError(it.kind)
        return out

    def table(self, ctype_, items) -> str:
        name = self.label("tab")
        self.tables.append(f"static const {ctype_} {name}[] = {{{', '.join(items)}}};")
        return name

    def strings(self, strs) -> str:
        return self.table("char *const", [f'"{c_escape(s)}"' for s in strs])

    def rtcall(self, st: ir.RTCall) -> list[str]:
        a, n = st.args, st.name
        b = a.get("buf")
        if n == "load_idx":
            return [f'anvil_load_idx("{c_escape(a["path"])}", {self.bname(b)}, {b.numel}L, {a["header"]}L, {1 if b.dtype == I32 else 0});']
        if n == "load_csv":
            return [f'anvil_load_csv("{c_escape(a["path"])}", {self.bname(b)}, {b.numel}L, {a["skip"]}L);']
        if n == "load_npy":
            return [f'anvil_load_npy("{c_escape(a["path"])}", {self.bname(b)}, {b.numel}L, {a["offset"]}L, '
                    f'{a["kind"]}, {a["size"]});']
        if n == "save_npy":
            tab = self.table("unsigned char", [str(x) for x in a["header"]])
            return [f'anvil_save_npy("{c_escape(a["path"])}", {tab}, {len(a["header"])}L, {self.bname(b)}, {4 * b.numel}L);']
        if n == "load_bytes":
            return [f'anvil_load_bytes("{c_escape(a["path"])}", {self.bname(b)}, {b.numel}L);']
        if n == "seed":
            return [f"anvil_seed_value = {a['value'] & 0xFFFFFFFF}u; anvil_stream = 0;"]
        if n == "clock":
            return [f"{self.bname(b)}[0] = (float)(anvil_now() - anvil_t0);"]
        if n == "show":
            return [f"anvil_show({self.bname(b)}, {a['rows']}L, {a['cols']}L, {self.strings(a['glyphs'])}, {len(a['glyphs'])});"]
        if n == "sleep":
            return ["fflush(stdout);", f"if ({self.bname(b)}[0] > 0) usleep((useconds_t)({self.bname(b)}[0] * 1e6f));"]
        if n == "nonzero":
            src = a["src"]
            return [f"anvil_nonzero({self.bname(src)}, {src.numel}L, {self.bname(b)}, {b.numel}L);"]
        if n == "input":
            return [f"if (anvil_input({self.bname(b)})) goto anvil_end;"]
        if n in ("save", "load"):
            tab = self.label("ckpt")
            entries = ", ".join(f"{{{self.bname(x)}, {x.numel}L}}" for x in a["bufs"])
            self.ckpts.append(f"    anvil_ckpt {tab}[] = {{{entries}}};")
            if n == "save":
                return [f'anvil_save("{c_escape(a["path"])}", {tab}, {len(a["bufs"])}L);']
            return [f'{self.bname(b)}[0] = anvil_load("{c_escape(a["path"])}", {tab}, {len(a["bufs"])}L);']
        if n in ("iota", "iota_once"):
            return [f"anvil_iota({self.bname(b)}, {b.numel}L);"]
        if n == "shuffle":
            return [f"anvil_shuffle({self.bname(b)}, {b.numel}L);"]
        if n == "skip_rand":
            return ["anvil_stream++;"]
        if n == "exit":
            return ["fflush(stdout);", f"exit({a['value']});"]
        raise NotImplementedError(n)


def kernel_loads(k: ir.Kernel) -> list[Load]:
    """Every load in k, including the index loads of gathers (in expressions and in store offsets)."""
    out = []

    def offset(a: Affine):
        for g in a.gathers():
            out.append(g.load)
            offset(g.load.offset)

    for e in k.exprs():
        for x in ir.iter_expr(e):
            if isinstance(x, Load):
                out.append(x)
                offset(x.offset)
            elif isinstance(x, Index):
                offset(x.affine)
    for st in k.stores:
        offset(st.offset)
    return out


def injective_over(off: Affine, vars_) -> bool:
    """Different values of vars_ reach different offsets (every var must appear)."""
    items = []
    for v in vars_:
        c = off.coef(v)
        if v.extent > 1 and c == 0:
            return False
        if v.extent > 1:
            items.append((abs(c), v.extent))
    items.sort()
    reach = 0
    for c, n in items:
        if c <= reach:
            return False
        reach += c * (n - 1)
    return True


def generate(prog: ir.Program, seed: int = 0) -> str:
    return CudaGen(prog, seed).generate()


class CudaToolchainError(Exception):
    pass


def build(source: str, exe: str, emulate: bool):
    """Compile a generated .cu file: with nvcc, or as C++ on the CPU (emulate)."""
    with tempfile.TemporaryDirectory() as d:
        cu = os.path.join(d, "prog.cu")
        with open(cu, "w") as f:
            f.write(source)
        if emulate:
            cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
            if cxx is None:
                raise CudaToolchainError("no C++ compiler found for --cuda-emulate")
            cmd = [cxx, "-O1", "-std=c++17", "-fwrapv", "-w", "-DANVIL_EMULATE", "-x", "c++", cu, "-o", exe, "-lz"]
        else:
            nvcc = shutil.which("nvcc")
            if nvcc is None:
                raise CudaToolchainError("nvcc not found: install the CUDA toolkit, or use --cuda-emulate to run "
                                         "the CUDA code on the CPU (or `anvil cuda file.anvil` to write the .cu file)")
            cmd = [nvcc, "-O3", "-w", cu, "-o", exe, "-lz"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise CudaToolchainError(f"{os.path.basename(cmd[0])} failed:\n{r.stderr[-4000:]}")
