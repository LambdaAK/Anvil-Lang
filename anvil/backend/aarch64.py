"""Program-level AArch64 code generation (Apple / Mach-O).

Layout of the emitted file:
    _main                 control flow, prints, runtime calls, `bl` to kernels
    _anvil_kN ...           one leaf function per kernel (see kernel.py)
    runtime               hand-written support routines (runtime.s)
    data                  constant tensors, strings, and __bss buffers

Calling convention: kernels may clobber every register except sp, x18, x29, x30. `main`
saves all callee-saved registers on entry and never keeps values in registers across a
kernel call.
"""
from __future__ import annotations

import os
import re

from .. import ir
from ..ir import F32, I32, Buffer, KernelStmt
from .arena import Plan, plan
from .kernel import Lowering, choose_configs, f32_bits
from .mir import RegAllocFail, allocate, allocate_with_spills, render

RUNTIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime.s")
VARARG_SLOTS = 32
HEAP_MIN = 64 << 20             # tensors at least this large are allocated when the program starts
HEAP_KINDS = ("data", "temp", "var", "grad", "scalar")
PAR_MIN_WORK = 1 << 17          # loop iterations below which threading does not pay off
PAR_CHUNK_WORK = 1 << 15        # minimum iterations per chunk
PAR_CHUNKS = 32                 # target number of chunks (~4 per thread at 8 threads)
GATHER_WORK = 16                # a gathered load counts as this many plain operations


def parallel_grain(k: ir.Kernel) -> int | None:
    """Chunk size (in outermost-loop iterations) if k may run in parallel, else None.

    Safe when every store writes a region of memory that is disjoint for different values of
    the outermost loop variable, and the kernel does not draw random numbers (their stream
    counter is advanced once per call)."""
    if not k.domain or k.uses_rand():
        return None
    outer = k.domain[0]
    n = outer.extent
    work = kernel_work(k)
    if n < 2 or work < PAR_MIN_WORK:
        return None
    for st in k.stores:
        off = st.offset
        if off.gathers():
            return None
        c = abs(off.coef(outer))
        if c == 0:
            return None
        span = sum(abs(off.coef(v)) * (v.extent - 1) for v in off.direct_vars() if v is not outer)
        if span >= c:
            return None
    per = max(1, work // n)
    grain = max(-(-n // PAR_CHUNKS), -(-PAR_CHUNK_WORK // per))
    align = 16 if len(k.domain) == 1 else 4          # keep vector / register-tile steps whole
    grain = -(-grain // align) * align
    return grain if grain < n else None


def kernel_work(k: ir.Kernel) -> int:
    """Loop iterations, weighted by how much each one does: an optimizer update (four loads, three
    stores, a square root and a division per element) is worth threads at a tenth of the size of
    a copy, and so is a copy of gathered rows (a batch). A plain iteration (a load, an operation
    and a store, or a multiply-add) counts 1."""
    def nodes(e):
        return sum(1 for _ in ir.iter_expr(e))
    def gathers(e):                 # a gathered load waits on memory: worth many plain ones
        return sum(len(x.offset.gathers()) for x in ir.iter_expr(e) if isinstance(x, ir.Load))
    out = ir.prod(v.extent for v in k.domain)
    per = sum(nodes(st.value) + 1 + GATHER_WORK * gathers(st.value) for st in k.stmts)
    if k.red is not None:
        per += nodes(k.red.body) * ir.prod(v.extent for v in k.red.vars)
    return out * max(1, per // 4)


def check_description(k: ir.Kernel) -> str:
    """Where a kernel comes from, for --check: the source line and the computation."""
    head = f"  made by `{k.label or 'a computation'}`"
    line = ""
    if k.span is not None:
        head += f" at {k.span.location()}"
        line = "\n      " + k.span.file.line_text(k.span.line).strip()
    return f"{head}:{line}\n  computing ({k.name}):  {ir.fmt_kernel(k)}"


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s or "t"


def c_escape(text: str) -> str:
    out = []
    for b in text.encode("utf-8"):
        ch = chr(b)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif 32 <= b < 127:
            out.append(ch)
        else:
            out.append(f"\\{b:03o}")
    return "".join(out)


class KernelInfo:
    def __init__(self, k, cfg, lines):
        self.k = k
        self.cfg = cfg
        self.lines = lines


class ProgramGen:
    def __init__(self, prog: ir.Program, seed: int = 0, profile: bool = False, threads: bool = True,
                 share: bool = True, library: bool = False, check: str | None = None):
        self.prog = prog
        self.check = check          # "nan" | "inf": stop at the first nan (or infinity) a kernel makes
        if check:
            share = False           # every tensor keeps its own memory, so its values can be shown
        self.check_records: list = []
        self.blobs: list = []           # (label, bytes): constant data such as .npy headers
        from .blas import prepare
        self.gemms = prepare(prog)      # large matrix products go to Accelerate (cblas_sgemm)
        for b in prog.buffers:          # large tensors live on the heap (see ir.indirect)
            if b.root is b and b.kind in HEAP_KINDS and b.init is None and b.nbytes >= HEAP_MIN:
                b.heap = True
        self.library = library      # a shared library: `_anvil_entry` runs the program, and returns
        self.entry = "_anvil_entry" if library else "_main"
        self.share = share          # temporaries share memory (see arena.py)
        self.seed = seed
        self.profile = profile
        self.threads = threads
        self.par_grain: dict[int, int] = {}
        self.glyph_tables: dict = {}
        self.ckpt_tables: dict = {}             # save / load: {data, key} per tensor (ir.ckpt_key)
        self.prof_ids: dict[int, int] = {}
        self.prof_kernels: dict[int, ir.Kernel] = {}
        self.out: list[str] = []
        self.strings: dict[str, str] = {}
        self.dims: dict[tuple, str] = {}
        self.used: dict[int, Buffer] = {}
        self.label_n = 0
        self.loops: list[tuple[str, str]] = []
        self.kernels: list[KernelInfo] = []
        self.kernel_done: set[int] = set()

    # ------------------------------------------------------------------ symbols & data
    def sym(self, buf: Buffer) -> str:
        r = buf.root
        self.used[r.id] = r
        if r.kind in ir.EXTERN_KINDS:
            return f"_anvil_{r.name}"              # input0, output1, …: found by name in the library
        return f"_anvil_{sanitize(r.name)}_{r.id}"

    def cstring(self, text: str) -> str:
        lab = self.strings.get(text)
        if lab is None:
            lab = f"Lstr{len(self.strings)}"
            self.strings[text] = lab
        return lab

    def dims_label(self, shape) -> str:
        key = tuple(shape)
        lab = self.dims.get(key)
        if lab is None:
            lab = f"Ldims{len(self.dims)}"
            self.dims[key] = lab
        return lab

    def glyph_table(self, glyphs) -> str:
        key = tuple(glyphs)
        lab = self.glyph_tables.get(key)
        if lab is None:
            lab = f"Lglyphs{len(self.glyph_tables)}"
            self.glyph_tables[key] = (lab, [self.cstring(g) for g in glyphs] + [self.cstring("?")])
            return lab
        return lab[0]

    def ckpt_table(self, bufs) -> str:
        key = tuple(b.root.id for b in bufs)
        hit = self.ckpt_tables.get(key)
        if hit is None:
            if any(b.root.kind in ir.EXTERN_KINDS for b in bufs):
                raise NotImplementedError("save/load of a function's arguments or results")
            hit = (f"Lckpt{len(self.ckpt_tables)}", [(self.sym(b), ir.ckpt_key(b.root)) for b in bufs])
            self.ckpt_tables[key] = hit
        return hit[0]

    def label(self, hint: str) -> str:
        self.label_n += 1
        return f"L{hint}{self.label_n}"

    def emit(self, line: str):
        self.out.append(line)

    def ins(self, text: str, comment: str = ""):
        line = f"    {text}"
        if comment:
            line = f"{line:<44}// {comment}"
        self.out.append(line)

    def imm(self, reg: str, value: int):
        w = reg.startswith("w")
        bits = 32 if w else 64
        value &= (1 << bits) - 1
        if value == 0:
            self.ins(f"mov {reg}, #0")
            return
        chunks = [(value >> (16 * i)) & 0xFFFF for i in range(bits // 16)]
        first = True
        for i, c in enumerate(chunks):
            if c == 0:
                continue
            self.ins(f"movz {reg}, #{c}" + (f", lsl #{16 * i}" if i else "") if first else
                     f"movk {reg}, #{c}, lsl #{16 * i}")
            first = False
        if first:
            self.ins(f"movz {reg}, #0")

    def addr(self, reg: str, buf: Buffer):
        s = self.sym(buf)
        self.ins(f"adrp {reg}, {s}@PAGE")
        if ir.indirect(buf):
            self.ins(f"ldr {reg}, [{reg}, {s}@PAGEOFF]", f"{buf.root.name}: through its pointer")
        else:
            self.ins(f"add {reg}, {reg}, {s}@PAGEOFF")

    def mem(self, buf: Buffer) -> str:
        """Point x16 at buf's first element; returns the memory operand for it."""
        s = self.sym(buf)
        self.ins(f"adrp x16, {s}@PAGE")
        if ir.indirect(buf):
            self.ins(f"ldr x16, [x16, {s}@PAGEOFF]", f"{buf.root.name}: through its pointer")
            return "[x16]"
        return f"[x16, {s}@PAGEOFF]"

    def load_w(self, reg: str, buf: Buffer):
        s = self.mem(buf)
        self.ins(f"ldr {reg}, {s}")

    def store_w(self, reg: str, buf: Buffer):
        s = self.mem(buf)
        self.ins(f"str {reg}, {s}")

    def load_int(self, reg: str, x):
        if isinstance(x, int):
            self.imm(reg, x)
        else:
            self.load_w(reg, x)

    def str_label(self, reg: str, lab: str):
        self.ins(f"adrp {reg}, {lab}@PAGE")
        self.ins(f"add {reg}, {reg}, {lab}@PAGEOFF")

    # ------------------------------------------------------------------ main
    def generate(self) -> str:
        src = self.prog.source
        name = src.display_path() if src is not None else "<program>"
        head = [
            f"// Generated by anvil from {name} — AArch64 (Apple silicon, Mach-O).",
            "// Assemble with:  cc -arch arm64 program.s -lz -o program",
            "",
            "    .section __TEXT,__text,regular,pure_instructions",
            f"    .globl {self.entry}",
            "    .p2align 2",
            f"{self.entry}:",
        ]
        self.out = []
        self.ins("stp x29, x30, [sp, #-16]!")
        self.ins("mov x29, sp")
        self.ins("stp x19, x20, [sp, #-16]!")
        self.ins("stp x21, x22, [sp, #-16]!")
        self.ins("stp x23, x24, [sp, #-16]!")
        self.ins("stp x25, x26, [sp, #-16]!")
        self.ins("stp x27, x28, [sp, #-16]!")
        self.ins("stp d8, d9, [sp, #-16]!")
        self.ins("stp d10, d11, [sp, #-16]!")
        self.ins("stp d12, d13, [sp, #-16]!")
        self.ins("stp d14, d15, [sp, #-16]!")
        self.ins(f"sub sp, sp, #{8 * VARARG_SLOTS}", "space for printf's variadic arguments")
        if self.library:                        # a function called again and again: start the threads once
            done = self.label("ready")
            self.ins("adrp x9, _anvil_lib_ready@PAGE")
            self.ins("ldr w10, [x9, _anvil_lib_ready@PAGEOFF]")
            self.ins(f"cbnz w10, {done}")
            self.ins("mov w10, #1")
            self.ins("str w10, [x9, _anvil_lib_ready@PAGEOFF]")
            self.ins("bl _anvil_rt_init")
            self.ins("bl _anvil_alloc_heap")
            self.emit(f"{done}:")
            self.imm("w0", self.seed)
            self.ins("bl _anvil_rt_seed", "every call draws the same random numbers")
        else:
            self.ins("bl _anvil_rt_init")
            self.ins("bl _anvil_alloc_heap", "the large tensors")
        if self.seed and not self.library:
            self.imm("w0", self.seed)
            self.ins("bl _anvil_rt_seed")
        self.block(self.prog.main)
        self.emit("Lanvil_end:")                  # the end of the script (or of its input)
        if self.profile and self.prof_ids:
            self.ins("adrp x0, _anvil_prof_tab@PAGE")
            self.ins("add x0, x0, _anvil_prof_tab@PAGEOFF")
            self.imm("x1", len(self.prof_ids))
            self.ins("bl _anvil_rt_profile_report")
        self.ins("mov x0, #0")
        self.ins("bl _fflush")
        self.ins(f"add sp, sp, #{8 * VARARG_SLOTS}")
        self.ins("ldp d14, d15, [sp], #16")
        self.ins("ldp d12, d13, [sp], #16")
        self.ins("ldp d10, d11, [sp], #16")
        self.ins("ldp d8, d9, [sp], #16")
        self.ins("ldp x27, x28, [sp], #16")
        self.ins("ldp x25, x26, [sp], #16")
        self.ins("ldp x23, x24, [sp], #16")
        self.ins("ldp x21, x22, [sp], #16")
        self.ins("ldp x19, x20, [sp], #16")
        self.ins("ldp x29, x30, [sp], #16")
        self.ins("mov w0, #0")
        self.ins("ret")
        main = head + self.out
        parts = ["\n".join(main), self.heap_allocator()]
        for ki in self.kernels:
            parts.append("\n".join(ki.lines))
        with open(RUNTIME_PATH) as f:
            parts.append(f.read())
        parts.append(self.data_section())
        return "\n\n".join(parts) + "\n"

    def block(self, b: ir.Block):
        for st in b.stmts:
            self.stmt(st)

    def stmt(self, st):
        if isinstance(st, KernelStmt):
            k = st.kernel
            loc = f"  {k.span.location()}" if k.span is not None else ""
            if k.id in self.gemms:
                call = (lambda: self.call_gemm(self.gemms[k.id], k, loc))
            else:
                self.kernel_fn(k)
                grain = self.par_grain.get(k.id)
                call = (lambda: self.call_kernel(k, grain, loc))
            if self.profile:
                slot = self.prof_ids.setdefault(k.id, len(self.prof_ids))
                self.prof_kernels[k.id] = k
                self.ins("isb")
                self.ins("mrs x9, cntvct_el0")
                self.ins("adrp x16, _anvil_prof_t@PAGE")
                self.ins("str x9, [x16, _anvil_prof_t@PAGEOFF]")
                call()
                self.ins("isb")
                self.ins("mrs x10, cntvct_el0")
                self.ins("adrp x16, _anvil_prof_t@PAGE")
                self.ins("ldr x9, [x16, _anvil_prof_t@PAGEOFF]")
                self.ins("sub x10, x10, x9")
                self.ins("adrp x16, _anvil_prof_tab@PAGE")
                self.ins("add x16, x16, _anvil_prof_tab@PAGEOFF")
                self.imm("x17", 24 * slot + 8)
                self.ins("add x16, x16, x17")
                self.ins("ldp x11, x12, [x16]")
                self.ins("add x11, x11, x10")
                self.ins("add x12, x12, #1")
                self.ins("stp x11, x12, [x16]")
            else:
                call()
            if self.check:
                self.emit_checks(k)
        elif isinstance(st, ir.For):
            self.stmt_for(st)
        elif isinstance(st, ir.While):
            self.stmt_while(st)
        elif isinstance(st, ir.If):
            self.stmt_if(st)
        elif isinstance(st, ir.Break):
            self.ins(f"b {self.loops[-1][1]}", "break")
        elif isinstance(st, ir.Continue):
            self.ins(f"b {self.loops[-1][0]}", "continue")
        elif isinstance(st, ir.Print):
            self.stmt_print(st)
        elif isinstance(st, ir.Check):
            self.stmt_check(st)
        elif isinstance(st, ir.RTCall):
            self.stmt_rt(st)
        else:
            raise NotImplementedError(type(st).__name__)

    def heap_allocator(self) -> str:
        """_anvil_alloc_heap: calloc each large tensor (pages are zero, and only touched ones cost)."""
        lines = ["// the large tensors, allocated when the program starts", "    .p2align 2", "_anvil_alloc_heap:",
                 "    stp x29, x30, [sp, #-16]!", "    mov x29, sp"]
        for b in sorted(self.used.values(), key=lambda b: b.id):
            if getattr(b, "heap", False):
                lines += ["    mov x0, #1"]
                v = (b.nbytes + 15) // 16 * 16
                chunks = [(v >> (16 * i)) & 0xFFFF for i in range(4)]
                first = True
                for i, c in enumerate(chunks):
                    if c:
                        lines.append(f"    {'movz' if first else 'movk'} x1, #{c}" + (f", lsl #{16 * i}" if i else ""))
                        first = False
                lines += ["    bl _calloc", "    cbz x0, Lheap_fail",
                          f"    adrp x9, {self.sym(b)}@PAGE", f"    str x0, [x9, {self.sym(b)}@PAGEOFF]   // {b.name}"]
        lines += ["    ldp x29, x30, [sp], #16", "    ret",
                  "Lheap_fail:", "    adrp x0, Lheap_msg@PAGE", "    add x0, x0, Lheap_msg@PAGEOFF", "    bl _puts",
                  "    mov w0, #1", "    bl _exit",
                  "    .section __TEXT,__cstring,cstring_literals",
                  'Lheap_msg: .asciz "anvil: out of memory (allocating a large tensor)"',
                  "    .section __TEXT,__text,regular,pure_instructions"]
        return "\n".join(lines)

    def emit_checks(self, k: ir.Kernel):
        """--check: after k, look for a nan (or an infinity) in each f32 tensor it wrote."""
        written = []
        for st in k.stores:
            b = st.buf.root
            if b.dtype == F32 and b.kind not in ir.EXTERN_KINDS and b not in written:
                written.append(b)
        if not written:
            return
        desc = check_description(k)
        inputs = [b for b in sorted(k.reads(), key=lambda b: b.id)
                  if b.dtype == F32 and b not in written and not ir.indirect(b) and b.numel > 0]
        for b in written:
            rec = f"Lchkrec{len(self.check_records)}"
            name = f"`{b.name}` ({b.dtype}{list(b.shape)})" if b.shape else f"`{b.name}` (an f32 scalar)"
            self.check_records.append((rec, self.cstring(name), self.dims_label(b.shape) if b.shape else None,
                                       len(b.shape), self.cstring(desc),
                                       [(self.sym(x), x.numel, self.cstring(f"`{x.name}`")) for x in inputs]))
            self.addr("x0", b)
            self.imm("x1", b.numel)
            self.str_label("x2", rec)
            self.imm("x3", 2 if self.check == "inf" else 1)
            self.ins("bl _anvil_rt_check", f"--check {b.name}")

    def call_gemm(self, g, k: ir.Kernel, loc: str):
        """cblas_sgemm(RowMajor, transA, transB, M, N, K, 1, A, lda, B, ldb, beta, C, ldc), once per
        element of the batch loops. The last four arguments go on the stack, each at its natural
        alignment (Apple's arm64 convention), in the outgoing-argument area main keeps at the bottom
        of its frame; the batch counters sit above them, at [sp, #32 + 8·d]."""
        loops = g.loops
        tops = []
        for d, lp in enumerate(loops):
            self.ins(f"str xzr, [sp, #{32 + 8 * d}]", f"batch index {d} (of {lp.extent})")
            top = self.label("gemm")
            self.emit(f"{top}:")
            tops.append(top)

        def address(reg, buf, off, which):
            self.addr(reg, buf)
            if off:
                self.imm("x16", 4 * off)
                self.ins(f"add {reg}, {reg}, x16")
            for d, lp in enumerate(loops):
                step = getattr(lp, which)
                if step:
                    self.ins(f"ldr x10, [sp, #{32 + 8 * d}]")
                    self.imm("x11", 4 * step)
                    self.ins(f"madd {reg}, x10, x11, {reg}")
        address("x9", g.B, g.b_off, "b")
        self.ins("str x9, [sp]")
        self.imm("w10", g.ldb)
        self.ins("str w10, [sp, #8]")
        address("x9", g.C, g.c_off, "c")
        self.ins("str x9, [sp, #16]")
        self.imm("w10", g.ldc)
        self.ins("str w10, [sp, #24]")
        address("x6", g.A, g.a_off, "a")
        summed = [d for d, lp in enumerate(loops) if lp.summed]
        if summed:                     # the first product of a sum stores (or adds, for +=), the rest add
            self.ins("mov x12, #0")
            for d in summed:
                self.ins(f"ldr x10, [sp, #{32 + 8 * d}]")
                self.ins("orr x12, x12, x10")
            self.ins("fmov s1, #1.0")
            if not g.beta:
                first = self.label("gemm_first")
                self.ins(f"cbnz x12, {first}")
                self.ins("movi d1, #0")
                self.emit(f"{first}:")
        else:
            self.ins("fmov s1, #1.0" if g.beta else "movi d1, #0")
        self.imm("w0", 101)                                     # CblasRowMajor
        self.imm("w1", 112 if g.trans_a else 111)               # CblasTrans / CblasNoTrans
        self.imm("w2", 112 if g.trans_b else 111)
        self.imm("w3", g.M)
        self.imm("w4", g.N)
        self.imm("w5", g.K)
        self.ins("fmov s0, #1.0")
        self.imm("w7", g.lda)
        batch = f" × {g.calls}" if loops else ""
        self.ins("bl _cblas_sgemm", f"{k.label or 'matmul'}{loc}  ({g.M}×{g.K} @ {g.K}×{g.N}{batch} on the AMX coprocessor)")
        for d in reversed(range(len(loops))):
            self.ins(f"ldr x10, [sp, #{32 + 8 * d}]")
            self.ins("add x10, x10, #1")
            self.ins(f"str x10, [sp, #{32 + 8 * d}]")
            self.imm("x11", loops[d].extent)
            self.ins("cmp x10, x11")
            self.ins(f"b.lt {tops[d]}")

    def call_kernel(self, k: ir.Kernel, grain, loc: str):
        if grain is None:
            self.ins(f"bl _anvil_{k.name}", f"{k.label or 'kernel'}{loc}")
            return
        self.ins(f"adrp x0, _anvil_{k.name}@PAGE")
        self.ins(f"add x0, x0, _anvil_{k.name}@PAGEOFF")
        self.imm("x1", k.domain[0].extent)
        self.imm("x2", grain)
        self.ins("bl _anvil_rt_parallel", f"{k.label or 'kernel'}{loc}  (parallel, chunks of {grain})")

    def stmt_for(self, st: ir.For):
        top = self.label("for")
        cont = top + "_next"
        end = top + "_end"
        loc = f"  ({st.span.location()})" if st.span is not None else ""
        self.emit(f"    // for {st.counter.name} in range(...){loc}")
        self.load_int("w9", st.start)
        self.store_w("w9", st.counter)
        self.emit(f"{top}:")
        self.load_w("w9", st.counter)
        self.load_int("w10", st.stop)
        self.ins("cmp w9, w10")
        self.ins(f"b.ge {end}")
        self.loops.append((cont, end))
        self.block(st.body)
        self.loops.pop()
        self.emit(f"{cont}:")
        self.load_w("w9", st.counter)
        self.ins(f"add w9, w9, #{st.step}")
        self.store_w("w9", st.counter)
        self.ins(f"b {top}")
        self.emit(f"{end}:")

    def test_zero(self, buf: Buffer, target: str):
        """Branch to target if the scalar in buf is zero."""
        s = self.mem(buf)
        if buf.dtype == I32:
            self.ins(f"ldr w9, {s}")
            self.ins(f"cbz w9, {target}")
        else:
            self.ins(f"ldr s0, {s}")
            self.ins("fcmp s0, #0.0")
            self.ins(f"b.eq {target}")

    def stmt_while(self, st: ir.While):
        top = self.label("while")
        end = top + "_end"
        self.emit(f"{top}:")
        self.block(st.cond_block)
        self.test_zero(st.cond, end)
        self.loops.append((top, end))
        self.block(st.body)
        self.loops.pop()
        self.ins(f"b {top}")
        self.emit(f"{end}:")

    def stmt_if(self, st: ir.If):
        lab = self.label("if")
        els = lab + "_else"
        end = lab + "_end"
        self.test_zero(st.cond, els if st.orelse.stmts else end)
        self.block(st.then)
        if st.orelse.stmts:
            self.ins(f"b {end}")
            self.emit(f"{els}:")
            self.block(st.orelse)
        self.emit(f"{end}:")

    def stmt_print(self, st: ir.Print):
        fmt: list[str] = []
        args: list[Buffer] = []

        def flush():
            if not fmt and not args:
                return
            text = "".join(fmt)
            for i, b in enumerate(args):
                if isinstance(b, ir.PrintItem):       # a pick: the address of the chosen string
                    s = self.mem(b.buf)
                    self.ins(f"ldrsw x9, {s}")
                    self.imm("x10", len(b.choices))
                    self.ins("cmp x9, x10")
                    self.ins("csel x9, x10, x9, hs")       # out of range: the table's "?"
                    self.str_label("x11", self.glyph_table(b.choices))
                    self.ins("ldr x9, [x11, x9, lsl #3]")
                    self.ins(f"str x9, [sp, #{8 * i}]")
                    continue
                s = self.mem(b)
                if b.dtype == I32:
                    self.ins(f"ldrsw x9, {s}")
                    self.ins(f"str x9, [sp, #{8 * i}]")
                else:
                    self.ins(f"ldr s0, {s}")
                    self.ins("fcvt d0, s0")
                    self.ins(f"str d0, [sp, #{8 * i}]")
            if st.err:
                self.ins("mov x0, #0")
                self.ins("bl _fflush", "what was printed so far comes first")
                self.ins("adrp x8, ___stderrp@GOTPAGE")
                self.ins("ldr x8, [x8, ___stderrp@GOTPAGEOFF]")
                self.ins("ldr x0, [x8]")
                self.str_label("x1", self.cstring(text))
                self.ins("bl _fprintf")
            else:
                self.str_label("x0", self.cstring(text))
                self.ins("bl _printf")
            fmt.clear()
            args.clear()
        items = list(st.items) + [ir.PrintItem("text", st.end)]
        for it in items:
            if it.kind == "text":
                fmt.append(it.text.replace("%", "%%"))
            elif it.kind == "scalar" and not it.fmt:
                flush()
                s_ = self.mem(it.buf)
                if it.buf.dtype == I32:
                    self.ins(f"ldrsw x9, {s_}")
                    self.ins("str x9, [sp]")
                    self.str_label("x0", self.cstring("%d"))
                    self.ins("bl _printf")
                else:
                    self.ins(f"ldr s0, {s_}")
                    self.ins("bl _anvil_rt_print_f32")
            elif it.kind == "scalar":
                if len(args) >= VARARG_SLOTS:
                    flush()
                fmt.append(it.fmt)
                args.append(it.buf)
            elif it.kind == "chars":
                flush()
                self.addr("x0", it.buf)
                self.imm("x1", it.buf.numel)
                self.ins("bl _anvil_rt_print_chars")
            elif it.kind == "pick":
                if len(args) >= VARARG_SLOTS:
                    flush()
                fmt.append("%s")
                args.append(it)
            else:
                flush()
                b = it.buf
                self.addr("x0", b)
                self.ins(f"mov x1, #{1 if b.dtype == I32 else 0}")
                self.ins(f"mov x2, #{len(b.shape)}")
                self.str_label("x3", self.dims_label(b.shape))
                self.ins("bl _anvil_rt_print_tensor")
        flush()

    def stmt_check(self, st: ir.Check):
        ok = self.label("ok")
        bad = ok + "_bad"
        self.imm("x9", st.value.const)
        for t, c in st.value.terms.items():
            if isinstance(t, ir.ScalarRef):
                s = self.mem(t.buf)
                self.ins(f"ldrsw x10, {s}")
                self.imm("x11", c)
                self.ins("madd x9, x10, x11, x9")
        self.imm("x10", st.lo)
        self.ins("cmp x9, x10")
        self.ins(f"b.lt {bad}")
        self.imm("x10", st.hi)
        self.ins("cmp x9, x10")
        self.ins(f"b.le {ok}")
        self.emit(f"{bad}:")
        loc = f"{st.span.location()}: " if st.span is not None else ""
        self.str_label("x0", self.cstring(loc + st.message))
        self.ins("mov x1, x9")
        self.ins("bl _anvil_rt_oob")
        self.emit(f"{ok}:")

    def stmt_rt(self, st: ir.RTCall):
        a = st.args
        if st.name == "load_idx":
            b = a["buf"]
            self.str_label("x0", self.cstring(a["path"]))
            self.addr("x1", b)
            self.imm("x2", b.numel)
            self.imm("x3", a["header"])
            self.ins(f"mov x4, #{1 if b.dtype == I32 else 0}")
            self.str_label("x5", self.dims_label(a["dims"]))
            self.ins(f"mov x6, #{len(a['dims'])}")
            self.ins("bl _anvil_rt_load_idx", f"load {os.path.basename(a['path'])}")
        elif st.name == "exit":
            self.imm("w0", a["value"])
            self.ins("bl _exit", "a failed assertion")
        elif st.name == "skip_rand":
            self.ins("adrp x9, _anvil_stream@PAGE", "a removed kernel's random numbers")
            self.ins("ldr w10, [x9, _anvil_stream@PAGEOFF]")
            self.ins("add w10, w10, #1")
            self.ins("str w10, [x9, _anvil_stream@PAGEOFF]")
        elif st.name == "load_bytes":
            b = a["buf"]
            self.str_label("x0", self.cstring(a["path"]))
            self.addr("x1", b)
            self.imm("x2", b.numel)
            self.ins("bl _anvil_rt_load_bytes", f"load {os.path.basename(a['path'])}")
        elif st.name == "load_npy":
            b = a["buf"]
            self.str_label("x0", self.cstring(a["path"]))
            self.addr("x1", b)
            self.imm("x2", b.numel)
            self.imm("x3", a["offset"])
            self.imm("x4", a["kind"])
            self.imm("x5", a["size"])
            self.ins("bl _anvil_rt_load_npy", f"load {os.path.basename(a['path'])}")
        elif st.name == "save_npy":
            b = a["buf"]
            lab = f"Lnpyhdr{len(self.blobs)}"
            self.blobs.append((lab, a["header"]))
            self.str_label("x0", self.cstring(a["path"]))
            self.str_label("x1", lab)
            self.imm("x2", len(a["header"]))
            self.addr("x3", b)
            self.imm("x4", 4 * b.numel)
            self.ins("bl _anvil_rt_save_npy", f"save {os.path.basename(a['path'])}")
        elif st.name == "load_csv":
            b = a["buf"]
            self.str_label("x0", self.cstring(a["path"]))
            self.addr("x1", b)
            self.imm("x2", b.numel)
            self.imm("x3", a["skip"])
            self.ins("bl _anvil_rt_load_csv", f"load {os.path.basename(a['path'])}")
        elif st.name == "seed":
            self.imm("w0", a["value"])
            self.ins("bl _anvil_rt_seed")
        elif st.name in ("iota", "iota_once"):
            b = a["buf"]
            self.addr("x0", b)
            self.imm("x1", b.numel)
            self.ins("bl _anvil_rt_iota")
        elif st.name == "shuffle":
            b = a["buf"]
            self.addr("x0", b)
            self.imm("x1", b.numel)
            self.ins("bl _anvil_rt_shuffle")
        elif st.name == "clock":
            self.addr("x0", a["buf"])
            self.ins("bl _anvil_rt_clock")
        elif st.name == "show":
            table = self.glyph_table(a["glyphs"])
            self.addr("x0", a["buf"])
            self.imm("x1", a["rows"])
            self.imm("x2", a["cols"])
            self.str_label("x3", table)
            self.imm("x4", len(a["glyphs"]))
            self.ins("bl _anvil_rt_show")
        elif st.name == "sleep":
            s_ = self.mem(a["buf"])
            self.ins(f"ldr s0, {s_}")
            self.ins("bl _anvil_rt_sleep")
        elif st.name in ("save", "load"):
            self.str_label("x0", self.cstring(a["path"]))
            self.str_label("x1", self.ckpt_table(a["bufs"]))
            self.imm("x2", len(a["bufs"]))
            self.ins(f"bl _anvil_rt_{st.name}", f"{st.name} {os.path.basename(a['path'])}")
            if st.name == "load":
                self.ins("ucvtf s0, w0")
                s_ = self.mem(a["buf"])
                self.ins(f"str s0, {s_}")
        elif st.name == "input":
            self.addr("x0", a["buf"])
            self.ins("bl _anvil_rt_input")
            self.ins("cbnz w0, Lanvil_end", "the input has ended: so does the program")
        elif st.name == "nonzero":
            self.addr("x0", a["src"])
            self.imm("x1", a["src"].numel)
            self.addr("x2", a["buf"])
            self.imm("x3", a["buf"].numel)
            self.ins("bl _anvil_rt_nonzero")
        else:
            raise NotImplementedError(st.name)

    # ------------------------------------------------------------------ kernels
    def kernel_fn(self, k: ir.Kernel):
        if k.id in self.kernel_done:
            return
        self.kernel_done.add(k.id)
        cfgs = choose_configs(k)
        grain = parallel_grain(k) if self.threads else None
        if grain is not None:
            self.par_grain[k.id] = grain
            for c in cfgs:
                c.parallel = True
        for cfg in cfgs:
            try:
                low = Lowering(k, cfg, self.sym, self.cstring_fail)
                items = low.lower()
                phys = allocate(items)
            except RegAllocFail:
                continue
            self.finish_kernel(k, cfg, items, phys, 0)
            return
        # nothing fits in registers: use the leanest schedule and spill to the stack
        cfg = cfgs[-1]
        items = Lowering(k, cfg, self.sym, self.cstring_fail).lower()
        items, phys, frame = allocate_with_spills(items)
        self.finish_kernel(k, cfg, items, phys, frame)

    def finish_kernel(self, k, cfg, items, phys, frame):
        body = render(items, phys)
        pro = [f"    sub sp, sp, #{frame}"] if frame else []
        epi = [f"    add sp, sp, #{frame}"] if frame else []
        lines = self.kernel_header(k, cfg, frame) + [f"_anvil_{k.name}:"] + pro + body + epi + ["    ret"]
        self.kernels.append(KernelInfo(k, cfg, lines))

    def cstring_fail(self, text: str) -> str:
        return self.cstring(text)

    def kernel_header(self, k: ir.Kernel, cfg, frame: int = 0) -> list[str]:
        loc = k.span.location() if k.span is not None else ""
        title = f"// ── {k.name} · {k.label or 'kernel'}" + (f" · {loc}" if loc else "") + " "
        title = title + "─" * max(4, 88 - len(title))
        lines = ["    .p2align 4", title]
        text = ir.fmt_kernel(k)
        for part in text.split("; "):
            lines.append(f"//   {part}")
        dom = " × ".join(f"{v.name}<{v.extent}" for v in k.domain) or "scalar"
        red = ""
        if k.red is not None:
            red = "  reduce " + " × ".join(f"{v.name}<{v.extent}" for v in k.red.vars)
        spills = f"  ·  {frame // 16} values spilled to the stack" if frame else ""
        lines.append(f"//   loops: {dom}{red}  ·  {cfg.describe()}{spills}")
        return lines

    # ------------------------------------------------------------------ data
    def data_section(self) -> str:
        out = []
        prof_names = {}
        if self.profile and self.prof_ids:
            for k in self.prof_kernels.values():
                loc = f"{os.path.basename(k.span.file.path)}:{k.span.line}" if k.span is not None else ""
                label = (k.label or "kernel") + (" (AMX)" if k.id in self.gemms else "")
                prof_names[self.prof_ids[k.id]] = self.cstring(f"{k.name:<6} {label[:34]:<34} {loc}")
        if self.strings:
            out.append("    .section __TEXT,__cstring,cstring_literals")
            for text, lab in self.strings.items():
                out.append(f'{lab}: .asciz "{c_escape(text)}"')
        if self.blobs:
            out.append("    .section __TEXT,__const")
            for lab, data in self.blobs:
                out.append(f"{lab}:")
                for i in range(0, len(data), 16):
                    out.append("    .byte " + ", ".join(str(x) for x in data[i:i + 16]))
        if self.check_records:
            out.append("    .section __DATA,__const")
            out.append("    .p2align 3")
            for rec, name, dims, rank, desc, inputs in self.check_records:
                out.append(f"{rec}: .quad {name}, {dims or 0}, {rank}, {desc}, {len(inputs)}")
                for sym, n, nm in inputs:
                    out.append(f"    .quad {sym}, {n}, {nm}")
        if self.glyph_tables or self.ckpt_tables:
            out.append("    .section __DATA,__const")
            out.append("    .p2align 3")
            for lab, strs in self.glyph_tables.values():
                out.append(f"{lab}: .quad " + ", ".join(strs))
            for lab, entries in self.ckpt_tables.values():
                out.append(f"{lab}: .quad " + ", ".join(f"{s}, {n}" for s, n in entries))
        if self.dims:
            out.append("    .section __TEXT,__const")
            out.append("    .p2align 3")
            for shape, lab in self.dims.items():
                vals = ", ".join(str(d) for d in shape) or "0"
                out.append(f"{lab}: .quad {vals}")
        consts = [b for b in self.used.values() if b.kind == "const"]
        if consts:
            out.append("    .section __DATA,__data")
            for b in consts:
                out.append("    .p2align 4")
                out.append(f"{self.sym(b)}:  // {b.name}{list(b.shape)}")
                words = [f32_bits(float(v)) if b.dtype == F32 else (int(v) & 0xFFFFFFFF) for v in b.init]
                for i in range(0, len(words), 8):
                    out.append("    .long " + ", ".join(f"0x{w:08x}" for w in words[i:i + 8]))
        if prof_names:
            out.append("    .section __DATA,__data")
            out.append("    .p2align 3")
            out.append("_anvil_prof_t: .quad 0")
            out.append("_anvil_prof_tab:   // {name, ticks, calls} per kernel")
            for i in range(len(self.prof_ids)):
                out.append(f"    .quad {prof_names[i]}, 0, 0")
        out.append("")
        out.append("// tensors (zero-initialized, statically allocated)")
        self.arena = plan(self.prog, set(self.used)) if self.share else Plan({}, 0, 0)
        total = self.arena.size
        shared = []
        for b in sorted(self.used.values(), key=lambda b: b.id):
            if b.kind == "const":
                continue
            if getattr(b, "heap", False):
                out.append(f".zerofill __DATA,__bss,{self.sym(b)},8,3   // {b.kind} {b.name}: {b.dtype}{list(b.shape)}, "
                           f"allocated when the program starts ({b.nbytes:,} bytes)")
                continue
            if b.kind in ir.EXTERN_KINDS:
                out.append(f"    .globl {self.sym(b)}")
                out.append(f".zerofill __DATA,__bss,{self.sym(b)},8,3   // {b.kind} {b.name}: {b.dtype}{list(b.shape)}, "
                           f"a pointer to the caller's array")
                continue
            if b.id in self.arena.offsets:
                shared.append(f"{self.sym(b)} = _anvil_arena + {self.arena.offsets[b.id]}   // {b.kind} {b.name}: "
                              f"{b.dtype}{list(b.shape)}")
                continue
            size = max(16, (b.nbytes + 15) // 16 * 16)
            total += size
            out.append(f".zerofill __DATA,__bss,{self.sym(b)},{size},4   // {b.kind} {b.name}: "
                       f"{b.dtype}{list(b.shape)}")
        if shared:
            out.append("")
            out.append(f"// {len(shared)} temporaries share one arena of {self.arena.size} bytes "
                       f"({self.arena.unshared} bytes unshared): each lives at an offset that no other")
            out.append("// temporary alive at the same time uses")
            out.append(f".zerofill __DATA,__bss,_anvil_arena,{self.arena.size},6")
            out.extend(shared)
        if self.library:
            out.append(".zerofill __DATA,__bss,_anvil_lib_ready,4,2")
        self.bss_bytes = total
        return "\n".join(out)


def generate(prog: ir.Program, seed: int = 0, profile: bool = False,
             threads: bool = True, share: bool = True, library: bool = False,
             check: str | None = None) -> tuple[str, ProgramGen]:
    g = ProgramGen(prog, seed=seed, profile=profile, threads=threads, share=share, library=library, check=check)
    text = g.generate()
    return text, g
