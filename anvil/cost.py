"""`anvil cost file.anvil`: what a program will cost, before it runs.

Shapes and most loop counts are known at compile time, so the compiler can count the arithmetic
and the memory traffic of every kernel, and how many times each runs: the totals for the whole
program, and where they come from, line by line.

FLOPs count each arithmetic operation (a multiply-add is 2, as usual; exp, log and the like count
1). Memory traffic counts each tensor a kernel reads once, and each element it writes once: a
lower bound on what crosses the memory bus. A loop whose length is only known at run time (a
`while`, or `for` with a run-time bound) is counted once and marked with `≥`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ir
from .ir import Binary, Kernel, KernelStmt, Select, Unary

FREE_UNARY = {"detach", "f32", "i32"}


def kernel_flops(k: Kernel) -> int:
    def ops(e) -> int:
        n = 0
        for x in ir.iter_expr(e):
            if isinstance(x, Binary) or isinstance(x, Select):
                n += 1
            elif isinstance(x, Unary) and x.op not in FREE_UNARY:
                n += 1
        return n
    dom = ir.prod(v.extent for v in k.domain)
    per = 0
    if k.red is not None:
        red = ir.prod(v.extent for v in k.red.vars)
        per += red * (ops(k.red.body) + 1)               # + the accumulation
    per += sum(ops(st.value) for st in k.stmts)
    return dom * per


def kernel_bytes(k: Kernel) -> int:
    """Each tensor read counts once, but no more elements than the loads can reach (a gather of a
    batch from a dataset reads the batch, not the dataset)."""
    dom = ir.prod(v.extent for v in k.domain)
    red = ir.prod(v.extent for v in k.red.vars) if k.red is not None else 1
    reach: dict = {}
    for e, n in ([(k.red.body, dom * red)] if k.red is not None else []) + [(st.value, dom) for st in k.stmts]:
        for ld in ir.loads_in(e):
            b = ld.buf.root
            reach[b] = reach.get(b, 0) + n
    read = sum(4 * min(b.numel, n) for b, n in reach.items())
    written = sum(4 * dom for _ in k.stores)
    return read + written


@dataclass
class Line:
    flops: int = 0
    bytes: int = 0
    runs: int = 0
    kernels: set = field(default_factory=set)
    unbounded: bool = False
    file: object = None                     # the SourceFile, for showing the line


@dataclass
class Cost:
    flops: int = 0
    bytes: int = 0
    unbounded: bool = False
    lines: dict = field(default_factory=dict)        # (path, line) -> Line


def program_cost(prog: ir.Program) -> Cost:
    cost = Cost()

    def block(b: ir.Block, times: int, unbounded: bool):
        for st in b.stmts:
            if isinstance(st, KernelStmt):
                k = st.kernel
                f, m = kernel_flops(k) * times, kernel_bytes(k) * times
                cost.flops += f
                cost.bytes += m
                key = (k.span.file.path, k.span.line) if k.span is not None else ("?", 0)
                ln = cost.lines.setdefault(key, Line())
                ln.flops += f
                ln.bytes += m
                ln.runs = max(ln.runs, times)
                ln.kernels.add(k.id)
                ln.unbounded |= unbounded
                ln.file = k.span.file if k.span is not None else None
            elif isinstance(st, ir.For):
                if isinstance(st.start, int) and isinstance(st.stop, int):
                    n = max(0, -(-(st.stop - st.start) // st.step))
                    block(st.body, times * n, unbounded)
                else:
                    cost.unbounded = True
                    block(st.body, times, True)
            elif isinstance(st, ir.While):
                cost.unbounded = True
                block(st.cond_block, times, True)
                block(st.body, times, True)
            elif isinstance(st, ir.If):
                block(st.then, times, unbounded)            # both branches: an upper bound
                block(st.orelse, times, unbounded)
    block(prog.main, 1, False)
    return cost


def human(n: float, unit: str = "") -> str:
    for p, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= p:
            return f"{n / p:.1f} {s}{unit}"
    return f"{n:.0f} {unit}".rstrip()


def report(compiled, path: str, rate: float = 300e9) -> str:
    """The text `anvil cost` prints. rate: FLOP/s for the rough time estimate."""
    from .backend.aarch64 import generate
    from .cli import human_bytes, memory_report
    prog = compiled.program
    _, gen = generate(prog)
    mem = memory_report(prog, set(gen.used), gen.arena)
    params = [b for b in prog.buffers if b.kind == "param" and b.root is b]
    nparams = sum(b.numel for b in params)
    c = program_cost(prog)
    ge = "≥ " if c.unbounded else ""
    total_mem = sum(mem.values())
    lines = [f"{path}: {nparams:,} parameters ({human_bytes(4 * nparams)}), "
             f"memory {human_bytes(total_mem)} (" + ", ".join(f"{k} {human_bytes(v)}" for k, v in
                                                       sorted(mem.items(), key=lambda kv: -kv[1])) + ")",
             f"the whole run: {ge}{human(c.flops, 'FLOP')}, {ge}{human(c.bytes, 'B')} of memory traffic "
             f"(roughly {c.flops / rate:.2g} s at {rate / 1e9:.0f} GFLOP/s)", ""]
    rows = sorted(c.lines.items(), key=lambda kv: -kv[1].flops)
    lines.append("  share      FLOP     traffic        runs   line")
    shown = 0
    for (p, line), ln in rows:
        if shown >= 15 or (ln.flops < c.flops * 0.002 and shown >= 5):
            break
        share = ln.flops / c.flops if c.flops else 0
        text = ln.file.line_text(line).strip() if ln.file is not None else ""
        import os
        where = f"{os.path.basename(p)}:{line}"
        runs = ("≥ " if ln.unbounded else "") + f"{ln.runs:,}"
        lines.append(f"  {share:5.1%}  {human(ln.flops):>9}  {human(ln.bytes, 'B'):>10}  {runs:>10}   "
                     f"{where:<16} {text[:60]}")
        shown += 1
    if len(rows) > shown:
        rest = sum(ln.flops for _, ln in rows[shown:])
        lines.append(f"  {rest / c.flops if c.flops else 0:5.1%}  {human(rest):>9}  (the other {len(rows) - shown} lines)")
    return "\n".join(lines)
