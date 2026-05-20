"""Memory reuse: temporaries whose lifetimes do not overlap share the bytes of one static arena.

The program is flattened into a sequence of statement positions. A buffer's lifetime runs from its
first use to its last, and a lifetime that reaches into a loop from outside it (or out of it)
covers the whole loop, since every iteration comes back to the same statements. Only temporaries
whose first use writes all of them take part, so nothing relies on them starting at zero: the
first write must also not sit inside an `if` or a loop that some later use of the buffer is outside
of (it might not have run). Every other buffer keeps its own zero-initialized memory."""
from __future__ import annotations

from dataclasses import dataclass

from .. import ir
from ..ir import Buffer, KernelStmt

ALIGN = 64
SHARED_KINDS = ("temp", "grad")
FULL_WRITE_CALLS = ("nonzero", "input", "clock")       # runtime calls that write all of args["buf"]


@dataclass
class Use:
    pos: int
    chain: tuple          # the loops and if-branches around the statement, outermost first
    write: bool
    full: bool            # writes every element


@dataclass
class Plan:
    offsets: dict         # buffer id -> byte offset in the arena
    size: int             # arena bytes
    unshared: int         # what the same buffers would take without sharing


def flatten(prog: ir.Program):
    from ..optimize import identity_of, stmt_reads, stmt_writes
    uses: dict[Buffer, list[Use]] = {}
    loops: list[tuple[int, int]] = []
    pos = 0

    def note(b, chain, write=False, full=False):
        if isinstance(b, Buffer):
            uses.setdefault(b.root, []).append(Use(pos, chain, write, full))

    def full_writes(st) -> set:
        out = set()
        if isinstance(st, KernelStmt):
            k = st.kernel
            for s in k.stores:
                b = s.buf.root
                if not s.accumulate and [v.extent for v in k.domain] == list(b.shape) \
                        and s.offset == identity_of(b, k.domain):
                    out.add(b)
        elif isinstance(st, ir.RTCall) and st.name in FULL_WRITE_CALLS:
            out.add(st.args["buf"].root)
        return out

    def block(b: ir.Block, chain):
        for st in b.stmts:
            stmt(st, chain)

    def stmt(st, chain):
        nonlocal pos
        pos += 1
        if isinstance(st, ir.For):
            start = pos
            for x in (st.start, st.stop):
                note(x, chain)
            note(st.counter, chain, write=True)
            block(st.body, chain + (("loop", id(st)),))
            pos += 1
            loops.append((start, pos))
        elif isinstance(st, ir.While):
            start = pos
            inner = chain + (("loop", id(st)),)
            block(st.cond_block, inner)
            pos += 1
            note(st.cond, inner)
            block(st.body, inner)
            pos += 1
            loops.append((start, pos))
        elif isinstance(st, ir.If):
            note(st.cond, chain)
            block(st.then, chain + (("then", id(st)),))
            block(st.orelse, chain + (("else", id(st)),))
            pos += 1
        elif isinstance(st, (ir.Break, ir.Continue)):
            pass
        else:
            fulls = full_writes(st)
            for b in stmt_reads(st):
                note(b, chain)
            for b in stmt_writes(st):
                if isinstance(b, Buffer):
                    note(b, chain, write=True, full=b in fulls)
    block(prog.main, ())
    return uses, loops


def lifetime(us: list[Use], loops) -> tuple[int, int] | None:
    """[first, last] statement position, or None if the buffer must keep its own memory."""
    first = us[0].pos
    at_first = [u for u in us if u.pos == first]
    if any(not u.write for u in at_first) or not any(u.full for u in at_first):
        return None                                   # it is read before it is all written
    chain = at_first[0].chain
    if any(u.chain[:len(chain)] != chain for u in us):
        return None                                   # the first write might not have run
    lo, hi = first, us[-1].pos
    changed = True
    while changed:
        changed = False
        for (a, b) in loops:
            if lo <= b and a <= hi and not (a <= lo and hi <= b) and (a < lo or b > hi):
                lo, hi = min(lo, a), max(hi, b)
                changed = True
    return lo, hi


def plan(prog: ir.Program, used_ids) -> Plan:
    uses, loops = flatten(prog)
    items = []
    for b, us in uses.items():
        if b.id not in used_ids or b.kind not in SHARED_KINDS or getattr(b, "heap", False):
            continue
        span = lifetime(us, loops)
        if span is not None:
            items.append((b, span, max(ALIGN, (b.nbytes + ALIGN - 1) // ALIGN * ALIGN)))
    items.sort(key=lambda it: (-it[2], it[1][0], it[0].id))       # ids break ties: the same layout every run
    placed: list[tuple[int, int, int, int]] = []        # (lo, hi, offset, size)
    offsets: dict[int, int] = {}
    size = 0
    for b, (lo, hi), n in items:
        busy = sorted((off, off + sz) for (l2, h2, off, sz) in placed if l2 <= hi and lo <= h2)
        off = 0
        for (s, e) in busy:
            if off + n <= s:
                break
            off = max(off, e)
        placed.append((lo, hi, off, n))
        offsets[b.id] = off
        size = max(size, off + n)
    return Plan(offsets, size, sum(n for _, _, n in items))
