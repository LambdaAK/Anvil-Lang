"""Machine IR for one kernel: instructions over virtual registers, plus a linear-scan
register allocator with loop-extended live ranges."""
from __future__ import annotations

import re

# allocatable physical registers. x16/x17 are scratch for address arithmetic, x18 is
# reserved by the platform, x29/x30 are the frame pointer and link register.
GPR_POOL = list(range(0, 16)) + list(range(19, 29))
VEC_POOL = list(range(0, 32))


class RegAllocFail(Exception):
    def __init__(self, msg, victim=None):
        super().__init__(msg)
        self.victim = victim


class VReg:
    __slots__ = ("id", "cls", "name")
    _n = 0

    def __init__(self, cls: str, name: str = ""):
        VReg._n += 1
        self.id = VReg._n
        self.cls = cls          # 'x' (general purpose) or 'v' (SIMD)
        self.name = name

    def __repr__(self):
        return f"%{self.cls}{self.id}{('.' + self.name) if self.name else ''}"


class Ins:
    """One machine instruction. `fmt` uses {0}, {1:w}, {2:q}, ... for register operands
    (spec: x w for GPRs; v q s d for SIMD)."""
    __slots__ = ("fmt", "regs", "defs", "uses", "comment")

    def __init__(self, fmt: str, regs=(), defs=(), uses=(), comment: str = ""):
        self.fmt = fmt
        self.regs = list(regs)
        self.defs = list(defs)
        self.uses = list(uses)
        self.comment = comment


class Label:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class LoopMark:
    __slots__ = ("kind", "id")

    def __init__(self, kind, id_):
        self.kind = kind
        self.id = id_


class Raw:
    """Verbatim assembly text (comments, directives)."""
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class Seq:
    """A growable instruction list that may contain nested sequences (spliced at the end)."""
    __slots__ = ("items",)

    def __init__(self):
        self.items = []

    def add(self, x):
        self.items.append(x)
        return x

    def flatten(self, out=None):
        out = [] if out is None else out
        for it in self.items:
            if isinstance(it, Seq):
                it.flatten(out)
            else:
                out.append(it)
        return out


# ----------------------------------------------------------------------------- allocation

def allocate(items: list, gpr_pool=GPR_POOL, vec_pool=VEC_POOL) -> dict:
    """Linear scan. Returns VReg -> physical register number."""
    start: dict[VReg, int] = {}
    end: dict[VReg, int] = {}
    loops: dict[int, list] = {}
    for i, it in enumerate(items):
        if isinstance(it, Ins):
            for r in it.regs:
                if isinstance(r, VReg):
                    if r not in start:
                        start[r] = i
                    end[r] = i
        elif isinstance(it, LoopMark):
            loops.setdefault(it.id, [None, None])
            loops[it.id][0 if it.kind == "begin" else 1] = i
    # a value live into a loop (defined before it, used inside) stays live to the loop end
    spans = [tuple(v) for v in loops.values() if v[0] is not None and v[1] is not None]
    changed = True
    while changed:
        changed = False
        for (b, e) in spans:
            for r in start:
                if start[r] < b <= end[r] < e:
                    end[r] = e
                    changed = True
    order = sorted(start, key=lambda r: (start[r], r.id))
    free = {"x": list(gpr_pool), "v": list(vec_pool)}
    active: list[VReg] = []
    phys: dict[VReg, int] = {}
    for r in order:
        s = start[r]
        still = []
        for a in active:
            if end[a] < s:
                free[a.cls].append(phys[a])
            else:
                still.append(a)
        active = still
        pool = free[r.cls]
        if not pool:
            # spill candidate: the live value of this class that stays live the longest
            cands = [a for a in active + [r] if a.cls == r.cls and a.name != "spill"]
            victim = max(cands, key=lambda a: (end[a] - start[a], end[a])) if cands else None
            raise RegAllocFail(f"out of {r.cls} registers", victim)
        # prefer low registers for readability
        pool.sort()
        phys[r] = pool.pop(0)
        active.append(r)
    return phys


_spec_re = re.compile(r"\{(\d+)(?::(\w))?\}")


def render(items: list, phys: dict) -> list[str]:
    out = []
    for it in items:
        if isinstance(it, Label):
            out.append(f"{it.name}:")
        elif isinstance(it, Raw):
            out.append(it.text)
        elif isinstance(it, LoopMark):
            continue
        else:
            def sub(m):
                r = it.regs[int(m.group(1))]
                spec = m.group(2)
                if isinstance(r, VReg):
                    n = phys[r]
                    if r.cls == "x":
                        return f"{'w' if spec == 'w' else 'x'}{n}"
                    return f"{spec or 'v'}{n}"
                return str(r)
            line = "    " + _spec_re.sub(sub, it.fmt)
            if it.comment:
                line = f"{line:<44}// {it.comment}"
            out.append(line)
    return out


# ----------------------------------------------------------------------------- spilling

_NO_DEF = {"str", "stp", "stur", "st1", "cmp", "cmn", "tst", "fcmp", "fcmpe", "cbz", "cbnz", "b", "bl", "ret"}
_RMW = {"fmla", "fmls", "bsl", "bit", "bif", "movk"}


def roles(ins: Ins):
    """(def positions, use positions) of an instruction's register operands."""
    mnem = ins.fmt.split()[0]
    n = len(ins.regs)
    if mnem in _NO_DEF or mnem.startswith("b."):
        return set(), set(range(n))
    rmw = mnem in _RMW or "{0}.s[" in ins.fmt or "{0}.d[" in ins.fmt or mnem in ("ld1",) and "}[" in ins.fmt
    defs = {0}
    uses = set(range(1, n)) | ({0} if rmw else set())
    # a register that appears both as destination and source
    for i in range(1, n):
        if ins.regs[i] is ins.regs[0]:
            uses.add(0)
    return defs, uses


def spill(items: list, victim: VReg, offset: int) -> list:
    """Rewrite `victim` to live in the stack slot at [sp, #offset]."""
    out = []
    for it in items:
        if not isinstance(it, Ins) or not any(r is victim for r in it.regs):
            out.append(it)
            continue
        defs, uses = roles(it)
        pos = [i for i, r in enumerate(it.regs) if r is victim]
        t = VReg(victim.cls, "spill")
        is_use = any(p in uses for p in pos)
        is_def = any(p in defs for p in pos)
        ld = "ldr {0:q}" if victim.cls == "v" else "ldr {0}"
        st = "str {0:q}" if victim.cls == "v" else "str {0}"
        if is_use:
            out.append(Ins(f"{ld}, [sp, #{offset}]", [t], comment="reload"))
        out.append(Ins(it.fmt, [t if r is victim else r for r in it.regs], comment=it.comment))
        if is_def:
            out.append(Ins(f"{st}, [sp, #{offset}]", [t], comment="spill"))
    return out


def allocate_with_spills(items: list, max_spills: int = 256):
    """Allocate, spilling as needed. Returns (items, phys, frame_bytes)."""
    slots = 0
    while True:
        try:
            phys = allocate(items)
            return items, phys, 16 * slots
        except RegAllocFail as e:
            if e.victim is None or slots >= max_spills:
                raise
            items = spill(items, e.victim, 16 * slots)
            slots += 1
