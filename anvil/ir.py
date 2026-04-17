"""Kernel IR.

A program is a tree of blocks. A block is a list of statements: kernels (tensor
comprehensions) and structured control flow. A kernel is

    for d in domain:                     # output loops
        acc = ⊕_{r in red.vars} red.body  # optional reduction
        let ... ; store buf[offset] (=|+=) value ...

Every memory access is `buffer + affine offset` (in elements). An affine offset is
c0 + Σ ci·vari + Σ dj·scalarj + Σ ek·gatherk, so views (slices, transposes, reshapes,
broadcasts, gathers) never copy.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

F32 = "f32"
I32 = "i32"

# Buffers that belong to the caller of a compiled function (anvil.function): the program reaches
# them through a pointer that the caller sets before each call.
EXTERN_KINDS = ("input", "output")


def indirect(b) -> bool:
    """Reached through a pointer: the caller's arrays, and large tensors allocated when the
    program starts (a multi-gigabyte static segment would collide with the system's shared
    libraries)."""
    r = b.root
    return r.kind in EXTERN_KINDS or getattr(r, "heap", False)

_ids = itertools.count(1)


def fresh_id() -> int:
    return next(_ids)


def prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def row_major_strides(shape) -> list[int]:
    strides = []
    s = 1
    for d in reversed(shape):
        strides.append(s)
        s *= d
    return list(reversed(strides))


# ----------------------------------------------------------------------------- buffers & vars

class Buffer:
    """A statically allocated array. kind: param | state | data | temp | var | const | grad | scalar,
    or input | output (the caller's arrays, see EXTERN_KINDS)."""

    def __init__(self, name: str, shape, dtype: str = F32, kind: str = "temp", init=None, span=None):
        self.id = fresh_id()
        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.kind = kind
        self.init = init          # list of python numbers for const buffers
        self.span = span
        self.storage = self       # buffers may alias another buffer's storage (in-place reuse)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return 4 * self.numel

    @property
    def strides(self) -> list[int]:
        return row_major_strides(self.shape)

    @property
    def root(self) -> "Buffer":
        b = self
        while b.storage is not b:
            b = b.storage
        return b

    def __repr__(self) -> str:
        return f"{self.name}#{self.id}"


class Var:
    """A loop variable with a static extent (extent may be filled in after range inference)."""

    def __init__(self, name: str, extent: int | None = None):
        self.id = fresh_id()
        self.name = name
        self.extent = extent

    def __repr__(self) -> str:
        return self.name


# ----------------------------------------------------------------------------- affine offsets

class ScalarRef:
    """The runtime value of an i32 0-d buffer, used in an index."""

    def __init__(self, buf: Buffer):
        self.buf = buf

    def key(self):
        return ("s", self.buf.id)

    def __eq__(self, o):
        return isinstance(o, ScalarRef) and o.buf is self.buf

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return self.buf.name


class Gather:
    """A data-dependent index: the i32 value loaded from load.buf at load.offset.
    `bound` is the extent of the indexed dimension (checked at run time)."""

    def __init__(self, load: "Load", bound: int, where: str = "", rest: "Affine | None" = None):
        self.load = load
        self.bound = bound
        self.where = where
        self.rest = rest          # `ids[i] + t`: the index is the gathered value plus rest (checked together)

    def key(self):
        return ("g", self.load.key(), self.bound, self.rest.key() if self.rest is not None else None)

    def check_range(self) -> tuple[int, int]:
        """The gathered value v must satisfy lo <= v < hi (v + rest stays inside the dimension)."""
        if self.rest is None:
            return 0, self.bound
        lo, hi = self.rest.var_bounds()
        return -lo, self.bound - hi

    def __eq__(self, o):
        return isinstance(o, Gather) and o.key() == self.key()

    def __hash__(self):
        return hash(self.key())

    def subst(self, m) -> "Gather":
        return Gather(Load(self.load.buf, self.load.offset.subst(m)), self.bound, self.where,
                      self.rest.subst(m) if self.rest is not None else None)

    def __repr__(self):
        return f"{self.load!r}"


def _term_key(t):
    if isinstance(t, Var):
        return ("v", t.id)
    return t.key()


class Affine:
    """Immutable affine integer expression."""

    __slots__ = ("const", "terms")

    def __init__(self, const: int = 0, terms: dict | None = None):
        self.const = int(const)
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    @staticmethod
    def of(x) -> "Affine":
        if isinstance(x, Affine):
            return x
        if isinstance(x, int):
            return Affine(x)
        return Affine(0, {x: 1})

    def __add__(self, o) -> "Affine":
        o = Affine.of(o)
        t = dict(self.terms)
        for k, v in o.terms.items():
            t[k] = t.get(k, 0) + v
        return Affine(self.const + o.const, t)

    __radd__ = __add__

    def __neg__(self) -> "Affine":
        return Affine(-self.const, {k: -v for k, v in self.terms.items()})

    def __sub__(self, o) -> "Affine":
        return self + (-Affine.of(o))

    def __rsub__(self, o) -> "Affine":
        return Affine.of(o) - self

    def __mul__(self, c: int) -> "Affine":
        assert isinstance(c, int)
        return Affine(self.const * c, {k: v * c for k, v in self.terms.items()})

    __rmul__ = __mul__

    def is_const(self) -> bool:
        return not self.terms

    def vars(self) -> list[Var]:
        out = []
        for k in self.terms:
            if isinstance(k, Var):
                out.append(k)
            elif isinstance(k, Gather):
                for v in k.load.offset.vars():
                    if v not in out:
                        out.append(v)
        return out

    def direct_vars(self) -> list[Var]:
        return [k for k in self.terms if isinstance(k, Var)]

    def coef(self, v) -> int:
        return self.terms.get(v, 0)

    def scalars(self) -> list[ScalarRef]:
        return [k for k in self.terms if isinstance(k, ScalarRef)]

    def gathers(self) -> list[Gather]:
        return [k for k in self.terms if isinstance(k, Gather)]

    def single_var(self):
        """If this is exactly `1*v`, return v."""
        if self.const == 0 and len(self.terms) == 1:
            (k, c), = self.terms.items()
            if isinstance(k, Var) and c == 1:
                return k
        return None

    def subst(self, m: dict) -> "Affine":
        """Substitute vars -> Affine (and recurse into gathers)."""
        out = Affine(self.const)
        for k, c in self.terms.items():
            if isinstance(k, Var) and k in m:
                out = out + Affine.of(m[k]) * c
            elif isinstance(k, Gather):
                out = out + Affine(0, {k.subst(m): c})
            else:
                out = out + Affine(0, {k: c})
        return out

    def key(self):
        return (self.const, tuple(sorted(((_term_key(k), c) for k, c in self.terms.items()), key=repr)))

    def __eq__(self, o):
        return isinstance(o, Affine) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def bounds(self) -> tuple[int, int] | None:
        """Min/max over var extents; None if it depends on runtime values."""
        lo = hi = self.const
        for k, c in self.terms.items():
            if not isinstance(k, Var) or k.extent is None:
                return None
            span = c * (k.extent - 1)
            if span >= 0:
                hi += span
            else:
                lo += span
        return lo, hi

    def var_bounds(self) -> tuple[int, int]:
        """Min/max of the var part only (ignores scalars/gathers)."""
        lo = hi = self.const
        for k, c in self.terms.items():
            if isinstance(k, Var):
                span = c * ((k.extent or 1) - 1)
                if span >= 0:
                    hi += span
                else:
                    lo += span
        return lo, hi

    def __repr__(self) -> str:
        return fmt_affine(self)


def fmt_affine(a: Affine) -> str:
    parts = []
    for k, c in sorted(a.terms.items(), key=lambda kv: -abs(kv[1])):
        name = k.name if isinstance(k, Var) else repr(k)
        if c == 1:
            parts.append(f"+ {name}")
        elif c == -1:
            parts.append(f"- {name}")
        elif c < 0:
            parts.append(f"- {-c}*{name}")
        else:
            parts.append(f"+ {c}*{name}")
    if a.const or not parts:
        parts.append(f"+ {a.const}" if a.const >= 0 else f"- {-a.const}")
    s = " ".join(parts)
    if s.startswith("+ "):
        s = s[2:]
    elif s.startswith("- "):
        s = "-" + s[2:]
    return s


def decompose(offset: Affine, shape) -> list[Affine] | None:
    """Best-effort split of a flat offset into per-dimension indices (for display)."""
    if not shape:
        return []
    strides = row_major_strides(shape)
    dims = [Affine(0) for _ in shape]
    const = offset.const
    for d, s in enumerate(strides):
        q, const = divmod(const, s) if s else (0, const)
        dims[d] = dims[d] + q
    for k, c in offset.terms.items():
        for d, s in enumerate(strides):
            if s and c % s == 0:
                dims[d] = dims[d] + Affine(0, {k: c // s})
                break
        else:
            return None
    return dims


# ----------------------------------------------------------------------------- scalar expressions

UNARY_OPS = {
    "neg", "abs", "exp", "log", "sqrt", "tanh", "sigmoid", "sin", "cos", "floor", "ceil",
    "round", "sign", "not", "f32", "i32", "detach", "rsqrt", "recip",
}
BINARY_OPS = {
    "add", "sub", "mul", "div", "max", "min", "pow", "idiv", "mod",
    "lt", "le", "gt", "ge", "eq", "ne", "and", "or",
}
COMPARISONS = {"lt", "le", "gt", "ge", "eq", "ne"}
SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "idiv": "//", "mod": "%", "pow": "**",
    "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "and", "or": "or",
}
PREC = {"or": 1, "and": 2, "lt": 3, "le": 3, "gt": 3, "ge": 3, "eq": 3, "ne": 3,
        "add": 4, "sub": 4, "mul": 5, "div": 5, "idiv": 5, "mod": 5, "pow": 7}


class Expr:
    dtype: str = F32

    def key(self):
        raise NotImplementedError

    def children(self) -> list["Expr"]:
        return []

    def rebuild(self, kids: list["Expr"]) -> "Expr":
        return self

    def __eq__(self, o):
        return isinstance(o, Expr) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return fmt_expr(self)


class Const(Expr):
    def __init__(self, value, dtype=F32):
        self.dtype = dtype
        self.value = float(value) if dtype == F32 else int(value)

    def key(self):
        return ("c", self.dtype, self.value)


class Load(Expr):
    def __init__(self, buf: Buffer, offset: Affine):
        self.buf = buf
        self.offset = offset
        self.dtype = buf.dtype

    def key(self):
        return ("ld", self.buf.id, self.offset.key())


class Index(Expr):
    """Integer value of an affine expression of loop vars (e.g. `i`, `i + 1`)."""
    dtype = I32

    def __init__(self, affine: Affine):
        self.affine = affine

    def key(self):
        return ("ix", self.affine.key())


class Unary(Expr):
    def __init__(self, op: str, a: Expr, dtype: str | None = None):
        assert op in UNARY_OPS, op
        self.op = op
        self.a = a
        if dtype is None:
            dtype = {"f32": F32, "i32": I32, "not": F32}.get(op, a.dtype)
            if op in ("exp", "log", "sqrt", "tanh", "sigmoid", "sin", "cos", "rsqrt", "recip"):
                dtype = F32
        self.dtype = dtype

    def key(self):
        return ("u", self.op, self.dtype, self.a.key())

    def children(self):
        return [self.a]

    def rebuild(self, kids):
        return Unary(self.op, kids[0], self.dtype)


class Binary(Expr):
    def __init__(self, op: str, a: Expr, b: Expr, dtype: str | None = None):
        assert op in BINARY_OPS, op
        self.op = op
        self.a = a
        self.b = b
        if dtype is None:
            if op in COMPARISONS or op in ("and", "or"):
                dtype = F32
            elif op in ("idiv", "mod"):
                dtype = I32 if (a.dtype == I32 and b.dtype == I32) else F32
            elif op == "div":
                dtype = F32
            else:
                dtype = I32 if (a.dtype == I32 and b.dtype == I32) else F32
        self.dtype = dtype

    def key(self):
        return ("b", self.op, self.dtype, self.a.key(), self.b.key())

    def children(self):
        return [self.a, self.b]

    def rebuild(self, kids):
        return Binary(self.op, kids[0], kids[1], self.dtype)


class Select(Expr):
    def __init__(self, c: Expr, a: Expr, b: Expr):
        self.c = c
        self.a = a
        self.b = b
        self.dtype = F32 if F32 in (a.dtype, b.dtype) else I32

    def key(self):
        return ("sel", self.c.key(), self.a.key(), self.b.key())

    def children(self):
        return [self.c, self.a, self.b]

    def rebuild(self, kids):
        return Select(*kids)


class Acc(Expr):
    """The result of the kernel's reduction (only valid in kernel stmts)."""

    def __init__(self, dtype=F32):
        self.dtype = dtype

    def key(self):
        return ("acc", self.dtype)


class LetRef(Expr):
    def __init__(self, let: "Let"):
        self.let = let
        self.dtype = let.value.dtype

    def key(self):
        return ("let", id(self.let))


class Rand(Expr):
    """Uniform f32 in [0, 1) (or (0, 1] if open_low) from a counter-based hash of
    (seed, stream, element index, salt)."""
    dtype = F32

    def __init__(self, salt: int, open_low: bool = False):
        self.salt = salt
        self.open_low = open_low

    def key(self):
        return ("rand", self.salt, self.open_low)


# ---- traversal helpers

def map_expr(e: Expr, fn) -> Expr:
    """Bottom-up rebuild: fn(node_with_new_children) -> node."""
    kids = e.children()
    if kids:
        new = [map_expr(k, fn) for k in kids]
        if any(n is not k for n, k in zip(new, kids)):
            e = e.rebuild(new)
    return fn(e)


def iter_expr(e: Expr):
    yield e
    for k in e.children():
        yield from iter_expr(k)
    if isinstance(e, Load):
        for g in e.offset.gathers():
            yield from iter_expr(g.load)


def loads_in(e: Expr) -> list[Load]:
    return [x for x in iter_expr(e) if isinstance(x, Load)]


def subst_expr(e: Expr, m: dict) -> Expr:
    """Substitute loop vars in all offsets / index values."""
    if not m:
        return e

    def f(x):
        if isinstance(x, Load):
            return Load(x.buf, x.offset.subst(m))
        if isinstance(x, Index):
            return Index(x.affine.subst(m))
        return x
    return map_expr(e, f)


def expr_vars(e: Expr) -> set[Var]:
    out: set[Var] = set()
    for x in iter_expr(e):
        if isinstance(x, Load):
            out.update(x.offset.vars())
        elif isinstance(x, Index):
            out.update(x.affine.vars())
    return out


def uses_rand(e: Expr) -> bool:
    return any(isinstance(x, Rand) for x in iter_expr(e))


# ----------------------------------------------------------------------------- kernels

@dataclass
class Let:
    name: str
    value: Expr

    def __hash__(self):
        return id(self)

    def __eq__(self, o):
        return self is o


@dataclass
class Store:
    buf: Buffer
    offset: Affine
    value: Expr
    accumulate: bool = False


@dataclass
class Reduction:
    vars: list[Var]
    op: str           # sum | max | min | prod | argmax | argmin
    body: Expr

    @property
    def dtype(self):
        return I32 if self.op in ("argmax", "argmin") else self.body.dtype


@dataclass
class Kernel:
    domain: list[Var]
    stmts: list           # list[Let | Store]
    red: Reduction | None = None
    name: str = ""
    label: str = ""
    span: object = None
    id: int = field(default_factory=fresh_id)

    @property
    def stores(self) -> list[Store]:
        return [s for s in self.stmts if isinstance(s, Store)]

    def exprs(self):
        if self.red is not None:
            yield self.red.body
        for s in self.stmts:
            yield s.value

    def reads(self) -> set[Buffer]:
        out = set()
        for e in self.exprs():
            for ld in loads_in(e):
                out.add(ld.buf.root)
        for s in self.stores:
            if s.accumulate:
                out.add(s.buf.root)
            for g in s.offset.gathers():
                out.add(g.load.buf.root)
        return out

    def writes(self) -> set[Buffer]:
        return {s.buf.root for s in self.stores}

    def all_vars(self) -> list[Var]:
        return list(self.domain) + (list(self.red.vars) if self.red else [])

    def iterations(self) -> int:
        return prod(v.extent for v in self.all_vars())

    def uses_rand(self) -> bool:
        return any(uses_rand(e) for e in self.exprs())


# ----------------------------------------------------------------------------- program statements

class Stmt:
    pass


@dataclass
class KernelStmt(Stmt):
    kernel: Kernel


@dataclass
class Block:
    stmts: list = field(default_factory=list)
    is_loop: bool = False


@dataclass
class For(Stmt):
    counter: Buffer          # i32 0-d
    start: object            # int | Buffer (i32 0-d)
    stop: object
    step: int
    body: Block
    span: object = None


@dataclass
class While(Stmt):
    cond_block: Block        # computes cond
    cond: Buffer
    body: Block
    span: object = None


@dataclass
class If(Stmt):
    cond: Buffer             # 0-d, f32 or i32; nonzero = true
    then: Block
    orelse: Block
    span: object = None


@dataclass
class Break(Stmt):
    pass


@dataclass
class Continue(Stmt):
    pass


@dataclass
class PrintItem:
    kind: str                # text | scalar | tensor | pick
    text: str = ""
    buf: Buffer | None = None
    fmt: str = ""            # printf conversion for scalars
    choices: list | None = None     # pick: the strings that buf (an i32 scalar) chooses from


@dataclass
class Print(Stmt):
    items: list
    end: str = "\n"
    err: bool = False        # to stderr (scalars and strings only)


@dataclass
class Check(Stmt):
    """Runtime assertion lo <= value <= hi where value is an affine of runtime scalars."""
    value: Affine
    lo: int
    hi: int
    message: str
    span: object = None


@dataclass
class RTCall(Stmt):
    """Runtime routine: load_idx, shuffle, iota, seed, clock, show, sleep, nonzero, input, save, load."""
    name: str
    args: dict
    span: object = None


@dataclass
class Program:
    main: Block
    buffers: list
    source: object = None


def walk_blocks(block: Block):
    """Yield every block (pre-order)."""
    yield block
    for s in block.stmts:
        if isinstance(s, For):
            yield from walk_blocks(s.body)
        elif isinstance(s, While):
            yield from walk_blocks(s.cond_block)
            yield from walk_blocks(s.body)
        elif isinstance(s, If):
            yield from walk_blocks(s.then)
            yield from walk_blocks(s.orelse)


def all_kernels(prog: Program) -> list[Kernel]:
    out = []
    for b in walk_blocks(prog.main):
        for s in b.stmts:
            if isinstance(s, KernelStmt):
                out.append(s.kernel)
    return out


# ----------------------------------------------------------------------------- pretty printing

FN_NAMES = {"neg": "-", "f32": "f32", "i32": "i32"}


def fmt_load(ld: Load) -> str:
    dims = decompose(ld.offset, ld.buf.shape)
    if dims is None:
        return f"{ld.buf.name}{{{fmt_affine(ld.offset)}}}"
    if not dims:
        return ld.buf.name
    return f"{ld.buf.name}[{', '.join(fmt_affine(d) for d in dims)}]"


def fmt_expr(e: Expr, parent_prec: int = 0) -> str:
    if isinstance(e, Const):
        if e.dtype == F32:
            v = e.value
            if math.isinf(v):
                return "∞" if v > 0 else "-∞"
            s = f"{v:.6g}"
            return s if ("." in s or "e" in s or "n" in s) else s + ".0"
        return str(e.value)
    if isinstance(e, Load):
        return fmt_load(e)
    if isinstance(e, Index):
        return fmt_affine(e.affine)
    if isinstance(e, Acc):
        return "acc"
    if isinstance(e, LetRef):
        return e.let.name
    if isinstance(e, Rand):
        return f"rand{'⁺' if e.open_low else ''}{e.salt}"
    if isinstance(e, Unary):
        if e.op == "neg":
            s = "-" + fmt_expr(e.a, 6)
            return f"({s})" if parent_prec > 6 else s
        if e.op == "not":
            return f"not {fmt_expr(e.a, 3)}"
        return f"{e.op}({fmt_expr(e.a)})"
    if isinstance(e, Binary):
        if e.op in ("max", "min"):
            return f"{e.op}({fmt_expr(e.a)}, {fmt_expr(e.b)})"
        p = PREC[e.op]
        right_p = p + 1 if e.op != "pow" else p
        s = f"{fmt_expr(e.a, p if e.op != 'pow' else p + 1)} {SYMBOL[e.op]} {fmt_expr(e.b, right_p)}"
        return f"({s})" if p < parent_prec else s
    if isinstance(e, Select):
        s = f"{fmt_expr(e.a, 1)} if {fmt_expr(e.c, 1)} else {fmt_expr(e.b, 1)}"
        return f"({s})" if parent_prec > 0 else s
    return f"<{type(e).__name__}>"


RED_SYMBOL = {"sum": "Σ", "max": "max", "min": "min", "prod": "Π", "argmax": "argmax", "argmin": "argmin"}


def fmt_kernel(k: Kernel) -> str:
    # lets that reuse a name (an accumulation chain) are shown as x, x′, x″, ...
    seen: dict[str, int] = {}
    shown: dict[int, str] = {}
    for s in k.stmts:
        if isinstance(s, Let):
            n = seen.get(s.name, 0)
            seen[s.name] = n + 1
            shown[id(s)] = s.name + ("′" * n if n < 3 else f"_{n}")
    renamed = {}
    for s in k.stmts:
        if isinstance(s, Let) and shown[id(s)] != s.name:
            renamed[id(s)] = shown[id(s)]
    if renamed:
        k = Kernel(domain=k.domain, red=k.red, name=k.name, label=k.label, span=k.span,
                   stmts=[_display_let(s, shown) for s in k.stmts])
    lines = []
    dom = ", ".join(v.name for v in k.domain)
    red = ""
    if k.red is not None:
        rv = ",".join(v.name for v in k.red.vars)
        red = f"{RED_SYMBOL[k.red.op]}_{rv} {fmt_expr(k.red.body, 5)}"
    for s in k.stmts:
        if isinstance(s, Let):
            lines.append(f"let {s.name} = {fmt_expr(s.value).replace('acc', red) if red else fmt_expr(s.value)}")
        else:
            val = fmt_expr(s.value)
            if red:
                val = val.replace("acc", red)
            dims = decompose(s.offset, s.buf.shape)
            if dims is None:
                tgt = f"{s.buf.name}{{{fmt_affine(s.offset)}}}"
            elif not dims:
                tgt = s.buf.name
            else:
                tgt = f"{s.buf.name}[{', '.join(fmt_affine(d) for d in dims)}]"
            lines.append(f"{tgt} {'+=' if s.accumulate else '='} {val}")
    return "; ".join(lines) if len(lines) > 1 else (lines[0] if lines else f"<empty kernel over {dom}>")


def fmt_program(prog: Program, show_buffers: bool = False) -> str:
    """Human-readable dump of a program's blocks (used by `anvil ir`)."""
    out: list[str] = []

    def bound(x):
        return str(x) if isinstance(x, int) else x.name

    def blk(b: Block, ind: str):
        for s in b.stmts:
            if isinstance(s, KernelStmt):
                k = s.kernel
                tag = f"{k.name}"
                lab = f"  ⟨{k.label}⟩" if k.label else ""
                out.append(f"{ind}{tag:<6} {fmt_kernel(k)}{lab}")
            elif isinstance(s, For):
                out.append(f"{ind}for {s.counter.name} in {bound(s.start)}..{bound(s.stop)}"
                           f"{'' if s.step == 1 else f' step {s.step}'}:")
                blk(s.body, ind + "    ")
            elif isinstance(s, While):
                out.append(f"{ind}while:")
                blk(s.cond_block, ind + "  ? ")
                out.append(f"{ind}  ? test {s.cond.name}")
                blk(s.body, ind + "    ")
            elif isinstance(s, If):
                out.append(f"{ind}if {s.cond.name}:")
                blk(s.then, ind + "    ")
                if s.orelse.stmts:
                    out.append(f"{ind}else:")
                    blk(s.orelse, ind + "    ")
            elif isinstance(s, Print):
                parts = []
                for it in s.items:
                    if it.kind == "text":
                        parts.append(repr(it.text)[1:-1])
                    elif it.kind == "scalar":
                        parts.append(f"{{{it.buf.name}:{it.fmt}}}")
                    elif it.kind == "pick":
                        choices = ", ".join(repr(c)[1:-1] for c in it.choices)
                        parts.append(f"{{[{choices}][{it.buf.name}]}}")
                    else:
                        parts.append(f"{{{it.buf.name}}}")
                out.append(f"{ind}print \"{''.join(parts)}\"")
            elif isinstance(s, Check):
                out.append(f"{ind}check {s.lo} ≤ {fmt_affine(s.value)} ≤ {s.hi}")
            elif isinstance(s, RTCall):
                def arg(v):
                    if isinstance(v, list):
                        return "[" + ", ".join(b.name for b in v) + "]"
                    return v.name if isinstance(v, Buffer) else v
                shown = ("src", "bufs", "buf", "value", "path")
                args = ", ".join(f"{k}={arg(v)}" for k, v in s.args.items() if k in shown)
                out.append(f"{ind}{s.name}({args})")
            elif isinstance(s, Break):
                out.append(f"{ind}break")
            elif isinstance(s, Continue):
                out.append(f"{ind}continue")
    blk(prog.main, "")
    if show_buffers:
        out.append("")
        for b in prog.buffers:
            out.append(f"  {b.kind:<6} {b.name}: {b.dtype}{list(b.shape)}")
    return "\n".join(out)


def _display_let(s, shown: dict):
    """Copy of a statement with let names replaced by their display names."""
    names = {}

    def f(x):
        if isinstance(x, LetRef):
            nl = names.get(id(x.let))
            if nl is None:
                nl = Let(shown.get(id(x.let), x.let.name), x.let.value)
                names[id(x.let)] = nl
            return LetRef(nl)
        return x
    if isinstance(s, Let):
        return Let(shown[id(s)], map_expr(s.value, f))
    return Store(s.buf, s.offset, map_expr(s.value, f), s.accumulate)
