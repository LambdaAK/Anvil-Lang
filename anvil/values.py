"""Compile-time values manipulated by the elaborator."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ir import F32, I32, Affine, Buffer, Expr, Load, Unary, Var, row_major_strides


class Val:
    kind = "value"


@dataclass
class CVal(Val):
    """A compile-time number or bool."""
    value: object

    kind = "constant"

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, (int, bool)) and not isinstance(self.value, float)

    @property
    def dtype(self) -> str:
        return I32 if self.is_int else F32


@dataclass
class SVal(Val):
    value: str
    kind = "string"


@dataclass
class FStrVal(Val):
    parts: list         # str | (Val, spec, span)
    kind = "string"


@dataclass
class PickVal(Val):
    """One of several strings, chosen at run time: `NAMES[k]` with k a run-time integer."""
    choices: list       # list[str]
    index: object       # TVal: an i32 scalar buffer
    kind = "string"


@dataclass
class TextVal(Val):
    """`decode(codes)`: an i32 tensor of byte values, printed as the text they spell."""
    buf: object
    kind = "string"


@dataclass
class PoisonVal(Val):
    """What a name is bound to after the statement defining it failed to compile."""
    kind = "error"


@dataclass
class NoneVal(Val):
    kind = "nothing"


class TVal(Val):
    """A (possibly strided / gathered) view of a buffer.

    Element (p0, ..., pr-1) lives at flat offset `tmpl` with placeholder vars `pvars`
    substituted by the index expressions."""
    kind = "tensor"

    def __init__(self, buf: Buffer, shape, tmpl: Affine, pvars: list[Var], detached: bool = False):
        self.buf = buf
        self.shape = tuple(shape)
        self.tmpl = tmpl
        self.pvars = pvars
        self.detached = detached

    @staticmethod
    def of(buf: Buffer) -> "TVal":
        pvars = [Var(f"p{d}", n) for d, n in enumerate(buf.shape)]
        tmpl = Affine(0, {v: s for v, s in zip(pvars, row_major_strides(buf.shape))})
        return TVal(buf, buf.shape, tmpl, pvars)

    @property
    def dtype(self) -> str:
        return self.buf.dtype

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def offset(self, idx: list[Affine]) -> Affine:
        assert len(idx) == len(self.pvars), (idx, self.pvars)
        return self.tmpl.subst({p: Affine.of(i) for p, i in zip(self.pvars, idx)})

    def load(self, idx: list[Affine]) -> Expr:
        e = Load(self.buf, self.offset(idx))
        if self.detached:
            e = Unary("detach", e)
        return e

    def is_identity(self) -> bool:
        """True if this view is exactly the whole buffer in row-major order."""
        if self.shape != self.buf.shape:
            return False
        want = Affine(0, {v: s for v, s in zip(self.pvars, row_major_strides(self.shape))})
        return self.tmpl == want

    def is_contiguous(self) -> bool:
        """Row-major dense (possibly with a constant/runtime base offset)."""
        strides = row_major_strides(self.shape)
        for v, s in zip(self.pvars, strides):
            if self.shape and self.tmpl.coef(v) != s and not (v.extent == 1):
                return False
        if any(g for g in self.tmpl.gathers()):
            return False
        return True

    def with_detached(self) -> "TVal":
        return TVal(self.buf, self.shape, self.tmpl, self.pvars, True)

    def __repr__(self):
        return f"TVal({self.buf.name}{list(self.shape)})"


@dataclass
class AffVal(Val):
    """A runtime integer that is an affine function of runtime scalars (loop counters):
    kept symbolic so that `x[b*64 : b*64 + 64]` has a provably static length."""
    affine: Affine
    kind = "integer"

    dtype = I32


@dataclass
class IdxVal(Val):
    """An integer index expression inside index notation (e.g. `i`, `2*i + u`)."""
    affine: Affine
    kind = "index"


@dataclass
class EVal(Val):
    """A scalar expression inside index notation."""
    expr: Expr
    kind = "index expression"

    @property
    def dtype(self):
        return self.expr.dtype


@dataclass
class TupleVal(Val):
    items: list
    kind = "tuple"


@dataclass
class FnVal(Val):
    decl: object
    scope: "Scope"
    name: str
    kind = "function"


@dataclass
class BuiltinVal(Val):
    name: str
    kind = "builtin function"


@dataclass
class MethodVal(Val):
    func: Val
    self_val: Val
    kind = "method"


@dataclass
class ModelDefVal(Val):
    decl: object
    scope: "Scope"
    kind = "model"


@dataclass
class ModelInstVal(Val):
    name: str
    scope: "Scope"
    decl: object
    kind = "model instance"


@dataclass
class OptDefVal(Val):
    decl: object
    scope: "Scope"
    kind = "optimizer"


@dataclass
class OptSpecVal(Val):
    opt: OptDefVal
    hyper: dict
    span: object = None
    kind = "optimizer"


@dataclass
class DistVal(Val):
    dist: str
    args: dict
    span: object = None
    kind = "distribution"


@dataclass
class RangeVal(Val):
    start: Val
    stop: Val
    step: int
    kind = "range"


@dataclass
class BatchesVal(Val):
    tensors: list
    size: int
    shuffle: bool
    span: object = None
    kind = "batches"


@dataclass
class Binding:
    val: Val
    ref: Buffer | None = None     # assignments write through to this buffer (params, state)
    what: str = "variable"        # variable | param | state | const | loop variable | function ...
    span: object = None


@dataclass
class Scope:
    parent: "Scope | None" = None
    kind: str = "global"          # global | fn | model | opt
    vars: dict = field(default_factory=dict)
    runtime_names: set = field(default_factory=set)   # names that must be runtime variables
    prefix: str = ""              # for model instances: "net.l1."

    def lookup(self, name: str) -> Binding | None:
        s = self
        while s is not None:
            b = s.vars.get(name)
            if b is not None:
                return b
            s = s.parent
        return None

    def all_names(self):
        s = self
        out = set()
        while s is not None:
            out.update(s.vars)
            s = s.parent
        return out
