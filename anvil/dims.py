"""Dimensions that remember where they came from.

A shape is a tuple of ints. Here an int can also be a `Dim`: an int that carries a formula over named
sizes, such as `T`, `BATCH*T`, `D/HEADS` or `H - K + 1`. Every integer `const` is a name; arithmetic
on dims keeps the formula (an exact division stays symbolic, as a negative power), and everything that
only looks at the number (comparisons, hashing, printing, code generation) sees a plain int.

Two dimensions that must agree (the same index over two tensors, the two sides of `+`, the inner
dimensions of `@`) agree if their numbers are equal. With formulas they must also be the same
dimension: `T` and `D` are different dimensions even when both are 64, so a program that would fail
at other sizes does not compile at these. (A size with no name, a literal 784 or a dimension read
from a data file, matches any formula of the same value.)
"""
from __future__ import annotations

import contextvars
from fractions import Fraction

# a monomial: ((name, exponent), ...) sorted by name; a formula: {monomial: coefficient}


class Sym:
    __slots__ = ("terms", "_key")

    def __init__(self, terms: dict):
        self.terms = {m: c for m, c in terms.items() if c != 0}
        self._key = tuple(sorted(self.terms.items()))

    @staticmethod
    def atom(name: str) -> "Sym":
        return Sym({((name, 1),): Fraction(1)})

    @staticmethod
    def const(n) -> "Sym":
        return Sym({(): Fraction(n)})

    def __eq__(self, other):
        return isinstance(other, Sym) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def is_constant(self) -> bool:
        return all(m == () for m in self.terms)

    def monomial(self):
        """(monomial, coefficient) if this formula is a single term, else None."""
        if len(self.terms) == 1:
            (m, c), = self.terms.items()
            return m, c
        return None

    def __add__(self, other: "Sym") -> "Sym":
        t = dict(self.terms)
        for m, c in other.terms.items():
            t[m] = t.get(m, 0) + c
        return Sym(t)

    def __neg__(self) -> "Sym":
        return Sym({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: "Sym") -> "Sym":
        return self + (-other)

    def __mul__(self, other: "Sym") -> "Sym":
        t: dict = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = mono_mul(m1, m2)
                t[m] = t.get(m, 0) + c1 * c2
        return Sym(t)

    def inverse(self) -> "Sym | None":
        """1 / this formula, if it is a single term."""
        mc = self.monomial()
        if mc is None:
            return None
        m, c = mc
        return Sym({tuple((n, -e) for n, e in m): 1 / c})

    def __str__(self):
        return render(self)


def mono_mul(a, b):
    exps: dict = {}
    for n, e in a + b:
        exps[n] = exps.get(n, 0) + e
    return tuple(sorted((n, e) for n, e in exps.items() if e != 0))


def render_mono(m, c) -> str:
    num = [n if e == 1 else f"{n}^{e}" for n, e in m if e > 0]
    den = [n if e == -1 else f"{n}^{-e}" for n, e in m if e < 0]
    c = abs(c)
    if c.denominator != 1:
        den.insert(0, str(c.denominator))
    if c.numerator != 1 or not num:
        num.insert(0, str(c.numerator))
    s = "*".join(num)
    if den:
        s += "/" + "/".join(den)
    return s


def render(sym: Sym, aliases: bool = True) -> str:
    names = DIM_NAMES.get() if aliases else None
    if names and sym in names:
        return names[sym]
    order = sorted(sym.terms.items(), key=lambda mc: (-sum(e for _, e in mc[0]), mc[0]))
    out = ""
    for m, c in order:
        part = render_mono(m, c)
        if not out:
            out = ("-" if c < 0 else "") + part
        else:
            out += (" - " if c < 0 else " + ") + part
    return out or "0"


# what derived constants are called (`const DH = D // HEADS`): formula -> name, for display; set by the
# elaborator for the program it compiles
DIM_NAMES: contextvars.ContextVar = contextvars.ContextVar("anvil_dim_names", default=None)


class Dim(int):
    """An int with a formula. Only arithmetic is overridden: it prints, compares and hashes as an int."""

    def __new__(cls, value: int, sym: Sym):
        d = int.__new__(cls, value)
        d.sym = sym
        return d

    def __reduce__(self):
        return (int, (int(self),))

    def __add__(self, o): return combine(self, o, "add")
    def __radd__(self, o): return combine(o, self, "add")
    def __sub__(self, o): return combine(self, o, "sub")
    def __rsub__(self, o): return combine(o, self, "sub")
    def __mul__(self, o): return combine(self, o, "mul")
    def __rmul__(self, o): return combine(o, self, "mul")
    def __floordiv__(self, o): return combine(self, o, "div")
    def __rfloordiv__(self, o): return combine(o, self, "div")

    def __neg__(self):
        return Dim(-int(self), -self.sym)

    def __pos__(self):
        return self


def sym_of(x) -> Sym | None:
    if isinstance(x, Dim):
        return x.sym
    if isinstance(x, int) and not isinstance(x, bool):
        return Sym.const(x)
    return None


def combine(a, b, op):
    """a op b for ints, at least one of them a Dim."""
    if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool) or isinstance(b, bool):
        x, y = (int(a) if isinstance(a, Dim) else a), (int(b) if isinstance(b, Dim) else b)
        return {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y, "div": lambda: x // y}[op]()
    x, y = int(a), int(b)
    sa, sb = sym_of(a), sym_of(b)
    if op == "add":
        return make(x + y, sa + sb)
    if op == "sub":
        return make(x - y, sa - sb)
    if op == "mul":
        return make(x * y, sa * sb)
    value = x // y
    inv = sb.inverse() if y != 0 else None
    if inv is not None and x % y == 0:                      # an exact division by a single term: D/HEADS
        return make(value, sa * inv)
    return value                                              # a floor division: only the number


def make(value: int, sym: Sym):
    return int(value) if sym.is_constant() else Dim(value, sym)


def named(value: int, name: str) -> int:
    """A constant's value as the dimension it names (`const T = 64`)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    if isinstance(value, Dim):                                # `const DH = D // HEADS`: keep the formula
        return value
    return Dim(value, Sym.atom(name))


def show(n) -> str:
    """A dimension for people: its formula if it has one, else its number."""
    return str(n.sym) if isinstance(n, Dim) else str(n)


def formula(n) -> str | None:
    """How a derived size is computed (`D/HEADS` for `const DH = D // HEADS`), or None for a plain name."""
    if not isinstance(n, Dim):
        return None
    mc = n.sym.monomial()
    if mc is not None and mc[1] == 1 and len(mc[0]) == 1 and mc[0][0][1] == 1:
        return None
    return render(n.sym, aliases=False)


def show_shape(shape) -> str:
    return "[" + ", ".join(show(d) for d in shape) + "]"


def has_names(shape) -> bool:
    return any(isinstance(d, Dim) for d in shape)


def conflict(a, b) -> bool:
    """True if a and b are the same number but different dimensions (`T` and `D`, both 64)."""
    return isinstance(a, Dim) and isinstance(b, Dim) and int(a) == int(b) and a.sym != b.sym


def first_conflict(sa, sb):
    """(axis, a, b) for the first axis where two shapes of equal sizes name different dimensions."""
    for k, (a, b) in enumerate(zip(sa, sb)):
        if conflict(a, b):
            return k, a, b
    return None


def adopt(declared, actual):
    """A shape's names, taken from a declaration where the value has none (`images: [N, 28, 28] = idx(...)`)."""
    return tuple(d if isinstance(d, Dim) and not isinstance(a, Dim) and int(d) == int(a) else a
                 for d, a in zip(declared, actual))


def mismatch_note(a, b) -> str:
    return (f"`{show(a)}` and `{show(b)}` are both {int(a)} here, but they are different dimensions: the program "
            f"would fail with other sizes")


def same_size_help(a, b) -> str:
    return (f"if they are meant to be the same size, define one from the other (e.g. `const {show(b)} = {show(a)}`)"
            if isinstance(b, Dim) and b.sym.monomial() and len(b.sym.monomial()[0]) == 1 else
            "use the same dimension on both sides")
