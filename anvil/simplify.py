"""Smart constructors for scalar expressions (type promotion, folding, algebraic
simplification) and symbolic differentiation."""
from __future__ import annotations

import math

from .ir import (F32, I32, Acc, Binary, Const, Expr, Index, LetRef, Load, Rand, Select, Unary,
                 map_expr)


def const(v, dtype=F32) -> Const:
    return Const(v, dtype)


ZERO = Const(0.0)
ONE = Const(1.0)


def is_const(e: Expr, v=None) -> bool:
    return isinstance(e, Const) and (v is None or e.value == v)


def cast(e: Expr, dtype: str) -> Expr:
    if e.dtype == dtype:
        return e
    if isinstance(e, Const):
        return Const(e.value if dtype == F32 else int(e.value), dtype)
    return Unary(dtype, e, dtype)


def _fold_unary(op: str, v):
    try:
        if op == "neg": return -v
        if op == "abs": return abs(v)
        if op == "exp": return math.exp(min(v, 88.7))
        if op == "log": return math.log(v) if v > 0 else (-math.inf if v == 0 else math.nan)
        if op == "sqrt": return math.sqrt(v) if v >= 0 else math.nan
        if op == "rsqrt": return 1 / math.sqrt(v) if v > 0 else math.inf
        if op == "recip": return 1 / v if v != 0 else math.inf
        if op == "tanh": return math.tanh(v)
        if op == "sigmoid": return 1 / (1 + math.exp(-v)) if v > -88 else 0.0
        if op == "sin": return math.sin(v)
        if op == "cos": return math.cos(v)
        if op == "floor": return math.floor(v)
        if op == "ceil": return math.ceil(v)
        if op == "round": return round(v)
        if op == "sign": return (v > 0) - (v < 0)
        if op == "not": return 1.0 if v == 0 else 0.0
        if op == "detach": return v
    except (OverflowError, ValueError):
        return None
    return None


def mk_unary(op: str, a: Expr) -> Expr:
    if op in ("f32", "i32"):
        return cast(a, F32 if op == "f32" else I32)
    if op == "detach" and isinstance(a, (Const, Index, Rand)):
        return a
    if op in ("exp", "log", "sqrt", "tanh", "sigmoid", "sin", "cos", "rsqrt", "recip") and a.dtype != F32:
        a = cast(a, F32)
    if isinstance(a, Const):
        v = _fold_unary(op, a.value)
        if v is not None:
            if op in ("floor", "ceil", "round", "sign") and a.dtype == I32:
                return Const(v, I32)
            return Const(v, F32 if op not in ("neg", "abs") else a.dtype)
    if op == "neg" and isinstance(a, Unary) and a.op == "neg":
        return a.a
    if op in ("floor", "ceil", "round") and a.dtype == I32:
        return a
    return Unary(op, a)


def _fold_binary(op, x, y, dtype):
    try:
        if op == "add": return x + y
        if op == "sub": return x - y
        if op == "mul": return x * y
        if op == "div": return x / y if y != 0 else (math.copysign(math.inf, x) if x != 0 else math.nan)
        if op == "idiv": return x // y if y != 0 else None
        if op == "mod": return x % y if y != 0 else None
        if op == "max": return max(x, y)
        if op == "min": return min(x, y)
        if op == "pow":
            r = x ** y
            return None if isinstance(r, complex) else r
        if op == "lt": return float(x < y)
        if op == "le": return float(x <= y)
        if op == "gt": return float(x > y)
        if op == "ge": return float(x >= y)
        if op == "eq": return float(x == y)
        if op == "ne": return float(x != y)
        if op == "and": return float(bool(x) and bool(y))
        if op == "or": return float(bool(x) or bool(y))
    except (OverflowError, ZeroDivisionError):
        return None
    return None


def mk_binary(op: str, a: Expr, b: Expr) -> Expr:
    # ---- type promotion
    if op in ("div", "pow"):
        a, b = cast(a, F32), cast(b, F32) if op == "div" else b
        if op == "pow" and b.dtype == I32 and not isinstance(b, Const):
            b = cast(b, F32)
        if op == "pow" and isinstance(b, Const):
            b = Const(b.value, F32)
    elif op in ("and", "or"):
        a, b = cast(a, F32), cast(b, F32)
    elif a.dtype != b.dtype:
        a, b = cast(a, F32), cast(b, F32)
    # ---- constant folding
    if isinstance(a, Const) and isinstance(b, Const):
        rdtype = Binary(op, a, b).dtype
        v = _fold_binary(op, a.value, b.value, rdtype)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return Const(v, rdtype)
    # ---- algebra
    if op == "add":
        if is_const(a, 0): return b
        if is_const(b, 0): return a
        if isinstance(b, Unary) and b.op == "neg": return mk_binary("sub", a, b.a)
    elif op == "sub":
        if is_const(b, 0): return a
        if is_const(a, 0): return mk_unary("neg", b)
        if a == b and a.dtype == I32: return Const(0, I32)
        if isinstance(b, Unary) and b.op == "neg": return mk_binary("add", a, b.a)
    elif op == "mul":
        if is_const(a, 0) or is_const(b, 0): return Const(0, Binary(op, a, b).dtype)
        if is_const(a, 1): return b
        if is_const(b, 1): return a
        if is_const(a, -1): return mk_unary("neg", b)
        if is_const(b, -1): return mk_unary("neg", a)
        if isinstance(a, Unary) and a.op == "neg" and isinstance(b, Unary) and b.op == "neg":
            return mk_binary("mul", a.a, b.a)
        # fold constant factors: c1 * (c2 * x)
        if isinstance(a, Const) and isinstance(b, Binary) and b.op == "mul" and isinstance(b.a, Const):
            return mk_binary("mul", mk_binary("mul", a, b.a), b.b)
        if isinstance(b, Const) and not isinstance(a, Const):
            return mk_binary("mul", b, a)       # canonical: constants on the left
    elif op == "div":
        if is_const(b, 1): return a
        if is_const(a, 0) and isinstance(b, Const) and b.value != 0 and not math.isnan(b.value):
            return Const(0.0)        # (not for any x: 0 / 0 is nan, and `loss == loss` must see it)
        if isinstance(b, Const) and b.value != 0 and math.isfinite(b.value):
            return mk_binary("mul", Const(1.0 / b.value), a)
    elif op == "pow":
        if isinstance(b, Const):
            if b.value == 1: return a
            if b.value == 0: return Const(1.0)
            if b.value == 2: return mk_binary("mul", a, a)
            if b.value == 0.5: return mk_unary("sqrt", a)
            if b.value == -1: return mk_binary("div", Const(1.0), a)
    elif op in ("max", "min"):
        if a == b: return a
    return Binary(op, a, b)


def mk_select(c: Expr, a: Expr, b: Expr) -> Expr:
    if isinstance(c, Const):
        return a if c.value != 0 else b
    if a == b:
        return a
    if a.dtype != b.dtype:
        a, b = cast(a, F32), cast(b, F32)
    return Select(c, a, b)


def mk_neg(a): return mk_unary("neg", a)
def add(a, b): return mk_binary("add", a, b)
def sub(a, b): return mk_binary("sub", a, b)
def mul(a, b): return mk_binary("mul", a, b)
def div(a, b): return mk_binary("div", a, b)


# ----------------------------------------------------------------------------- derivatives

def deriv(e: Expr, wrt: Load) -> Expr:
    """∂e/∂wrt, treating every load structurally equal to `wrt` as the variable."""
    key = wrt.key()
    memo: dict[int, Expr] = {}

    def d(x: Expr) -> Expr:
        k = id(x)
        if k in memo:
            return memo[k]
        r = _d(x)
        memo[k] = r
        return r

    def _d(x: Expr) -> Expr:
        if isinstance(x, Load):
            return ONE if x.key() == key else ZERO
        if isinstance(x, (Const, Index, Rand, Acc, LetRef)):
            return ZERO
        if x.dtype == I32 and not isinstance(x, Unary):
            return ZERO
        if isinstance(x, Unary):
            op, a = x.op, x.a
            if op in ("detach", "floor", "ceil", "round", "sign", "not", "i32"):
                return ZERO
            if op == "f32":
                return ZERO if a.dtype == I32 else d(a)
            da = d(a)
            if is_const(da, 0):
                return ZERO
            if op == "neg": return mk_neg(da)
            if op == "abs": return mul(mk_unary("sign", a), da)
            if op == "exp": return mul(x, da)
            if op == "log": return div(da, a)
            if op == "sqrt": return div(mul(Const(0.5), da), x)
            if op == "rsqrt": return mul(mul(Const(-0.5), mul(x, mul(x, x))), da)
            if op == "recip": return mk_neg(mul(mul(x, x), da))
            if op == "tanh": return mul(sub(ONE, mul(x, x)), da)
            if op == "sigmoid": return mul(mul(x, sub(ONE, x)), da)
            if op == "sin": return mul(mk_unary("cos", a), da)
            if op == "cos": return mk_neg(mul(mk_unary("sin", a), da))
            raise NotImplementedError(f"derivative of {op}")
        if isinstance(x, Binary):
            op, a, b = x.op, x.a, x.b
            if op in ("lt", "le", "gt", "ge", "eq", "ne", "and", "or", "idiv", "mod"):
                return ZERO
            da, db = d(a), d(b)
            if is_const(da, 0) and is_const(db, 0):
                return ZERO
            if op == "add": return add(da, db)
            if op == "sub": return sub(da, db)
            if op == "mul": return add(mul(da, b), mul(a, db))
            if op == "div":
                # (da - (a/b) db) / b
                if is_const(db, 0):
                    return div(da, b)
                return div(sub(da, mul(x, db)), b)
            # ties route the gradient to the second operand, so relu'(0) = 0 as in PyTorch
            if op == "max": return mk_select(mk_binary("gt", a, b), da, db)
            if op == "min": return mk_select(mk_binary("lt", a, b), da, db)
            if op == "pow":
                if isinstance(b, Const):
                    c = b.value
                    return mul(mul(Const(c), mk_binary("pow", a, Const(c - 1))), da)
                t1 = mul(db, mk_unary("log", a)) if not is_const(db, 0) else ZERO
                t2 = mul(div(b, a), da) if not is_const(da, 0) else ZERO
                return mul(x, add(t1, t2))
            raise NotImplementedError(f"derivative of {op}")
        if isinstance(x, Select):
            return mk_select(x.c, d(x.a), d(x.b))
        raise NotImplementedError(type(x).__name__)

    return d(e)


def replace_subexpr(e: Expr, target: Expr, repl: Expr) -> Expr:
    tk = target.key()

    def f(x):
        return repl if x.key() == tk else x
    if e.key() == tk:
        return repl
    return map_expr(e, f)
