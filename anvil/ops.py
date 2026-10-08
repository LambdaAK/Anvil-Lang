"""Tensor-level operations: each one emits a kernel (or returns a zero-copy view)."""
from __future__ import annotations

from . import dims
from .diagnostics import AnvilError, fmt_shape
from .ir import (F32, I32, Acc, Affine, Buffer, Const, Expr, Index, Kernel, Load, Reduction,
                 ScalarRef, Store, Var, row_major_strides, subst_expr)
from .simplify import cast, mk_binary, mk_select, mk_unary
from .values import AffVal, CVal, EVal, IdxVal, TVal, Val

VAR_NAMES = ["i", "j", "k", "l", "m", "n", "p", "q"]

BINOP = {"+": "add", "-": "sub", "*": "mul", "/": "div", "//": "idiv", "%": "mod", "**": "pow",
         "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne",
         "and": "and", "or": "or", "max": "max", "min": "min"}


def describe(val: Val) -> str:
    if isinstance(val, TVal):
        return f"{val.dtype}{fmt_shape(val.shape)}"
    if isinstance(val, CVal):
        return f"constant {val.value!r}"
    return val.kind


def identity_offset(buf: Buffer, vars) -> Affine:
    return Affine(0, {v: s for v, s in zip(vars, row_major_strides(buf.shape)) if buf.shape})


def fresh_vars(shape, names=None) -> list[Var]:
    names = names or VAR_NAMES
    out = []
    for d, n in enumerate(shape):
        nm = names[d] if d < len(names) else f"i{d}"
        out.append(Var(nm, n))
    return out


def clone_kernel_vars(domain, red_vars, exprs_fn):
    """Alpha-rename vars so that no two kernels share Var objects."""
    m = {}
    nd = []
    for v in domain:
        nv = Var(v.name, v.size)
        m[v] = Affine.of(nv)
        nd.append(nv)
    nr = []
    for v in red_vars:
        nv = Var(v.name, v.size)
        m[v] = Affine.of(nv)
        nr.append(nv)
    return nd, nr, m


class OpsMixin:
    # ------------------------------------------------------------------ kernel emission
    def make_kernel(self, domain, out: Buffer, body: Expr, red_vars=(), red_op=None, span=None,
                    label="", accumulate=False, out_offset: Affine | None = None) -> Kernel:
        domain = list(domain)
        red_vars = list(red_vars)
        nd, nr, m = clone_kernel_vars(domain, red_vars, None)
        body = subst_expr(body, m)
        if out_offset is None:
            out_offset = identity_offset(out, nd)
        else:
            out_offset = out_offset.subst(m)
        red = Reduction(nr, red_op, body) if red_vars or red_op else None
        value = Acc(red.dtype) if red is not None else body
        if red is not None and red.op in ("argmax", "argmin"):
            value = Acc(I32)
        if value.dtype != out.dtype:
            value = cast(value, out.dtype)
        k = Kernel(domain=nd, stmts=[Store(out, out_offset, value, accumulate)], red=red,
                   label=label or self.current_label(), span=self.current_span() or span)
        return k

    def emit_map(self, shape, body_fn, span=None, label="", name=None, dtype=None, names=None) -> TVal:
        vars = fresh_vars(shape, names)
        body = body_fn(vars)
        if dtype is not None and body.dtype != dtype:
            body = cast(body, dtype)
        buf = self.new_temp(shape, body.dtype, name)
        self.emit_kernel(self.make_kernel(vars, buf, body, span=span, label=label))
        return TVal.of(buf)

    def emit_reduce(self, out_shape, red_shape, body_fn, op, span=None, label="", names=None) -> TVal:
        allv = fresh_vars(list(out_shape) + list(red_shape), names)
        ov, rv = allv[:len(out_shape)], allv[len(out_shape):]
        body = body_fn(ov, rv)
        if op == "mean":
            n = 1
            for d in red_shape:
                n *= d
            body = mk_binary("mul", Const(1.0 / n), cast(body, F32))
            op = "sum"
        dtype = I32 if op in ("argmax", "argmin") else body.dtype
        buf = self.new_temp(out_shape, dtype)
        if not rv:
            # reduction over nothing: just a map
            self.emit_kernel(self.make_kernel(ov, buf, body, span=span, label=label))
        else:
            self.emit_kernel(self.make_kernel(ov, buf, body, red_vars=rv, red_op=op, span=span, label=label))
        return TVal.of(buf)

    # ------------------------------------------------------------------ operands
    def scalar_expr(self, val: Val, span) -> Expr:
        """Expr for a value usable as a scalar anywhere (no vars needed)."""
        if isinstance(val, CVal):
            v = val.value
            return Const(int(v), I32) if val.is_int else Const(float(v), F32)
        if isinstance(val, AffVal):
            return self.aff_expr(val.affine)
        if isinstance(val, TVal) and val.rank == 0:
            return val.load([])
        raise AnvilError(f"expected a scalar, found {describe(val)}", span)

    def aff_expr(self, a: Affine) -> Expr:
        """Integer expression for an affine of runtime scalars / index vars."""
        e: Expr = Const(a.const, I32)
        for k, c in a.terms.items():
            if isinstance(k, ScalarRef):
                t = Load(k.buf, Affine(0))
            elif isinstance(k, Var):
                t = Index(Affine.of(k))
            else:
                t = k.load
            e = mk_binary("add", e, mk_binary("mul", Const(c, I32), t))
        return e

    def to_expr(self, val: Val, span) -> Expr:
        """Convert a value to a scalar Expr inside index notation."""
        if isinstance(val, EVal):
            return val.expr
        if isinstance(val, IdxVal):
            if val.affine.single_var() is not None or not val.affine.gathers():
                return Index(val.affine) if not val.affine.scalars() else self.aff_expr(val.affine)
            return self.aff_expr(val.affine)
        if isinstance(val, TVal) and val.rank > 0:
            raise AnvilError(f"tensor of shape {fmt_shape(val.shape)} used inside index notation without indices",
                           span, help="index it, e.g. `x[i]`, or reduce it first")
        return self.scalar_expr(val, span)

    def shape_of(self, val: Val, span) -> tuple:
        if isinstance(val, TVal):
            return val.shape
        if isinstance(val, (CVal, AffVal)):
            return ()
        raise AnvilError(f"expected a tensor or number, found {val.kind}", span)

    def bload(self, val: Val, out_vars, out_shape, span) -> Expr:
        """Load `val` broadcast to `out_shape` at `out_vars`."""
        if not isinstance(val, TVal):
            return self.scalar_expr(val, span)
        off = len(out_shape) - val.rank
        idx = []
        for d in range(val.rank):
            if val.shape[d] == 1 and out_shape[off + d] != 1:
                idx.append(Affine(0))
            else:
                idx.append(Affine.of(out_vars[off + d]))
        return val.load(idx)

    def broadcast(self, shapes, span, what="this operation", vals=None) -> tuple:
        r = max((len(s) for s in shapes), default=0)
        out = []
        for d in range(r):
            size = 1
            for s in shapes:
                k = d - (r - len(s))
                if k < 0:
                    continue
                n = s[k]
                if n == 1:
                    continue
                if size != 1 and n != size:
                    a, b = shapes[0], shapes[1] if len(shapes) > 1 else shapes[0]
                    raise AnvilError(f"cannot broadcast shapes {fmt_shape(a)} and {fmt_shape(b)}", span,
                                   label=f"{fmt_shape(a)} vs {fmt_shape(b)}",
                                   notes=[f"aligned from the right, dimension {d} is {size} on one side but {n} "
                                          f"on the other; sizes must match or be 1"])
                if dims.conflict(size, n):
                    a, b = shapes[0], shapes[1] if len(shapes) > 1 else shapes[0]
                    raise AnvilError(f"cannot combine shapes {fmt_shape(a)} and {fmt_shape(b)}: dimension {d} is "
                                   f"`{dims.show(size)}` on one side but `{dims.show(n)}` on the other", span,
                                   label=f"{dims.show_shape(a)} vs {dims.show_shape(b)}",
                                   notes=[dims.mismatch_note(size, n)], help=dims.same_size_help(size, n))
                if size == 1 or not isinstance(size, dims.Dim):   # keep a name if either side has one
                    size = n
            out.append(size)
        return tuple(out)

    # ------------------------------------------------------------------ elementwise
    def is_index_level(self, *vals) -> bool:
        return any(isinstance(v, (EVal, IdxVal)) for v in vals)

    def elementwise(self, fn, vals, span, label=""):
        """fn(list[Expr]) -> Expr applied elementwise with broadcasting."""
        if self.is_index_level(*vals):
            return EVal(fn([self.to_expr(v, span) for v in vals]))
        if all(isinstance(v, CVal) for v in vals):
            e = fn([self.scalar_expr(v, span) for v in vals])
            if isinstance(e, Const):
                return CVal(int(e.value) if e.dtype == I32 else e.value)
        shapes = [self.shape_of(v, span) for v in vals]
        out_shape = self.broadcast(shapes, span, vals=vals)
        return self.emit_map(out_shape, lambda vs: fn([self.bload(v, vs, out_shape, span) for v in vals]),
                             span=span, label=label)

    def unary_op(self, op: str, val: Val, span, label="") -> Val:
        if isinstance(val, CVal):
            e = mk_unary(op, self.scalar_expr(val, span))
            if isinstance(e, Const):
                return CVal(int(e.value) if e.dtype == I32 else e.value)
        if isinstance(val, AffVal) and op == "neg":
            return AffVal(-val.affine)
        if isinstance(val, IdxVal) and op == "neg":
            return IdxVal(-val.affine)
        return self.elementwise(lambda es: mk_unary(op, es[0]), [val], span, label or op)

    def binary_op(self, op: str, a: Val, b: Val, span, label="") -> Val:
        # compile-time constants: Python semantics
        if isinstance(a, CVal) and isinstance(b, CVal):
            return self.fold_binary(op, a, b, span)
        # affine integer arithmetic stays symbolic
        if op in ("add", "sub", "mul"):
            r = self.affine_arith(op, a, b)
            if r is not None:
                return r
        return self.elementwise(lambda es: mk_binary(op, es[0], es[1]), [a, b], span, label or op)

    def affine_arith(self, op, a, b):
        def aff(v):
            if isinstance(v, CVal) and v.is_int and not isinstance(v.value, bool):
                return Affine(int(v.value)), "c"
            if isinstance(v, AffVal):
                return v.affine, "a"
            if isinstance(v, IdxVal):
                return v.affine, "i"
            return None, None
        fa, ka = aff(a)
        fb, kb = aff(b)
        if fa is None or fb is None:
            return None
        kind = "i" if "i" in (ka, kb) else "a"
        if op == "add":
            r = fa + fb
        elif op == "sub":
            r = fa - fb
        else:
            if ka == "c":
                r = fb * fa.const
            elif kb == "c":
                r = fa * fb.const
            else:
                return None
        return IdxVal(r) if kind == "i" else AffVal(r)

    def fold_binary(self, op, a: CVal, b: CVal, span) -> Val:
        x, y = a.value, b.value
        try:
            if op == "add": r = x + y
            elif op == "sub": r = x - y
            elif op == "mul": r = x * y
            elif op == "div": r = x / y
            elif op == "idiv": r = x // y
            elif op == "mod": r = x % y
            elif op == "pow": r = x ** y
            elif op == "max": r = max(x, y)
            elif op == "min": r = min(x, y)
            elif op == "lt": r = x < y
            elif op == "le": r = x <= y
            elif op == "gt": r = x > y
            elif op == "ge": r = x >= y
            elif op == "eq": r = x == y
            elif op == "ne": r = x != y
            elif op == "and": r = bool(x) and bool(y)
            elif op == "or": r = bool(x) or bool(y)
            else:
                raise AnvilError(f"unsupported operation {op}", span)
        except ZeroDivisionError:
            raise AnvilError("division by zero in a constant expression", span)
        if isinstance(r, complex):
            raise AnvilError("constant expression has no real value", span)
        if isinstance(r, bool):
            return CVal(r)
        return CVal(r)

    def select_op(self, c: Val, a: Val, b: Val, span) -> Val:
        if isinstance(c, CVal):
            return a if c.value else b
        return self.elementwise(lambda es: mk_select(mk_binary("ne", es[0], Const(0.0)) if es[0].dtype == I32 else es[0],
                                                     es[1], es[2]), [c, a, b], span, "select")

    # ------------------------------------------------------------------ materialization / views
    def materialize(self, val: TVal, span, name=None) -> TVal:
        """Return an identity view (copy if needed)."""
        if val.is_identity() and not val.detached:
            return val
        return self.emit_map(val.shape, lambda vs: val.load([Affine.of(v) for v in vs]), span=span,
                             label="copy", name=name)

    def to_tensor(self, val: Val, span, dtype=None) -> TVal:
        if isinstance(val, TVal):
            if dtype and val.dtype != dtype:
                return self.emit_map(val.shape, lambda vs: cast(val.load([Affine.of(v) for v in vs]), dtype),
                                     span=span, label=dtype)
            return val
        if isinstance(val, (CVal, AffVal)):
            e = self.scalar_expr(val, span)
            if dtype:
                e = cast(e, dtype)
            return self.emit_map((), lambda vs: e, span=span, label="const")
        if isinstance(val, EVal):
            if self.expr_free_vars(val.expr):
                raise AnvilError("this index expression still has unbound indices", span)
            return self.emit_map((), lambda vs: val.expr if not dtype else cast(val.expr, dtype), span=span)
        raise AnvilError(f"expected a tensor, found {val.kind}", span)

    def view(self, val: TVal, new_shape, mapping, span) -> TVal:
        """mapping(new_pvars) -> list[Affine] index into val (one per val dim)."""
        pv = [Var(f"p{d}", n) for d, n in enumerate(new_shape)]
        idx = mapping(pv)
        tmpl = val.offset(idx)
        return TVal(val.buf, new_shape, tmpl, pv, val.detached)

    def transpose(self, val: TVal, perm, span) -> TVal:
        if sorted(perm) != list(range(val.rank)):
            raise AnvilError(f"invalid permutation {list(perm)} for a rank-{val.rank} tensor", span)
        new_shape = tuple(val.shape[p] for p in perm)

        def mapping(pv):
            idx = [None] * val.rank
            for new_d, old_d in enumerate(perm):
                idx[old_d] = Affine.of(pv[new_d])
            return idx
        return self.view(val, new_shape, mapping, span)

    def reshape(self, val: TVal, dims, span) -> TVal:
        dims = list(dims)
        n = val.numel
        if dims.count(-1) > 1:
            raise AnvilError("only one dimension can be -1 in reshape", span)
        if -1 in dims:
            known = 1
            for d in dims:
                if d != -1:
                    known *= d
            if known == 0 or n % known:
                raise AnvilError(f"cannot reshape {fmt_shape(val.shape)} ({n} elements) into {dims}", span)
            dims[dims.index(-1)] = n // known
        m = 1
        for d in dims:
            m *= d
        if m != n:
            raise AnvilError(f"cannot reshape {fmt_shape(val.shape)} ({n} elements) into {fmt_shape(dims)} "
                           f"({m} elements)", span)
        if not val.is_contiguous():
            val = self.materialize(val, span)
        # contiguous: base offset + row-major strides of the new shape
        base = val.tmpl.subst({p: Affine(0) for p in val.pvars})
        pv = [Var(f"p{d}", s) for d, s in enumerate(dims)]
        tmpl = base + Affine(0, {v: s for v, s in zip(pv, row_major_strides(dims))})
        return TVal(val.buf, tuple(dims), tmpl, pv, val.detached)

    # ------------------------------------------------------------------ matmul
    def matmul(self, a: Val, b: Val, span) -> Val:
        if not isinstance(a, TVal) or not isinstance(b, TVal):
            if self.is_index_level(a, b):
                raise AnvilError("`@` works on whole tensors; inside index notation write the sum explicitly",
                               span, help="e.g. `y[i] = sum W[i, j] * x[j]`")
            raise AnvilError(f"`@` needs two tensors, found {describe(a)} and {describe(b)}", span)
        if a.rank == 0 or b.rank == 0:
            raise AnvilError("`@` does not accept scalars; use `*`", span)
        sa, sb = a.shape, b.shape
        k = sa[-1]
        kb = sb[0] if b.rank <= 2 else sb[-2]
        if k != kb:
            raise AnvilError("shape mismatch in `@`", span,
                           label=f"cannot multiply {fmt_shape(sa)} by {fmt_shape(sb)}",
                           notes=[f"the inner dimensions must agree ({k} ≠ {kb})"])
        if dims.conflict(k, kb):
            raise AnvilError("shape mismatch in `@`", span,
                           label=f"cannot multiply {dims.show_shape(sa)} by {dims.show_shape(sb)}",
                           notes=[f"the inner dimensions must be the same dimension: " + dims.mismatch_note(k, kb)],
                           help=dims.same_size_help(k, kb))
        if isinstance(kb, dims.Dim) and not isinstance(k, dims.Dim):
            k = kb

        def mul_body(x, y):
            return mk_binary("mul", x, y)
        if b.rank == 1:
            out_shape = sa[:-1]

            def body(ov, rv):
                return mul_body(a.load([Affine.of(v) for v in ov] + [Affine.of(rv[0])]), b.load([Affine.of(rv[0])]))
            return self.emit_reduce(out_shape, (k,), body, "sum", span=span, label="matmul",
                                    names=self._mm_names(len(out_shape)) + ["k"])
        if b.rank == 2:
            m = sb[1]
            if a.rank == 1:
                out_shape = (m,)

                def body(ov, rv):
                    return mul_body(a.load([Affine.of(rv[0])]), b.load([Affine.of(rv[0]), Affine.of(ov[0])]))
                return self.emit_reduce(out_shape, (k,), body, "sum", span=span, label="matmul", names=["j", "k"])
            out_shape = sa[:-1] + (m,)
            nb = len(out_shape)

            def body(ov, rv):
                return mul_body(a.load([Affine.of(v) for v in ov[:-1]] + [Affine.of(rv[0])]),
                                b.load([Affine.of(rv[0]), Affine.of(ov[-1])]))
            return self.emit_reduce(out_shape, (k,), body, "sum", span=span, label="matmul",
                                    names=self._mm_names(nb - 1) + ["j", "k"])
        # batched
        if a.rank != b.rank or sa[:-2] != sb[:-2]:
            raise AnvilError(f"batched `@` needs matching batch dimensions: {fmt_shape(sa)} @ {fmt_shape(sb)}", span)
        bad = dims.first_conflict(sa[:-2], sb[:-2])
        if bad:
            raise AnvilError(f"batched `@` needs the same batch dimensions: {dims.show_shape(sa)} @ "
                           f"{dims.show_shape(sb)}", span, notes=[dims.mismatch_note(bad[1], bad[2])])
        out_shape = sa[:-1] + (sb[-1],)

        def body(ov, rv):
            batch = [Affine.of(v) for v in ov[:-2]]
            return mul_body(a.load(batch + [Affine.of(ov[-2]), Affine.of(rv[0])]),
                            b.load(batch + [Affine.of(rv[0]), Affine.of(ov[-1])]))
        return self.emit_reduce(out_shape, (k,), body, "sum", span=span, label="matmul")

    @staticmethod
    def _mm_names(n_lead):
        """Index names for the leading (row) dims of a matmul output."""
        return {0: [], 1: ["i"], 2: ["b", "i"], 3: ["a", "b", "i"]}.get(n_lead, [f"i{d}" for d in range(n_lead)])

    # ------------------------------------------------------------------ reductions
    def reduce_tensor(self, op: str, val: Val, axes, keepdims, span) -> Val:
        if isinstance(val, (CVal, AffVal)):
            if op in ("argmax", "argmin"):
                return CVal(0)
            return val
        if not isinstance(val, TVal):
            raise AnvilError(f"cannot reduce a {val.kind}", span)
        r = val.rank
        if axes is None:
            if op in ("argmax", "argmin"):
                axes = [r - 1] if r else []
            else:
                axes = list(range(r))
        axes = sorted({(a + r) % r if r else 0 for a in axes})
        for a in axes:
            if not 0 <= a < max(r, 1):
                raise AnvilError(f"axis {a} is out of range for a rank-{r} tensor", span)
        if op in ("argmax", "argmin") and len(axes) != 1:
            raise AnvilError(f"{op} reduces exactly one axis", span)
        keep = [d for d in range(r) if d not in axes]
        out_shape = tuple(val.shape[d] for d in keep)
        red_shape = tuple(val.shape[d] for d in axes)

        def body(ov, rv):
            idx = [None] * r
            for d, v in zip(keep, ov):
                idx[d] = Affine.of(v)
            for d, v in zip(axes, rv):
                idx[d] = Affine.of(v)
            return val.load(idx)
        res = self.emit_reduce(out_shape, red_shape, body, op, span=span, label=op)
        if keepdims:
            ks = tuple(1 if d in axes else val.shape[d] for d in range(r))
            res = self.reshape(res, ks, span)
        return res
