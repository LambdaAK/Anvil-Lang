"""Reverse-mode automatic differentiation over kernels, plus `minimize`.

For a forward kernel  Y[I] = ⊕_J body(I, J)  and a load X[f(I, J)] in its body:

    dX[f(I, J)] += dY[I] · ∂body/∂X[f(I, J)]

If f is an injective map of the loop variables (the usual case) this is emitted as a
regular reduction kernel over the variables f does not use. Otherwise (convolution
windows, gathers, runtime offsets) it becomes a scatter-accumulate kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ast as A
from . import ir
from .diagnostics import AnvilError
from .ir import F32, I32, Affine, Buffer, Const, Kernel, KernelStmt, Load, ScalarRef, Unary
from .simplify import deriv, is_const, mk_binary, mk_select, replace_subexpr
from .values import AffVal, Binding, CVal, EVal, ModelInstVal, OptDefVal, OptSpecVal, Scope, TVal


@dataclass
class GradInfo:
    buf: Buffer
    initialized: bool = False


def ad_region(elab) -> list[Kernel]:
    """Kernels that ran earlier in the current loop iteration (or program, outside loops)."""
    blocks = elab.blocks
    start = 0
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i].is_loop:
            start = i
            break
    out = []
    for b in blocks[start:]:
        for st in b.stmts:
            if isinstance(st, KernelStmt):
                out.append(st.kernel)
    return out


def diff_loads(e) -> list[Load]:
    """f32 loads not hidden behind detach()."""
    out = []

    def walk(x):
        if isinstance(x, Unary) and x.op == "detach":
            return
        if isinstance(x, Load):
            if x.dtype == F32:
                out.append(x)
            return
        for c in x.children():
            walk(c)
    walk(e)
    return out


def kernel_diff_loads(k: Kernel) -> list[Load]:
    out = []
    for e in k.exprs():
        out.extend(diff_loads(e))
    return out


def injective(off: Affine, vars_) -> bool:
    items = sorted(((abs(off.coef(v)), v.extent) for v in vars_ if off.coef(v) != 0 and v.extent > 1))
    reach = 0
    for c, n in items:
        if c <= reach:
            return False
        reach += c * (n - 1)
    return True


def covers(off: Affine, vars_, numel: int) -> bool:
    if off.const != 0:
        return False
    items = sorted((off.coef(v), v.extent) for v in vars_ if off.coef(v) != 0 and v.extent > 1)
    expect = 1
    for c, n in items:
        if c != expect:
            return False
        expect *= n
    return expect == numel


class Backprop:
    def __init__(self, elab, region: list[Kernel], span):
        self.elab = elab
        self.region = region
        self.span = span
        self.grads: dict[Buffer, GradInfo] = {}

    def run(self, loss: Buffer, targets: list[Buffer], seed: float):
        producers: dict[Buffer, list[Kernel]] = {}      # every kernel that writes each buffer
        for k in self.region:
            for st in k.stores:
                producers.setdefault(st.buf.root, []).append(k)
        target_set = {t.root for t in targets}
        # backward reachability from the loss (targets are leaves)
        need: set[Buffer] = set()
        stack = [loss.root]
        while stack:
            b = stack.pop()
            if b in need:
                continue
            need.add(b)
            if b in target_set:
                continue
            for k in producers.get(b, []):
                for ld in kernel_diff_loads(k):
                    stack.append(ld.buf.root)
        # forward reachability from the targets
        fwd = set(target_set)
        for k in self.region:
            if any(ld.buf.root in fwd for ld in kernel_diff_loads(k)):
                fwd |= k.writes()
        active = need & fwd
        if loss.root not in active:
            return {}
        # mutation hazard: a non-SSA buffer read by the active computation is overwritten later
        for i, k in enumerate(self.region):
            if not (k.writes() & active):
                continue
            for b in {ld.buf.root for e in k.exprs() for ld in ir.loads_in(e)}:
                if b.kind in ("temp", "grad", "const", "data", "input"):
                    continue
                if any(b in later.writes() for later in self.region[i + 1:]):
                    raise AnvilError(f"`{b.name}` is modified after it was used to compute the loss, so its "
                                   f"gradient would be wrong", self.span,
                                   help="compute the loss after the update, or take the gradient before it")
        self.grads[loss.root] = GradInfo(None, True)
        self.seed = seed
        for k in reversed(self.region):
            if len(k.stores) != 1:
                continue
            st = k.stores[0]
            y = st.buf.root
            if y not in active or y not in self.grads:
                continue
            if y in target_set and y is not loss.root:
                continue
            if st.accumulate and y.kind == "grad":
                raise AnvilError("differentiating through a gradient (higher-order derivatives) is not supported yet",
                               self.span)
            # (an accumulating store, `y[…] += f(x)`, passes y's gradient on unchanged to whatever
            # wrote y before; f's inputs get theirs below)
            self.backward_kernel(k, st, active)
            if not st.accumulate and any(p is not k for p in producers[y][:producers[y].index(k)]):
                self.zero_overwritten(y, k, st)
        return self.grads

    def zero_overwritten(self, y: Buffer, k: Kernel, st):
        """k overwrote part of y (`x[1] = …`): what was there before does not reach the loss."""
        gi = self.grads[y]
        if gi.buf is None:
            return
        self.elab.emit_kernel(self.elab.make_kernel(k.domain, gi.buf, Const(0.0), span=k.span or self.span,
                                                    label="zero", out_offset=st.offset))

    def dy(self, y: Buffer, off: Affine):
        gi = self.grads[y]
        if gi.buf is None:
            return Const(self.seed)
        return Load(gi.buf, off)

    def backward_kernel(self, k: Kernel, st, active):
        y = st.buf.root
        red = k.red
        if red is None:
            body = st.value
            rvars = []
        else:
            if red.op in ("argmax", "argmin"):
                return
            body = red.body
            rvars = red.vars
        dvars = list(k.domain)
        dy = self.dy(y, st.offset)
        yload = Load(st.buf, st.offset)
        seen = {}
        for ld in diff_loads(body):
            if ld.buf.root in active:
                seen.setdefault(ld.key(), ld)
        for ld in seen.values():
            g = deriv(body, ld)
            if is_const(g, 0):
                continue
            if red is None and st.buf.kind == "temp" and body.dtype == F32 and not st.accumulate:
                g = replace_subexpr(g, body, yload)       # y holds the body's value: reuse it
            if red is not None and red.op == "prod":
                g = mk_binary("mul", g, self.prod_factor(k, body))
            if red is not None and red.op in ("max", "min"):
                if st.accumulate:
                    raise AnvilError("differentiating `+= max …` is not supported", k.span or self.span)
                g = mk_select(mk_binary("eq", body, yload), g, Const(0.0))
            contrib = mk_binary("mul", dy, g)
            self.emit_grad(ld.buf.root, ld.offset, contrib, dvars + list(rvars), k)

    def prod_factor(self, k: Kernel, body):
        """∂(Π_J body)/∂body at one J: the product of the other factors. With nz the product of
        the non-zero factors and zc the number of zeros, that is nz / body if there are no zeros,
        nz at the zero if there is exactly one, and 0 otherwise."""
        cache = self.__dict__.setdefault("_prod", {})
        if k.id not in cache:
            shape = tuple(v.extent for v in k.domain)
            nz = self.elab.new_temp(shape, F32)
            zc = self.elab.new_temp(shape, F32)
            zero = mk_binary("eq", body, Const(0.0))
            for buf, e, op in ((nz, mk_select(zero, Const(1.0), body), "prod"), (zc, zero, "sum")):
                self.elab.emit_kernel(self.elab.make_kernel(k.domain, buf, e, red_vars=k.red.vars, red_op=op,
                                                            span=k.span or self.span, label="∂prod"))
            cache[k.id] = (nz, zc)
        nz, zc = cache[k.id]
        off = Affine(0)
        stride = 1
        for v in reversed(k.domain):
            off = off + Affine(0, {v: stride})
            stride *= v.extent
        nzv, zcv = Load(nz, off), Load(zc, off)
        others = mk_select(mk_binary("eq", zcv, Const(0.0)), mk_binary("div", nzv, body), Const(0.0))
        at_zero = mk_select(mk_binary("eq", zcv, Const(1.0)), nzv, Const(0.0))
        return mk_select(mk_binary("eq", body, Const(0.0)), at_zero, others)

    def grad_buf(self, x: Buffer) -> GradInfo:
        gi = self.grads.get(x)
        if gi is None:
            gbuf = self.elab.new_buffer("d" + x.name, x.shape, F32, "grad")
            gi = GradInfo(gbuf, False)
            self.grads[x] = gi
        return gi

    def zero_fill(self, gi: GradInfo):
        vars_ = [ir.Var(n, d) for n, d in zip("ijklmnpq", gi.buf.shape)]
        self.elab.emit_kernel(self.elab.make_kernel(vars_, gi.buf, Const(0.0), span=self.span, label="zero"))
        gi.initialized = True

    def emit_grad(self, x: Buffer, off: Affine, contrib, allvars, k: Kernel):
        gi = self.grad_buf(x)
        label = f"∂{k.label}" if k.label else "∂"
        span = k.span or self.span
        regular = not off.scalars() and not off.gathers() and injective(off, allvars)
        if regular:
            S = [v for v in allvars if off.coef(v) != 0]
            S.sort(key=lambda v: -abs(off.coef(v)))
            red_vars = [v for v in allvars if v not in S]
            full = covers(off, S, x.numel)
            accumulate = gi.initialized
            if not gi.initialized and not full:
                self.zero_fill(gi)
                accumulate = True
            kern = self.elab.make_kernel(S, gi.buf, contrib, red_vars=red_vars, red_op="sum" if red_vars else None,
                                         span=span, label=label, accumulate=accumulate, out_offset=off)
            self.elab.emit_kernel(kern)
            gi.initialized = True
            return
        if not gi.initialized:
            self.zero_fill(gi)
        # Scatter-accumulate. Variables the index does not mention are summed in registers
        # first; the rest are ordered so the innermost loop is contiguous in the store.
        used = set(off.direct_vars())
        for g in off.gathers():
            used.update(g.load.offset.vars())
        dom = [v for v in allvars if v in used]
        red_vars = [v for v in allvars if v not in used]
        dom.sort(key=lambda v: (-abs(off.coef(v)) if off.coef(v) else -(1 << 60), v.extent))
        kern = self.elab.make_kernel(dom, gi.buf, contrib, red_vars=red_vars,
                                     red_op="sum" if red_vars else None, span=span, label=label,
                                     accumulate=True, out_offset=off)
        self.elab.emit_kernel(kern)


def as_loss(elab, val, span) -> TVal:
    if isinstance(val, EVal):
        val = elab.to_tensor(val, span)
    if isinstance(val, (CVal, AffVal)):
        raise AnvilError("the loss is a constant, so it has no gradient", span)
    if not isinstance(val, TVal):
        raise AnvilError(f"the loss must be a tensor, found a {val.kind}", span)
    if val.rank != 0:
        raise AnvilError(f"the loss must be a scalar, but it has shape [{', '.join(map(str, val.shape))}]", span,
                       help="reduce it first, e.g. `mean(...)` or `sum(...)`")
    if val.dtype != F32:
        raise AnvilError("the loss must be f32", span)
    if not val.is_identity() or val.detached:
        val = elab.materialize(TVal(val.buf, val.shape, val.tmpl, val.pvars), span)
    return val


def grad(elab, loss_val, targets, node) -> list[TVal]:
    loss = as_loss(elab, loss_val, node.args[0].span)
    bufs = []
    for t in targets:
        tv, tnode = t if isinstance(t, tuple) else (t, node)
        if not tv.is_identity():
            raise AnvilError("can only differentiate with respect to a whole tensor variable, not a view", tnode.span,
                           help="copy it first: `x = copy(x)` before using it")
        bufs.append(tv.buf)
    bp = Backprop(elab, ad_region(elab), node.span)
    grads = bp.run(loss.buf, bufs, 1.0)
    out = []
    for b, t in zip(bufs, targets):
        gi = grads.get(b.root)
        if gi is None or gi.buf is None:
            elab.warnings.append(AnvilError(f"the loss does not depend on `{b.name}`; its gradient is zero",
                                          node.span, kind="warning"))
            out.append(elab.emit_map(b.shape, lambda vs: Const(0.0), span=node.span, label="zero"))
        else:
            out.append(TVal.of(gi.buf))
    return out


def clip_gradients(elab, spec, params, grads, span) -> dict:
    """clip_norm > 0: all the gradients scaled by min(1, clip_norm / their combined norm)."""
    clip = spec.hyper.get("clip_norm")
    if clip is None:
        return {}
    if not isinstance(clip, CVal):
        raise AnvilError("clip_norm must be a constant", span)
    if clip.value <= 0:
        return {}
    total = None
    for p in params:
        g = TVal.of(grads[p.root].buf)
        sq = elab.reduce_tensor("sum", elab.binary_op("mul", g, g, span), None, False, span)
        total = sq if total is None else elab.binary_op("add", total, sq, span)
    norm = elab.unary_op("sqrt", total, span)
    scale = elab.binary_op("min", CVal(1.0), elab.binary_op("div", CVal(float(clip.value)),
                                                            elab.binary_op("add", norm, CVal(1e-6), span), span), span)
    out = {}
    for p in params:
        g = elab.binary_op("mul", TVal.of(grads[p.root].buf), scale, span)
        out[p.root] = elab.materialize(g, span).buf
    return out


def minimize(elab, s: A.Minimize):
    span = s.span
    loss = as_loss(elab, elab.eval_top(s.objective), s.objective.span)
    spec = elab.eval_top(s.optimizer)
    if isinstance(spec, OptDefVal):
        spec = elab.optimizer_spec(spec, [], {}, s.optimizer)
    if not isinstance(spec, OptSpecVal):
        raise AnvilError(f"expected an optimizer after `with`, found a {spec.kind}", s.optimizer.span,
                       help="e.g. `with sgd(lr=0.1)` or `with adam()`")
    if s.over is not None:
        targets = []
        for o in s.over:
            v = elab.eval_top(o)
            if isinstance(v, ModelInstVal):
                targets.extend(elab.model_params(v))
            elif isinstance(v, TVal) and v.buf.kind == "param":
                targets.append(v.buf)
            else:
                raise AnvilError("`over` takes parameters or models", o.span)
    else:
        targets = [b for b in elab.buffers if b.kind == "param"]
    bp = Backprop(elab, ad_region(elab), span)
    grads = bp.run(loss.buf, targets, -1.0 if s.maximize else 1.0)
    params = [p for p in targets if p.root in grads and grads[p.root].buf is not None]
    if not params:
        raise AnvilError("the objective does not depend on any parameter", s.objective.span,
                       help="gradients flow through computations in the current loop iteration; "
                            "parameters must be declared with `param` (or inside a model)")
    untrained = [p for p in targets if p not in params]
    if untrained:
        names = ", ".join(f"`{p.name}`" for p in untrained[:4]) + (f" and {len(untrained) - 4} more"
                                                                   if len(untrained) > 4 else "")
        one = len(untrained) == 1
        elab.warnings.append(AnvilError(
            f"the objective does not depend on {names}, so `minimize` cannot train {'it' if one else 'them'}",
            span, kind="warning",
            help="is the parameter used in the computation of the objective (and not behind `detach`)?"))
    elab.trained_params.update(p.root for p in params)
    decl: A.OptimizerDecl = spec.opt.decl
    elab.minimize_sites += 1
    site = elab.minimize_sites
    opt_name = decl.name.id
    step_params = [n.id for n in decl.step_params]
    t_val = None
    if len(step_params) == 3:
        t_buf = elab.new_buffer(f"{opt_name}{site}.t", (), I32, "state")
        elab.write_buffer(t_buf, elab.binary_op("add", TVal.of(t_buf), CVal(1), span), span, fresh=True)
        t_val = AffVal(Affine.of(ScalarRef(t_buf)))
    clipped = clip_gradients(elab, spec, params, grads, span)
    elab.fn_stack.append(opt_name)
    try:
        for p in params:
            g = clipped.get(p.root) or grads[p.root].buf
            scope = Scope(parent=spec.opt.scope, kind="opt")
            for k, v in spec.hyper.items():
                scope.vars[k] = Binding(v, what="const" if isinstance(v, CVal) else "variable")
            scope.vars[step_params[0]] = Binding(TVal.of(p), ref=p, what="param")
            scope.vars[step_params[1]] = Binding(TVal.of(g), what="gradient")
            if t_val is not None:
                scope.vars[step_params[2]] = Binding(t_val, what="step count")
            for st_name in decl.state:
                sb = elab.new_buffer(f"{p.name}.{st_name.id}", p.shape, F32, "state")
                scope.vars[st_name.id] = Binding(TVal.of(sb), ref=sb, what="state")
            from .elaborate import compute_runtime_names
            scope.runtime_names = compute_runtime_names(decl.step_body)
            saved = elab.scope
            elab.scope = scope
            try:
                for st in decl.step_body:
                    elab.exec_stmt(st)
            finally:
                elab.scope = saved
    finally:
        elab.fn_stack.pop()
