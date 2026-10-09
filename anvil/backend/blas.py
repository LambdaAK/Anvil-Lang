"""Matrix products through Apple's Accelerate (cblas_sgemm, which runs on the AMX matrix
coprocessor: about 4× faster than NEON on all cores for a 1024³ product, and faster even for
products as small as 32³).

Before code generation, `prepare` finds kernels whose reduction is a matrix product,

    C[..., i, j] (+)= Σ_{..., k} A[..., i, k] · B[..., k, j]

with A and B row- or column-major in (i, k) and (k, j), and C row-major in (i, j). Other indices
are batches: an index of the result repeats the product (a stack of matrices: the images of a
convolution, the heads of attention), and a summed one adds products together (the gradient of a
convolution's weights sums over the images). The code generator calls cblas_sgemm once per batch
element. A kernel whose product has a fused epilogue (a bias, an activation, an optimizer update)
is split in two: the product into a scratch tensor, then the epilogue as an elementwise kernel.

ANVIL_BLAS=0 turns this off; ANVIL_BLAS_MIN sets the smallest product (M·N·K per call) that uses it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .. import ir
from ..ir import F32, Acc, Affine, Binary, Buffer, Kernel, KernelStmt, Load, Reduction, Store, Var

BLAS_MIN = 1 << 15          # M·N·K below which Anvil's own kernels are as fast (32³; measured on an M3 Max)
BATCH_MIN = 1 << 20         # … for each call of a stack of products (a call costs a few µs; Anvil's
                            # kernel computes the whole stack in one parallel pass)
MAX_CALLS = 1 << 16         # batch elements (calls) above which Anvil's own kernel is used instead


@dataclass
class Loop:
    """A batch index: its extent and how far one step moves A, B and C (in elements)."""
    extent: int
    a: int
    b: int
    c: int
    summed: bool                    # a reduction index: the products are added together


@dataclass
class Gemm:
    M: int
    N: int
    K: int
    A: Buffer
    a_off: int
    trans_a: bool
    lda: int
    B: Buffer
    b_off: int
    trans_b: bool
    ldb: int
    C: Buffer
    c_off: int
    ldc: int
    beta: float                     # 1 when the kernel adds to C (+=)
    loops: list = field(default_factory=list)

    @property
    def calls(self) -> int:
        return ir.prod(lp.extent for lp in self.loops)


def enabled() -> bool:
    return os.environ.get("ANVIL_BLAS", "1") != "0"


def min_size() -> int:
    try:
        return int(os.environ.get("ANVIL_BLAS_MIN", BLAS_MIN))
    except ValueError:
        return BLAS_MIN


def plain(off: Affine, allowed) -> dict | None:
    """The coefficients of an offset made only of the given loop variables (and a constant)."""
    out = {}
    for t, c in off.terms.items():
        if not isinstance(t, Var) or t not in allowed:
            return None
        out[t] = c
    return out


def layout(coef_row: int, coef_col: int, rows: int, cols: int):
    """(transposed, leading dimension) of a matrix with element (r, c) at r·coef_row + c·coef_col,
    as BLAS sees a row-major operand, or None if neither index is contiguous."""
    if coef_col == 1 and coef_row >= max(cols, 1):
        return False, coef_row
    if coef_row == 1 and coef_col >= max(rows, 1):
        return True, coef_col
    return None


@dataclass
class Product:
    a: Load
    b: Load
    vi: Var
    vj: Var
    vk: Var
    la: tuple
    lb: tuple
    batch: list                     # domain variables other than i and j (and those merged into i)
    summed: list                    # reduction variables other than k (and those merged into k)
    M: int = 0                      # rows: i's extent, times that of batch indices merged into it
    K: int = 0                      # the summed length: k's, times that of summed indices merged into it
    rows: list = field(default_factory=list)   # i, then the batch indices merged into it (inner to outer)


def product(k: Kernel, size_min: int) -> Product | None:
    """k's reduction as a (batched) matrix product BLAS can compute, big enough to pay for the
    calls, or None."""
    if k.red is None or k.red.op != "sum" or not k.red.vars or not k.domain:
        return None
    body = k.red.body
    if not (isinstance(body, Binary) and body.op == "mul" and body.dtype == F32):
        return None
    x, y = body.a, body.b
    if not (isinstance(x, Load) and isinstance(y, Load) and x.buf.dtype == F32 and y.buf.dtype == F32):
        return None
    D, R = list(k.domain), list(k.red.vars)
    allv = set(D) | set(R)
    for a, b in ((x, y), (y, x)):
        ca, cb = plain(a.offset, allv), plain(b.offset, allv)
        if ca is None or cb is None:
            continue
        for vk in reversed(R):                             # the innermost summed index first
            if not ca.get(vk) or not cb.get(vk):
                continue
            # i: an index of A's alone, j: of B's alone; any other index is a batch (the image of a
            # convolution appears in one operand only, a head of attention in both)
            ii = [v for v in reversed(D) if ca.get(v) and not cb.get(v)]
            jj = [v for v in reversed(D) if cb.get(v) and not ca.get(v)]
            for vi in ii:
                for vj in jj:
                    # Indices next to k in memory, in both operands, extend it: Σ_b,i over [b, i, ...] is
                    # one sum over b·i rows (a weight's gradient sums over the batch and the tokens). A
                    # batch index next to i in A (and not in B) extends i the same way: one product of
                    # [b·i, k] rows, not a call per b. (match checks that it is next to i in C too.)
                    K, ks = vk.extent, [vk]
                    M, rows = vi.extent, [vi]
                    grown = True
                    while grown:
                        grown = False
                        for v in R:
                            if v not in ks and ca.get(v) and ca.get(v) == ca[vk] * K and cb.get(v) == cb[vk] * K:
                                K *= v.extent
                                ks.append(v)
                                grown = True
                        for v in D:
                            if v not in rows and v is not vj and not cb.get(v) and ca.get(v) and ca[v] == ca[vi] * M:
                                M *= v.extent
                                rows.append(v)
                                grown = True
                    N = vj.extent
                    if M < 2 or N < 2 or M * N * K < size_min:
                        continue
                    la = layout(ca[vi], ca[vk], M, K)
                    lb = layout(cb[vk], cb[vj], K, N)
                    if la is None or lb is None:
                        continue
                    batch = [v for v in D if v is not vj and v not in rows]
                    summed = [v for v in R if v not in ks]
                    if ir.prod(v.extent for v in batch + summed) > MAX_CALLS:
                        continue
                    if any(v.extent > 1 for v in batch) and (M * N * K < max(size_min, BATCH_MIN) or min(M, N) < 16):
                        continue           # many small calls: Anvil's own kernel is faster
                    return Product(a, b, vi, vj, vk, la, lb, batch, summed, M, K, rows)
    return None


def match(k: Kernel, size_min: int) -> Gemm | None:
    """k as cblas_sgemm calls: exactly a product, stored (or added) to a row-major matrix."""
    p = product(k, size_min)
    if p is None or len(k.stmts) != 1 or not isinstance(k.stmts[0], Store):
        return None
    st = k.stmts[0]
    if not isinstance(st.value, Acc):
        return None
    cc = plain(st.offset, set(k.domain))
    if cc is None or cc.get(p.vj, 0) != 1 or cc.get(p.vi, 0) < p.vj.extent:
        return None
    step = cc[p.vi]
    for v in p.rows[1:]:                                   # rows merged into i must be next to it in C too
        if cc.get(v, 0) != step * v_prev_extent(p.rows, v):
            return None
    if any(not cc.get(v) for v in p.batch if v.extent > 1):
        return None                                        # every element of the batch has its own result
    if st.buf.root in (p.a.buf.root, p.b.buf.root):
        return None                                        # BLAS does not allow the result to overlap
    loops = [Loop(v.extent, p.a.offset.coef(v), p.b.offset.coef(v), cc.get(v, 0), False) for v in p.batch]
    loops += [Loop(v.extent, p.a.offset.coef(v), p.b.offset.coef(v), 0, True) for v in p.summed]
    return Gemm(p.M, p.vj.extent, p.K, p.a.buf, p.a.offset.const, p.la[0], p.la[1],
                p.b.buf, p.b.offset.const, p.lb[0], p.lb[1], st.buf, st.offset.const, cc[p.vi],
                1.0 if st.accumulate else 0.0, [lp for lp in loops if lp.extent > 1])


def v_prev_extent(rows: list, v) -> int:
    """The rows of i and of the indices merged into it before v: how far v steps, in rows."""
    n = 1
    for r in rows:
        if r is v:
            return n
        n *= r.extent
    raise ValueError(v)


def split(k: Kernel, prog: ir.Program, p: Product) -> Kernel:
    """A product with an epilogue: the product into a scratch tensor (returned, a new kernel), and k
    itself becomes the epilogue, reading the product from the scratch tensor."""
    order = p.batch + list(reversed(p.rows)) + [p.vj]     # the scratch tensor's dimensions
    shape = tuple(v.extent for v in order)
    strides = dict(zip(order, ir.row_major_strides(shape)))
    scratch = Buffer(f"{k.name}_product", shape, F32, "temp")
    prog.buffers.append(scratch)
    new = {v: Var(v.name, v.extent) for v in list(k.domain) + list(k.red.vars)}
    m = {v: Affine.of(nv) for v, nv in new.items()}
    prod_k = Kernel(domain=[new[v] for v in k.domain],
                    stmts=[Store(scratch, Affine(0, {new[v]: s for v, s in strides.items()}), Acc(F32))],
                    red=Reduction([new[v] for v in k.red.vars], "sum", ir.subst_expr(k.red.body, m)),
                    name=f"{k.name}p", label=f"{k.label or 'matmul'} (product)", span=k.span)
    got = Load(scratch, Affine(0, dict(strides)))

    def put(e):
        return ir.map_expr(e, lambda x: got if isinstance(x, Acc) else x)
    for st in k.stmts:
        st.value = put(st.value)
    k.red = None
    return prod_k


def preview(p: Product) -> Gemm:
    """The Gemm a split product would become (its result is a fresh row-major tensor: C is None)."""
    return Gemm(p.M, p.vj.extent, p.K, p.a.buf, p.a.offset.const, p.la[0], p.la[1],
                p.b.buf, p.b.offset.const, p.lb[0], p.lb[1], None, 0, p.vj.extent, 0.0,
                [Loop(v.extent, p.a.offset.coef(v), p.b.offset.coef(v), 0, False) for v in p.batch if v.extent > 1]
                + [Loop(v.extent, p.a.offset.coef(v), p.b.offset.coef(v), 0, True) for v in p.summed if v.extent > 1])


def prepare(prog: ir.Program, size_min: int | None = None, accept=None) -> dict:
    """Find (and split off) the products that go to a matrix-multiply library: {kernel id: Gemm}.
    accept(gemm) may refuse one (for a split product, the gemm's C is None)."""
    if not enabled():
        return {}
    size_min = min_size() if size_min is None else size_min
    accept = accept or (lambda g: True)
    found = {}
    for blk in ir.walk_blocks(prog.main):
        out = []
        for st in blk.stmts:
            if isinstance(st, KernelStmt):
                k = st.kernel
                g = match(k, size_min)
                if g is not None and accept(g):
                    found[k.id] = g
                elif g is None:
                    p = product(k, size_min)
                    if p is not None and not any(s.accumulate for s in k.stores) and accept(preview(p)):
                        prod_k = split(k, prog, p)
                        g = match(prod_k, size_min)
                        if g is None:                        # (the scratch layout always matches)
                            raise AssertionError("blas: a split product does not match")
                        found[prod_k.id] = g
                        out.append(KernelStmt(prod_k))
            out.append(st)
        blk.stmts = out
    return found
