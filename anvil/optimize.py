"""Kernel-level optimizations.

* write-back coalescing: the final producer of a loop-carried variable writes straight
  into its canonical buffer (this is how in-place updates arise)
* elementwise inlining: a producer is substituted into its consumer(s) when no work is
  duplicated (or when it is trivially cheap)
* vertical fusion: an elementwise consumer runs inside its producer's store loop
  (`relu(x @ W + b)` becomes one kernel; so does the SGD update of a weight gradient)
* dead code / dead store elimination, including state that is never read
"""
from __future__ import annotations

from . import ir
from .ir import (Affine, Block, Expr, Gather, Index, Kernel, KernelStmt, Let, LetRef, Load,
                 ScalarRef, Select, Store, Unary, Binary, map_expr, row_major_strides, subst_expr)
from .simplify import cast, mk_binary, mk_select, mk_unary

TEMP_KINDS = ("temp", "grad")
# the buffer arguments each runtime call reads, and those it writes (a buffer or a list of them)
RT_READS = {"shuffle": ("buf",), "show": ("buf",), "sleep": ("buf",), "nonzero": ("src",), "save": ("bufs",),
            "save_npy": ("buf",)}
RT_WRITES = {"load_idx": ("buf",), "load_csv": ("buf",), "load_bytes": ("buf",), "load_npy": ("buf",), "iota": ("buf",),
             "iota_once": ("buf",), "shuffle": ("buf",), "clock": ("buf",), "nonzero": ("buf",),
             "input": ("buf",), "load": ("bufs", "buf")}


def rt_buffers(st, table) -> list:
    out = []
    for key in table.get(st.name, ()):
        v = st.args[key]
        out.extend(v if isinstance(v, list) else [v])
    return out
MAX_FUSED_NODES = 160
GEMM_EPILOGUE = 8          # the most expression nodes fused after a matmul-like reduction
GEMM_REUSE = 16            # a matmul reading each element of an operand more often than this keeps it in memory
MAX_FUSED_STORES = 6
MAX_FUSED_LOADS = 12
CANON = [ir.Var(f"_v{i}") for i in range(12)]


# ----------------------------------------------------------------------------- read/write sets

def affine_reads(a: Affine, out: set):
    for t in a.terms:
        if isinstance(t, ScalarRef):
            out.add(t.buf.root)
        elif isinstance(t, Gather):
            out.add(t.load.buf.root)
            affine_reads(t.load.offset, out)


def expr_reads(e: Expr, out: set):
    for x in ir.iter_expr(e):
        if isinstance(x, Load):
            out.add(x.buf.root)
            affine_reads(x.offset, out)
        elif isinstance(x, Index):
            affine_reads(x.affine, out)


def kernel_reads(k: Kernel, include_acc: bool = True) -> set:
    out: set = set()
    for e in k.exprs():
        expr_reads(e, out)
    for s in k.stores:
        affine_reads(s.offset, out)
        if s.accumulate and include_acc:
            out.add(s.buf.root)
    if k.uses_rand():
        out.add("rng")
    return out


def kernel_writes(k: Kernel) -> set:
    out = {s.buf.root for s in k.stores}
    if k.uses_rand():
        out.add("rng")
    return out


def stmt_reads(st) -> set:
    if isinstance(st, KernelStmt):
        return kernel_reads(st.kernel)
    out: set = set()
    if isinstance(st, ir.For):
        for b in (st.start, st.stop):
            if isinstance(b, ir.Buffer):
                out.add(b.root)
        out.add(st.counter.root)
        out |= block_reads(st.body)
    elif isinstance(st, ir.While):
        out |= block_reads(st.cond_block) | block_reads(st.body)
        out.add(st.cond.root)
    elif isinstance(st, ir.If):
        out.add(st.cond.root)
        out |= block_reads(st.then) | block_reads(st.orelse)
    elif isinstance(st, ir.Print):
        for it in st.items:
            if it.buf is not None:
                out.add(it.buf.root)
    elif isinstance(st, ir.Check):
        affine_reads(st.value, out)
    elif isinstance(st, ir.RTCall):
        out.update(b.root for b in rt_buffers(st, RT_READS))
        if st.name in ("shuffle", "skip_rand"):
            out.add("rng")
    return out


def stmt_writes(st) -> set:
    if isinstance(st, KernelStmt):
        return kernel_writes(st.kernel)
    out: set = set()
    if isinstance(st, ir.For):
        out.add(st.counter.root)
        out |= block_writes(st.body)
    elif isinstance(st, ir.While):
        out |= block_writes(st.cond_block) | block_writes(st.body)
    elif isinstance(st, ir.If):
        out |= block_writes(st.then) | block_writes(st.orelse)
    elif isinstance(st, ir.RTCall):
        out.update(b.root for b in rt_buffers(st, RT_WRITES))
        if st.name in ("shuffle", "seed", "skip_rand"):
            out.add("rng")
    return out


def block_reads(b: Block) -> set:
    out: set = set()
    for st in b.stmts:
        out |= stmt_reads(st)
    return out


def block_writes(b: Block) -> set:
    out: set = set()
    for st in b.stmts:
        out |= stmt_writes(st)
    return out


def count_writes(prog: ir.Program) -> dict:
    counts: dict = {}
    for b in ir.walk_blocks(prog.main):
        for st in b.stmts:
            if isinstance(st, (ir.For, ir.While, ir.If)):
                continue
            for w in stmt_writes(st):
                counts[w] = counts.get(w, 0) + 1
    return counts


def count_reads(prog: ir.Program, include_acc: bool = True) -> dict:
    """Number of statements reading each buffer, program-wide. With include_acc=False an
    accumulating store (`x += ...`) does not count as a use of x (for liveness)."""
    counts: dict = {}

    def visit(b: Block):
        for st in b.stmts:
            if isinstance(st, (ir.For, ir.While, ir.If)):
                own = set()
                if isinstance(st, ir.For):
                    own = {x.root for x in (st.start, st.stop) if isinstance(x, ir.Buffer)} | {st.counter.root}
                elif isinstance(st, ir.While):
                    own = {st.cond.root}
                else:
                    own = {st.cond.root}
                for r in own:
                    counts[r] = counts.get(r, 0) + 1
                for sub in ([st.body] if isinstance(st, ir.For) else
                            [st.cond_block, st.body] if isinstance(st, ir.While) else [st.then, st.orelse]):
                    visit(sub)
            else:
                rs = kernel_reads(st.kernel, include_acc) if isinstance(st, KernelStmt) else stmt_reads(st)
                for r in rs:
                    counts[r] = counts.get(r, 0) + 1
    visit(prog.main)
    return counts


# ----------------------------------------------------------------------------- expression utilities

def resimplify(e: Expr) -> Expr:
    def f(x):
        if isinstance(x, Unary):
            return mk_unary(x.op, x.a) if x.op not in ("f32", "i32") else cast(x.a, x.dtype)
        if isinstance(x, Binary):
            return mk_binary(x.op, x.a, x.b)
        if isinstance(x, Select):
            return mk_select(x.c, x.a, x.b)
        return x
    return map_expr(e, f)


def replace_loads(e: Expr, fn) -> Expr:
    """fn(Load) -> Expr | None (None keeps the load)."""
    def f(x):
        if isinstance(x, Load):
            r = fn(x)
            return x if r is None else r
        return x
    return map_expr(e, f)


def kernel_map_exprs(k: Kernel, fn):
    if k.red is not None:
        k.red.body = fn(k.red.body)
    for s in k.stmts:
        if isinstance(s, Let):
            s.value = fn(s.value)
        else:
            s.value = fn(s.value)


def rename_affine(a: Affine, old, new) -> Affine:
    terms = {}
    for t, c in a.terms.items():
        if isinstance(t, Gather):
            ld = t.load
            t = Gather(Load(new if ld.buf.root is old else ld.buf, rename_affine(ld.offset, old, new)), t.bound, t.where,
                       t.rest)
        elif isinstance(t, ScalarRef) and t.buf.root is old:
            t = ScalarRef(new)
        terms[t] = terms.get(t, 0) + c
    return Affine(a.const, terms)


def rename_in_expr(e: Expr, old, new) -> Expr:
    def f(x):
        if isinstance(x, Load):
            return Load(new if x.buf.root is old else x.buf, rename_affine(x.offset, old, new))
        if isinstance(x, Index):
            return Index(rename_affine(x.affine, old, new))
        return x
    return map_expr(e, f)


def gather_bufs(k: Kernel) -> set:
    """Buffers used as data-dependent indices anywhere in k."""
    out = set()

    def visit(a: Affine):
        for t in a.terms:
            if isinstance(t, Gather):
                out.add(t.load.buf.root)
                visit(t.load.offset)
            elif isinstance(t, ScalarRef):
                out.add(t.buf.root)
    for e in k.exprs():
        for x in ir.iter_expr(e):
            if isinstance(x, Load):
                visit(x.offset)
            elif isinstance(x, Index):
                visit(x.affine)
    for st in k.stores:
        visit(st.offset)
    return out


def expr_size(e: Expr) -> int:
    return sum(1 for _ in ir.iter_expr(e))


EXPENSIVE = {"exp", "log", "sqrt", "tanh", "sigmoid", "sin", "cos", "rsqrt", "recip"}


def is_cheap(e: Expr) -> bool:
    nodes = list(ir.iter_expr(e))
    loads = sum(1 for x in nodes if isinstance(x, Load))
    if any(isinstance(x, Unary) and x.op in EXPENSIVE for x in nodes):
        return False
    if any(isinstance(x, Binary) and x.op in ("div", "pow", "idiv", "mod") for x in nodes):
        return False
    return loads <= 1 and len(nodes) <= 4


def identity_of(buf: ir.Buffer, vars_) -> Affine:
    return Affine(0, {v: s for v, s in zip(vars_, row_major_strides(buf.shape))})


def split_offset(off: Affine, shape) -> list | None:
    """Exact per-dimension indices of a flat offset into a row-major buffer, or None."""
    if not shape:
        return [] if off.const == 0 and not off.terms else None
    dims = ir.decompose(off, shape)
    if dims is None:
        return None
    for d in range(1, len(shape)):
        a = dims[d]
        if a.scalars() or a.gathers():
            return None
        if any(v.extent is None for v in a.direct_vars()):
            return None
        lo, hi = a.var_bounds()
        if lo < 0 or hi > shape[d] - 1:
            return None
    # recombine check
    total = Affine(0)
    for a, s in zip(dims, row_major_strides(shape)):
        total = total + a * s
    if total != off:
        return None
    return dims


def forward_stores(stmts):
    """Within one kernel iteration: loads of a value stored earlier read the stored value,
    `+=` onto a known value becomes an add, and only the last store per location remains."""
    known = {}
    remap = {}
    out = []

    def fix(e):
        e = map_expr(e, lambda x: LetRef(remap[x.let]) if isinstance(x, LetRef) and x.let in remap else x)
        return replace_loads(e, lambda ld: known.get((ld.buf.root.id, ld.offset.key())))
    for s in stmts:
        if isinstance(s, Let):
            nl = Let(s.name, fix(s.value))
            remap[s] = nl
            out.append(nl)
            continue
        val = fix(s.value)
        key = (s.buf.root.id, s.offset.key())
        acc = s.accumulate
        if acc and key in known:
            val = mk_binary("add", known[key], val)
            acc = False
        if acc:
            known.pop(key, None)
            out.append(Store(s.buf, s.offset, val, True))
            continue
        if not isinstance(val, LetRef):
            nl = Let(s.buf.name, val)
            out.append(nl)
            val = LetRef(nl)
        known[key] = val
        out.append(Store(s.buf, s.offset, val, False))
    # a store followed by a plain store to the same location is dead
    final = []
    for i, s in enumerate(out):
        if isinstance(s, Store):
            key = (s.buf.root.id, s.offset.key())
            if any(isinstance(t, Store) and not t.accumulate and (t.buf.root.id, t.offset.key()) == key
                   for t in out[i + 1:]):
                continue
        final.append(s)
    return final


def cleanup_lets(stmts):
    """Drop `let a = b` where b is another let (alias it instead)."""
    alias = {}
    out = []

    def fix(e):
        return map_expr(e, lambda x: LetRef(alias[x.let]) if isinstance(x, LetRef) and x.let in alias else x)
    for s in stmts:
        if isinstance(s, Let):
            v = fix(s.value)
            if isinstance(v, LetRef):
                alias[s] = v.let
                continue
            nl = Let(s.name, v)
            alias_target = nl
            out.append(nl)
            alias[s] = alias_target
        else:
            out.append(Store(s.buf, s.offset, fix(s.value), s.accumulate))
    return out


def simple_store(k: Kernel):
    """The single plain identity store of k, or None."""
    if len(k.stmts) != 1 or not isinstance(k.stmts[0], Store):
        return None
    st = k.stmts[0]
    if st.accumulate or st.offset != identity_of(st.buf, k.domain) or len(k.domain) != len(st.buf.shape):
        return None
    if any(v.extent != n for v, n in zip(k.domain, st.buf.shape)):
        return None
    return st


# ----------------------------------------------------------------------------- the optimizer

class Optimizer:
    def __init__(self, prog: ir.Program):
        self.prog = prog
        self.changed = False

    def run(self):
        self.reads = count_reads(self.prog)
        self.reuse_dying_inputs(self.prog.main)
        for _ in range(8):
            self.changed = False
            self.reads = count_reads(self.prog)
            self.for_blocks(self.coalesce)
            self.reads = count_reads(self.prog)
            self.for_blocks(self.inline)
            self.for_blocks(self.simplify_block)
            self.for_blocks(self.cse)
            self.reads = count_reads(self.prog)
            self.for_blocks(self.fuse)
            self.for_blocks(self.cse)
            self.dce()
            if not self.changed:
                break
        for b in ir.walk_blocks(self.prog.main):
            for st in b.stmts:
                if isinstance(st, KernelStmt):
                    collapse_loops(st.kernel)

    def for_blocks(self, fn):
        self.changed = self.changed            # (clears the caches)
        for b in list(ir.walk_blocks(self.prog.main)):
            fn(b)

    # Read and write sets of statements are cached until the program changes: every change goes
    # through `self.changed = True`. (Without this, fusing in a block of n kernels takes n² set
    # computations, which matters for long unrolled `static for` loops.)
    @property
    def changed(self):
        return self._changed

    @changed.setter
    def changed(self, value):
        self._changed = value
        self._reads: dict[int, tuple] = {}
        self._writes: dict[int, tuple] = {}

    def sreads(self, st) -> set:
        hit = self._reads.get(id(st))
        if hit is None or hit[0] is not st:
            hit = (st, stmt_reads(st))
            self._reads[id(st)] = hit
        return hit[1]

    def swrites(self, st) -> set:
        hit = self._writes.get(id(st))
        if hit is None or hit[0] is not st:
            hit = (st, stmt_writes(st))
            self._writes[id(st)] = hit
        return hit[1]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def between(b: Block, i: int, j: int):
        return b.stmts[i + 1:j]

    def reads_in(self, stmts) -> set:
        out = set()
        for st in stmts:
            out |= self.sreads(st)
        return out

    def writes_in(self, stmts) -> set:
        out = set()
        for st in stmts:
            out |= self.swrites(st)
        return out

    def readers_in_block(self, b: Block, buf, start: int):
        """Indices (>= start) of statements in block b that read buf (incl. nested)."""
        return [i for i in range(start, len(b.stmts)) if buf in self.sreads(b.stmts[i])]

    # ---------------------------------------------------------------- simplify
    def simplify_block(self, b: Block):
        for st in b.stmts:
            if isinstance(st, KernelStmt):
                k = st.kernel
                before = [x.key() for x in k.exprs()]
                kernel_map_exprs(k, resimplify)
                if [x.key() for x in k.exprs()] != before:
                    self.changed = True

    # ---------------------------------------------------------------- coalescing
    def coalesce(self, b: Block):
        i = 0
        while i < len(b.stmts):
            st = b.stmts[i]
            if isinstance(st, KernelStmt) and self.try_coalesce(b, i):
                self.changed = True
                continue
            i += 1

    def try_coalesce(self, b: Block, ci: int) -> bool:
        copy = b.stmts[ci].kernel
        cst = simple_store(copy)
        if cst is None or copy.red is not None:
            return False
        canon = cst.buf.root
        if canon.kind in TEMP_KINDS or canon.kind == "const":
            return False
        v = cst.value
        if not isinstance(v, Load) or v.buf.root.kind not in TEMP_KINDS:
            return False
        tmp = v.buf.root
        if tmp.shape != canon.shape or tmp.dtype != canon.dtype or v.offset != identity_of(tmp, copy.domain):
            return False
        # find the producer of tmp in this block
        pi = None
        for j in range(ci - 1, -1, -1):
            s2 = b.stmts[j]
            if tmp in stmt_writes(s2):
                pi = j
                break
        if pi is None or not isinstance(b.stmts[pi], KernelStmt):
            return False
        prod = b.stmts[pi].kernel
        pst = simple_store(prod)
        if pst is None or pst.buf.root is not tmp:
            if not (len(prod.stores) >= 1 and all(s.buf.root is not tmp or s.offset == identity_of(tmp, prod.domain)
                                                   for s in prod.stores)):
                return False
            if not any(s.buf.root is tmp for s in prod.stores):
                return False
        mid = self.between(b, pi, ci)
        if canon in self.reads_in(mid) or canon in self.writes_in(mid):
            return False
        # tmp must not be read after the copy (or anywhere else outside [pi, ci])
        later = sum(1 for s in b.stmts[ci + 1:] if tmp in stmt_reads(s))
        total = self.reads.get(tmp, 0)
        inside = sum(1 for s in b.stmts[pi + 1:ci + 1] if tmp in stmt_reads(s))
        if later or total != inside:
            return False
        # in-place safety of the producer w.r.t. canon
        if not self.pointwise_ok(prod, canon):
            return False
        # retarget
        for s in prod.stores:
            if s.buf.root is tmp:
                s.buf = canon
        for s in mid:
            self.rename_reads(s, tmp, canon)
        del b.stmts[ci]
        return True

    @staticmethod
    def pointwise_ok(k: Kernel, buf) -> bool:
        """k may write `buf` in place if it reads buf only at its own (identity) index,
        and never inside its reduction."""
        ident = identity_of(buf, k.domain)
        if k.red is not None and any(ld.buf.root is buf for ld in ir.loads_in(k.red.body)):
            return False
        for st in k.stmts:
            for ld in ir.loads_in(st.value):
                if ld.buf.root is buf and ld.offset != ident:
                    return False
        return True

    def rename_reads(self, st, old, new):
        def fix(e):
            return rename_in_expr(e, old, new)
        if isinstance(st, KernelStmt):
            kernel_map_exprs(st.kernel, fix)
            for s in st.kernel.stores:
                s.offset = rename_affine(s.offset, old, new)
        elif isinstance(st, ir.Print):
            for it in st.items:
                if it.buf is not None and it.buf.root is old:
                    it.buf = new
        elif isinstance(st, ir.RTCall):
            for key in RT_READS.get(st.name, ()):
                v = st.args[key]
                if isinstance(v, list):
                    st.args[key] = [new if b.root is old else b for b in v]
                elif v.root is old:
                    st.args[key] = new
        elif isinstance(st, ir.If):
            if st.cond.root is old:
                st.cond = new
            for s in st.then.stmts + st.orelse.stmts:
                self.rename_reads(s, old, new)
        elif isinstance(st, ir.For):
            for s in st.body.stmts:
                self.rename_reads(s, old, new)
            if isinstance(st.stop, ir.Buffer) and st.stop.root is old:
                st.stop = new
            if isinstance(st.start, ir.Buffer) and st.start.root is old:
                st.start = new
        elif isinstance(st, ir.While):
            for s in st.cond_block.stmts + st.body.stmts:
                self.rename_reads(s, old, new)
            if st.cond.root is old:
                st.cond = new

    # ---------------------------------------------------------------- common kernels
    @staticmethod
    def kernel_signature(k: Kernel):
        st = simple_store(k)
        if st is None or st.buf.root.kind not in TEMP_KINDS or k.uses_rand():
            return None
        allv = k.all_vars()
        m = {v: Affine.of(CANON[i]) for i, v in enumerate(allv)}
        exprs = tuple(subst_expr(e, m).key() for e in k.exprs())
        red = (k.red.op, tuple(v.extent for v in k.red.vars)) if k.red else None
        return (tuple(v.extent for v in k.domain), red, st.buf.shape, st.buf.dtype, exprs)

    def cse(self, b: Block):
        """A kernel that recomputes exactly what an earlier one in the block computed (and
        whose inputs have not changed since) is replaced by the earlier result. Only results
        written exactly once qualify (a zero-filled gradient that is accumulated into later
        is not a value)."""
        writes = count_writes(self.prog)
        seen = {}
        i = 0
        while i < len(b.stmts):
            st = b.stmts[i]
            if isinstance(st, KernelStmt) and len(st.kernel.all_vars()) <= len(CANON):
                sig = self.kernel_signature(st.kernel)
                if sig is not None and writes.get(st.kernel.stores[0].buf.root, 0) != 1:
                    sig = None
                if sig is not None:
                    j = seen.get(sig)
                    if j is not None:
                        first = b.stmts[j].kernel
                        if not (kernel_reads(first) & self.writes_in(b.stmts[j + 1:i])):
                            old = st.kernel.stores[0].buf.root
                            new = first.stores[0].buf.root
                            for later in b.stmts[i + 1:]:
                                self.rename_reads(later, old, new)
                            del b.stmts[i]
                            self.changed = True
                            continue
                    seen[sig] = i
            i += 1

    # ---------------------------------------------------------------- in-place reuse
    def reuse_dying_inputs(self, b: Block):
        """At the top level: an elementwise kernel whose input is read by nothing else may
        overwrite that input instead of allocating a new buffer (e.g. `idx(...) / 255`)."""
        for i, st in enumerate(b.stmts):
            if not isinstance(st, KernelStmt) or st.kernel.red is not None:
                continue
            k = st.kernel
            pst = simple_store(k)
            if pst is None or pst.buf.root.kind not in TEMP_KINDS:
                continue
            tmp = pst.buf.root
            ident = identity_of(tmp, k.domain)
            for ld in [x for e in k.exprs() for x in ir.loads_in(e)]:
                src = ld.buf.root
                if src.kind not in ("data", "temp") or src is tmp or src.shape != tmp.shape:
                    continue
                if src.dtype != tmp.dtype or ld.offset != ident or self.reads.get(src, 0) != 1:
                    continue
                if not self.pointwise_ok(k, src):
                    continue
                pst.buf = src
                src.name = tmp.name
                for later in b.stmts[i + 1:]:
                    self.rename_reads(later, tmp, src)
                self.reads = count_reads(self.prog)       # src now has tmp's readers
                self.changed = True
                break

    # ---------------------------------------------------------------- inlining
    def inline(self, b: Block):
        i = 0
        while i < len(b.stmts):
            st = b.stmts[i]
            if isinstance(st, KernelStmt) and self.try_inline(b, i):
                self._changed = True          # (try_inline drops the cache entries it made stale)
                continue
            i += 1

    def try_inline(self, b: Block, pi: int) -> bool:
        p = b.stmts[pi].kernel
        if p.red is not None or p.uses_rand():
            return False
        pst = simple_store(p)
        if pst is None:
            return False
        if any(not injective(ld.offset, p.domain) for ld in ir.loads_in(pst.value)):
            return False      # it copies each input element to several places (im2col): keep it materialized
        tmp = pst.buf.root
        if tmp.kind not in TEMP_KINDS:
            return False
        readers = self.readers_in_block(b, tmp, pi + 1)
        total = self.reads.get(tmp, 0)
        if total == 0 or len(readers) != total:
            return False
        if any(not isinstance(b.stmts[r], KernelStmt) for r in readers):
            return False
        if any(s.buf.root is tmp for r in readers for s in b.stmts[r].kernel.stores):
            return False
        if not all(self.inline_order_safe(p, b.stmts[r].kernel, tmp) for r in readers):
            return False      # the inlined loads would observe the consumer's own writes
        if any(tmp in gather_bufs(b.stmts[r].kernel) for r in readers):
            return False      # used as an index: keep it materialized
        body = pst.value
        cheap = is_cheap(body)
        if not isinstance(body, Load) and any(self.recomputed_in_gemm(b.stmts[r].kernel, tmp) for r in readers):
            return False      # a matmul operand: computed once beats once per tile of the output
        if not cheap:
            if len(readers) != 1:
                return False
            c = b.stmts[readers[0]].kernel
            if self.duplication(c, tmp) > 1.0:
                return False
        # producer inputs must not change before the last reader
        last = readers[-1]
        p_in = kernel_reads(p)
        if p_in & self.writes_in(self.between(b, pi, last)):
            return False
        # every load of tmp must be splittable into per-dimension indices
        plan = []
        for r in readers:
            c = b.stmts[r].kernel
            for e in c.exprs():
                for ld in ir.loads_in(e):
                    if ld.buf.root is tmp and split_offset(ld.offset, tmp.shape) is None:
                        return False
            for s in c.stores:
                for g in s.offset.gathers():
                    if g.load.buf.root is tmp:
                        return False
            plan.append(c)
        dom = p.domain

        def sub(ld: Load):
            if ld.buf.root is not tmp:
                return None
            dims = split_offset(ld.offset, tmp.shape)
            return subst_expr(body, {v: a for v, a in zip(dom, dims)})
        for c in plan:
            kernel_map_exprs(c, lambda e: replace_loads(e, sub))
        for r in readers:
            self._reads.pop(id(b.stmts[r]), None)      # they read p's inputs now, not tmp
        del b.stmts[pi]
        return True

    @staticmethod
    def inline_order_safe(p: Kernel, c: Kernel, tmp) -> bool:
        """Substituting p's body into c is safe unless c writes something p reads. The
        exception is the pointwise update pattern (`m = 0.9*m + g`): p reads B at the index
        c stores it, and every use of tmp in c comes before that store."""
        conflict = kernel_reads(p) & kernel_writes(c)
        if not conflict:
            return True
        if any(not isinstance(x, ir.Buffer) for x in conflict):
            return False
        ident_tmp_c = identity_of(tmp, c.domain)
        for e in c.exprs():
            for ld in ir.loads_in(e):
                if ld.buf.root is tmp and ld.offset != ident_tmp_c:
                    return False
        for B in conflict:
            if B.shape != tmp.shape:
                return False
            for e in p.exprs():
                for ld in ir.loads_in(e):
                    if ld.buf.root is B and ld.offset != identity_of(B, p.domain):
                        return False
            idx = [i for i, st in enumerate(c.stmts) if isinstance(st, Store) and st.buf.root is B]
            if len(idx) != 1 or c.stmts[idx[0]].offset != identity_of(B, c.domain):
                return False
            first_store = idx[0]
            for i, st in enumerate(c.stmts):
                if i >= first_store and any(ld.buf.root is tmp for ld in ir.loads_in(st.value)):
                    return False
        return True

    def recomputed_in_gemm(self, c: Kernel, tmp) -> bool:
        """c is a matrix multiply that reads tmp in its reduction and would recompute each element
        of tmp many times (e.g. `relu(z)` as the left operand of a weight gradient: once per 16
        columns of the output). Materializing tmp costs one pass over it; the matmul re-reads it
        anyway."""
        if not gemm_like(c) or not any(ld.buf.root is tmp for ld in ir.loads_in(c.red.body)):
            return False
        return self.duplication(c, tmp) > GEMM_REUSE

    def duplication(self, c: Kernel, tmp) -> float:
        """How many times each element of tmp would be computed if inlined into c."""
        n = 0
        dom = ir.prod(v.extent for v in c.domain)
        red = ir.prod(v.extent for v in c.red.vars) if c.red else 1
        if c.red is not None:
            n += sum(1 for ld in ir.loads_in(c.red.body) if ld.buf.root is tmp) * dom * red
        for s in c.stmts:
            n += sum(1 for ld in ir.loads_in(s.value) if ld.buf.root is tmp) * dom
        return n / max(1, tmp.numel)

    # ---------------------------------------------------------------- vertical fusion
    def fuse(self, b: Block):
        i = 0
        while i < len(b.stmts):
            st = b.stmts[i]
            if isinstance(st, KernelStmt) and self.try_fuse(b, i):
                self._changed = True          # (the fused kernel is a new statement: not cached yet)
                continue
            i += 1

    def try_fuse(self, b: Block, pi: int) -> bool:
        p = b.stmts[pi].kernel
        # candidate outputs: identity stores of temps
        for pst in p.stores:
            if pst.accumulate:
                continue
            tmp = pst.buf.root
            if tmp.kind in ("const", "data"):
                continue
            if pst.offset != identity_of(tmp, p.domain) or len(p.domain) != len(tmp.shape):
                continue
            if sum(1 for s2 in p.stores if s2.buf.root is tmp) != 1:
                continue
            readers = self.readers_in_block(b, tmp, pi + 1)
            if not readers:
                continue
            ci = readers[0]
            cst = b.stmts[ci]
            if not isinstance(cst, KernelStmt):
                continue
            c = cst.kernel
            if c.red is not None or c is p:
                continue
            if tmp in gather_bufs(c):
                continue
            if [v.extent for v in c.domain] != [v.extent for v in p.domain]:
                continue
            ident_c = identity_of(tmp, c.domain)
            ok = True
            for e in c.exprs():
                for ld in ir.loads_in(e):
                    if ld.buf.root is tmp and ld.offset != ident_c:
                        ok = False
            for s in c.stores:
                if s.accumulate and s.offset != identity_of(s.buf, c.domain):
                    ok = False
                if s.offset.gathers():
                    ok = False
                if s.buf.root is tmp and s.offset != ident_c:
                    ok = False
            if not ok:
                continue
            if c.uses_rand() and p.uses_rand():
                continue
            epilogue = sum(1 for st in p.stmts for _ in ir.iter_expr(st.value)) + \
                sum(1 for e in c.exprs() for _ in ir.iter_expr(e))
            if gemm_like(p) and epilogue > GEMM_EPILOGUE:
                continue      # a big epilogue (Adam) would cost p its register tile: keep it separate
            mid = self.between(b, pi, ci)
            if tmp in self.writes_in(mid):
                continue      # c reads what those writes left in tmp, not p's values (`x[k] = v` in a loop)
            others = self.reads.get(tmp, 0) - (1 if tmp in kernel_reads(c) else 0)
            keep_tmp = others > 0 or tmp.kind not in TEMP_KINDS
            # choose where the fused kernel goes
            sink_ok = not (kernel_writes(p) & self.reads_in(mid)) and not (kernel_reads(p) & self.writes_in(mid)) \
                and not (kernel_writes(p) & self.writes_in(mid))
            hoist_ok = not (kernel_reads(c) - {tmp}) & self.writes_in(mid) and \
                not (kernel_writes(c) & (self.reads_in(mid) | self.writes_in(mid)))
            if not (sink_ok or hoist_ok):
                continue
            merged = self.merge(p, pst, c, keep_tmp)
            if merged is None:
                continue
            if sink_ok:
                b.stmts[ci] = KernelStmt(merged)
                del b.stmts[pi]
            else:
                b.stmts[pi] = KernelStmt(merged)
                del b.stmts[ci]
            return True
        return False

    def merge(self, p: Kernel, pst: Store, c: Kernel, keep_tmp: bool) -> Kernel | None:
        """Run c's statements inside p's store loop (after p's own statements)."""
        tmp = pst.buf.root
        m = {cv: Affine.of(pv) for cv, pv in zip(c.domain, p.domain)}
        stmts = list(p.stmts)
        remap = {}
        for s in c.stmts:
            val = subst_expr(s.value, m)
            val = map_expr(val, lambda x: LetRef(remap[x.let]) if isinstance(x, LetRef) and x.let in remap else x)
            if isinstance(s, Let):
                nl = Let(s.name, val)
                remap[s] = nl
                stmts.append(nl)
            else:
                stmts.append(Store(s.buf, s.offset.subst(m), val, s.accumulate))
        stmts = forward_stores(stmts)
        if not keep_tmp:
            stmts = [s for s in stmts if not (isinstance(s, Store) and s.buf.root is tmp)]
        stmts = cleanup_lets(stmts)
        k = Kernel(domain=p.domain, stmts=stmts, red=p.red, name=p.name,
                   label=self.join_labels(p.label, c.label), span=p.span or c.span)
        # keep fused kernels small enough to live in registers
        nodes = sum(1 for e in k.exprs() for _ in ir.iter_expr(e))
        accesses = {(ld.buf.root.id, ld.offset.key()) for e in k.exprs() for ld in ir.loads_in(e)}
        if nodes > MAX_FUSED_NODES or len(k.stores) > MAX_FUSED_STORES or len(accesses) > MAX_FUSED_LOADS:
            return None
        # in-place safety: a written buffer may only be read at its store offset, and never
        # inside the reduction
        for st in k.stores:
            if k.red is not None and any(ld.buf.root is st.buf.root for ld in ir.loads_in(k.red.body)):
                return None
            for e in k.exprs():
                for ld in ir.loads_in(e):
                    if ld.buf.root is st.buf.root and ld.offset != st.offset:
                        return None
        return k

    @staticmethod
    def join_labels(a, b):
        if not a:
            return b
        if not b or b == a or b in a.split("+"):
            return a
        return f"{a}+{b}"

    # ---------------------------------------------------------------- dead code
    def dce(self):
        while True:
            reads = count_reads(self.prog, include_acc=False)
            removed = False
            for b in ir.walk_blocks(self.prog.main):
                keep = []
                for st in b.stmts:
                    if isinstance(st, KernelStmt):
                        k = st.kernel
                        drew = k.uses_rand()
                        live = [s for s in k.stmts if not isinstance(s, Store) or self.store_live(s, reads)]
                        if len(live) != len(k.stmts):
                            removed = True
                            k.stmts = live
                        if not k.stores:
                            removed = True
                            if drew:      # later random numbers must not change: still use up its stream
                                keep.append(ir.RTCall("skip_rand", {}, span=k.span))
                            continue
                        # drop unused lets
                        used = set()
                        for s in k.stmts:
                            for x in ir.iter_expr(s.value):
                                if isinstance(x, LetRef):
                                    used.add(x.let)
                        k.stmts = [s for s in k.stmts if not isinstance(s, Let) or s in used]
                        if drew and not k.uses_rand():
                            keep.append(st)
                            keep.append(ir.RTCall("skip_rand", {}, span=k.span))
                            continue
                    keep.append(st)
                b.stmts = keep
            if not removed:
                break
            self.changed = True

    @staticmethod
    def store_live(s: Store, reads) -> bool:
        b = s.buf.root
        if b.kind in ("param", "data", "output"):          # outputs: the caller's arrays (anvil.function)
            return True
        return reads.get(b, 0) > 0


def optimize_program(prog: ir.Program):
    Optimizer(prog).run()
    return prog


# ----------------------------------------------------------------------------- loop collapsing

def injective(off: Affine, vars_) -> bool:
    """Different values of the vars that appear in off reach different offsets."""
    items = sorted(((abs(off.coef(v)), v.extent) for v in vars_ if off.coef(v) != 0 and v.extent > 1))
    reach = 0
    for c, n in items:
        if c <= reach:
            return False
        reach += c * (n - 1)
    return True


def collapse_loops(k: Kernel):
    """Merge adjacent loops a, b (a outer) into one loop of extent n_a·n_b wherever every access
    has coef(a) == coef(b)·n_b, so that they only ever see a·n_b + b: `dW[o, c, u, v]` with every
    access contiguous in (c, u, v) becomes `dW[o, q]` with q < c·u·v. The iteration order, and so
    every result, stays the same; the vector loop just gets longer."""
    if k.uses_rand():
        return
    for _ in range(16):
        for vars_ in ([k.domain] + ([k.red.vars] if k.red is not None else [])):
            for x in range(len(vars_) - 1):
                a, b = vars_[x], vars_[x + 1]
                if _collapsible(k, a, b):
                    w = ir.Var(f"{a.name}{b.name}", a.extent * b.extent)
                    _collapse(k, a, b, w)
                    vars_[x:x + 2] = [w]
                    break
            else:
                continue
            break
        else:
            return


def _affines(k: Kernel):
    """Every affine offset in k, including those inside gathers."""
    def of(aff):
        yield aff
        for g in aff.gathers():
            yield from of(g.load.offset)
            if g.rest is not None:
                yield g.rest
    for e in k.exprs():
        for x in ir.iter_expr(e):
            if isinstance(x, Load):
                yield from of(x.offset)
    for s in k.stores:
        yield from of(s.offset)


def _collapsible(k: Kernel, a, b) -> bool:
    for e in k.exprs():
        for x in ir.iter_expr(e):
            if isinstance(x, Index) and (a in x.affine.vars() or b in x.affine.vars()):
                return False          # the loop index is used as a number
    return all(aff.coef(a) == aff.coef(b) * b.extent for aff in _affines(k))


def _collapse(k: Kernel, a, b, w):
    def remap(aff: Affine) -> Affine:
        terms = {}
        for t, c in aff.terms.items():
            if t is a or t is b:
                continue
            if isinstance(t, Gather):
                t = Gather(Load(t.load.buf, remap(t.load.offset)), t.bound, t.where,
                           remap(t.rest) if t.rest is not None else None)
            terms[t] = terms.get(t, 0) + c
        if aff.coef(b):
            terms[w] = aff.coef(b)
        return Affine(aff.const, terms)

    def fix(e):
        return Load(e.buf, remap(e.offset)) if isinstance(e, Load) else e
    kernel_map_exprs(k, lambda e: map_expr(e, fix))
    for s in k.stores:
        s.offset = remap(s.offset)


def gemm_like(k: Kernel) -> bool:
    """A sum over a small body with at least 4 rows of 16-wide outputs: the backend gives it a
    4×16 register tile, if the registers are not needed for anything else."""
    if k.red is None or k.red.op != "sum" or len(k.domain) < 2:
        return False
    ext = collapsed_extents(k, k.domain)
    return len(ext) >= 2 and ext[-2] >= 4 and ext[-1] >= 16 and sum(1 for _ in ir.iter_expr(k.red.body)) <= 24


def collapsed_extents(k: Kernel, vars_) -> list[int]:
    """The loop extents collapse_loops would leave of vars_ (without changing k)."""
    if any(isinstance(x, Index) and set(x.affine.vars()) & set(vars_) for e in k.exprs() for x in ir.iter_expr(e)):
        return [v.extent for v in vars_]
    affs = list(_affines(k))
    groups = [(v, v.extent) for v in vars_]          # (innermost var of the group, extent)
    x = 0
    while x < len(groups) - 1:
        (a, ea), (b, eb) = groups[x], groups[x + 1]
        if all(aff.coef(a) == aff.coef(b) * eb for aff in affs):
            groups[x:x + 2] = [(b, ea * eb)]
        else:
            x += 1
    return [e for _, e in groups]
