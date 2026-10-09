"""Elaboration: AST -> kernel IR.

The elaborator runs the program at compile time over symbolic tensors. Compile-time
values are folded, functions/models are inlined, shapes are checked, every tensor
operation becomes a kernel, and runtime control flow is kept as structured IR.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import ast as A
from . import dims
from . import ir
from .builtins import BUILTIN_CONSTS, BUILTINS, BuiltinsMixin
from .diagnostics import MAX_ERRORS, AnvilError, AnvilErrors, Label, PoisonError, fmt_shape, suggest

MAX_CALL_LABELS = 3          # calls shown for an error inside inlined functions
from .ir import (F32, I32, Affine, Binary, Block, Buffer, Const, Expr, Gather, Index, Kernel, KernelStmt,
                 Load, ScalarRef, Var, row_major_strides)
from .ops import BINOP, OpsMixin, describe, identity_offset
from .parser import REDUCTIONS
from .simplify import cast, mk_binary
from .source import SourceFile, Span
from .values import (AffVal, BatchesVal, Binding, BuiltinVal, CVal, EVal, FStrVal, FnVal, IdxVal, MethodVal,
                     ModelDefVal, ModelInstVal, NoneVal, OptDefVal, OptSpecVal, PickVal, PoisonVal, RangeVal, SVal,
                     Scope, TVal, TupleVal, Val)



def is_model_list(v) -> bool:
    return isinstance(v, TupleVal) and bool(v.items) and all(isinstance(m, ModelInstVal) for m in v.items)


def set_extent(v, n):
    """An index variable's range: the number for code generation, and its name if it has one."""
    v.extent = int(n)
    v.dim = n if isinstance(n, dims.Dim) else None

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


# ----------------------------------------------------------------------------- index notation context

class PendingLoad(Expr):
    """A load from a reduction temp whose shape is not known until range inference."""

    def __init__(self, temp: "PendingTemp", idx: list[Affine]):
        self.temp = temp
        self.idx = idx
        self.dtype = temp.dtype

    def key(self):
        return ("pend", id(self.temp), tuple(a.key() for a in self.idx))


@dataclass
class PendingTemp:
    domain: list[Var]
    red_vars: list[Var]
    op: str
    body: Expr
    dtype: str
    span: object
    label: str
    buf: Buffer | None = None


@dataclass
class CompCtx:
    names: dict                     # index name -> Var
    binders: dict                   # id(reduction ast) -> [names]
    lhs: list = field(default_factory=list)          # LHS Vars (incl. ellipsis group)
    ellipsis: list | None = None
    ellipsis_is_index: bool = False
    constraints: list = field(default_factory=list)  # (affine, size, span, desc)
    pending: list = field(default_factory=list)      # PendingTemp
    bound: list = field(default_factory=list)        # reduction Vars currently bound, in binding order
    where: dict = field(default_factory=dict)        # Var -> extent
    checks: list = field(default_factory=list)
    lhs_items: list = field(default_factory=list)    # AST items of the left-hand side
    ellipsis_on_lhs: bool = False
    ellipsis_reducer: object = None


@dataclass
class LoopCtx:
    scope: Scope
    carried: dict                    # name -> canonical Buffer
    new_names: set
    counter_names: set


def _collect_free_index_vars(e: Expr) -> set:
    out = set()
    for x in ir.iter_expr(e):
        if isinstance(x, Load):
            out.update(x.offset.vars())
        elif isinstance(x, Index):
            out.update(x.affine.vars())
        elif isinstance(x, PendingLoad):
            for a in x.idx:
                out.update(a.vars())
    return out


def compute_runtime_names(stmts) -> set:
    """Names that must be runtime variables: assigned more than once, or inside a block."""
    counts: dict[str, int] = {}
    nested: set[str] = set()

    def visit(ss, inner):
        for s in ss:
            names = []
            if isinstance(s, A.Assign):
                names = [t.id for t in s.targets]
            elif isinstance(s, (A.AnnAssign, A.AugAssign)):
                names = [s.target.id]
                if isinstance(s, A.AugAssign):
                    counts[s.target.id] = counts.get(s.target.id, 0) + 1
            elif isinstance(s, A.IndexAssign):
                names = [s.target.id]
            elif isinstance(s, A.If):
                visit(s.body, True)
                visit(s.orelse, True)
            elif isinstance(s, A.For):
                for t in s.targets:
                    nested.add(t.id)
                visit(s.body, True)
            elif isinstance(s, A.While):
                visit(s.body, True)
            for n in names:
                counts[n] = counts.get(n, 0) + 1
                if inner:
                    nested.add(n)
    visit(stmts, False)
    return {n for n, c in counts.items() if c > 1} | nested


# ----------------------------------------------------------------------------- elaborator

class Elaborator(OpsMixin, BuiltinsMixin):
    def __init__(self, source=None, seed: int = 0, consts: dict | None = None):
        self.source = source
        self.const_overrides = dict(consts or {})     # `--set NAME=VALUE` from the command line
        self.dim_names: dict = {}                     # formula -> the constant that names it (dims.py)
        dims.DIM_NAMES.set(self.dim_names)
        self.buffers: list[Buffer] = []
        self.blocks: list[Block] = [Block()]
        self.prelude_scope = Scope(kind="global")
        self.globals = Scope(parent=self.prelude_scope, kind="global")
        self.scope = self.globals
        self.comps: list[CompCtx | None] = []
        self.loops: list[LoopCtx] = []
        self.fn_stack: list[str] = []
        self.span_stack: list = []
        self.recorder = None          # anvil.ide.Recorder: what each name holds, for editors
        self.trained_params: set = set()   # parameters some `minimize` updates
        self.used_files: set = set()  # `use "file.anvil"`: files already brought in
        self.using: set = set()       # … and those being brought in (a cycle is an error)
        self.runtime_depth = 0           # nesting of runtime blocks inside the current function
        self.fn_runtime_base: list[int] = []
        self.minimize_sites = 0
        self.rand_salt = 0
        self.temp_counter = 0
        self.naming: list[str] = []      # instance-name prefixes during model instantiation
        self.ad_region_start: list[int] = []
        self.seed = seed
        self.warnings: list[AnvilError] = []
        self.idx_cache: dict = {}
        self.instances: list[ModelInstVal] = []
        self.scope_stmts: dict[int, list] = {}      # id(scope) -> its statement list (for liveness)
        self.static_marks: list[int] = []           # len(self.loops) at each enclosing `static for`
        self.errors: list[AnvilError] = []            # reported together at the end

    # ------------------------------------------------------------------ infrastructure
    @property
    def block(self) -> Block:
        return self.blocks[-1]

    def emit(self, stmt):
        self.block.stmts.append(stmt)

    def emit_kernel(self, k: Kernel):
        if not k.name:
            k.name = f"k{k.id}"
        self.block.stmts.append(KernelStmt(k))

    def new_buffer(self, name, shape, dtype=F32, kind="temp", init=None, span=None) -> Buffer:
        b = Buffer(name, shape, dtype, kind, init, span)
        self.buffers.append(b)
        return b

    def new_temp(self, shape, dtype=F32, name=None) -> Buffer:
        if name is None:
            self.temp_counter += 1
            name = f"t{self.temp_counter}"
        return self.new_buffer(name, shape, dtype, "temp")

    def current_label(self) -> str:
        return self.fn_stack[-1] if self.fn_stack else ""

    def current_span(self):
        """The innermost statement in the user's file (not the prelude) being elaborated."""
        for sp in reversed(self.span_stack):
            if self.source is None or sp.file is self.source:
                return sp
        return self.span_stack[0] if self.span_stack else None

    def last_kernel(self):
        if self.block.stmts and isinstance(self.block.stmts[-1], KernelStmt):
            return self.block.stmts[-1].kernel
        return None

    @property
    def comp(self) -> CompCtx | None:
        return self.comps[-1] if self.comps else None

    # ------------------------------------------------------------------ entry
    def run(self, program: A.Program, prelude: A.Program | None = None) -> ir.Program:
        if prelude is not None:
            # the standard library lives in its own scope: it cannot see user globals
            self.prelude_scope.runtime_names |= compute_runtime_names(prelude.body)
            self.scope = self.prelude_scope
            for s in prelude.body:
                self.exec_stmt(s)
            self.scope = self.globals
        self.globals.runtime_names |= compute_runtime_names(program.body)
        self.scope_stmts[id(self.globals)] = program.body
        if self.source is not None:
            self.using.add(os.path.normpath(os.path.abspath(self.source.path)))
        self.check_overrides(program)
        self.exec_block(program.body)
        if self.errors:
            raise self.errors[0] if len(self.errors) == 1 else AnvilErrors(self.errors)
        self.warn_untrained(program)
        return ir.Program(self.blocks[0], self.buffers, self.source)

    def warn_untrained(self, program: A.Program):
        """A `param` of the program that no `minimize` trains (when something is trained) is
        usually a mistake: it keeps its initial values."""
        if not self.minimize_sites:
            return
        for s in program.body:
            if isinstance(s, A.ParamDecl):
                b = self.globals.vars.get(s.name.id)
                buf = b.ref if b is not None else None
                if buf is not None and buf.root not in self.trained_params:
                    self.warnings.append(AnvilError(
                        f"`{s.name.id}` is a parameter, but no `minimize` trains it", s.name.span, kind="warning",
                        help="it keeps its initial values; declare it without `param` if that is intended"))

    def check_overrides(self, program: A.Program):
        """Every `--set NAME=…` must name a `const` declared at the top of the program."""
        declared = [s.name.id for s in program.body if isinstance(s, A.ConstDecl)]
        for name in self.const_overrides:
            if name not in declared:
                where = os.path.basename(self.source.path) if self.source is not None else "the program"
                s = suggest(name, declared)
                listing = ", ".join(declared) if declared else "none"
                raise AnvilError(f"`--set {name}=…`: {where} has no constant `{name}`", None,
                               help=f"did you mean `{s}`?" if s else f"its constants: {listing}")

    def exec_block(self, stmts):
        for s in stmts:
            if self.fn_stack:                 # in an inlined function the first error ends the call
                self.exec_stmt(s)
                continue
            try:
                self.exec_stmt(s)
            except PoisonError:
                self.poison(s)
            except AnvilError as e:
                self.errors.extend(e.errors if isinstance(e, AnvilErrors) else [e])
                self.poison(s)
                if len(self.errors) >= MAX_ERRORS:
                    raise AnvilErrors(self.errors)

    def poison(self, s: A.Stmt):
        """Bind what s would have defined to an error value: later uses of it stay quiet."""
        names = A.walk_assigned_names([s])
        if isinstance(s, (A.FnDecl, A.ModelDecl, A.OptimizerDecl, A.ParamDecl)):
            names.append(s.name.id)
        for name in names:
            self.scope.vars[name] = Binding(PoisonVal(), what="error")

    def exec_stmt(self, s: A.Stmt):
        self.span_stack.append(s.span)
        try:
            meth = getattr(self, "st_" + type(s).__name__)
            meth(s)
            if self.recorder is not None:
                self.note_definitions(s)
        finally:
            self.span_stack.pop()

    def note_definitions(self, s: A.Stmt):
        """For editors (`anvil check --json`): what each name that s defines holds now."""
        targets = []
        if isinstance(s, A.Assign):
            targets = s.targets
        elif isinstance(s, (A.AnnAssign, A.AugAssign, A.IndexAssign)):
            targets = [s.target]
        elif isinstance(s, (A.ConstDecl, A.ParamDecl, A.FnDecl, A.ModelDecl)):
            targets = [s.name]
        for t in targets:
            b = self.scope.lookup(t.id)
            if b is not None:
                self.recorder.note(t.span, b.val, definition=True, annotated=isinstance(s, (A.AnnAssign, A.ParamDecl)))

    # ------------------------------------------------------------------ declarations
    def st_Use(self, s: A.Use):
        """`use "model.anvil"`: run another file's declarations here (once per file)."""
        if self.scope is not self.globals or self.loops or self.fn_stack:
            raise AnvilError("`use` belongs at the top level of a program", s.span)
        base = s.span.file.dir if s.span is not None else os.getcwd()
        path = os.path.normpath(os.path.abspath(os.path.join(base, s.path)))
        if path in self.used_files:
            return                                     # already here
        if path in self.using:
            raise AnvilError(f"`{s.path}` uses itself (through the files it uses)", s.span)
        try:
            src = SourceFile.read(path)
        except OSError:
            raise AnvilError(f"cannot read `{s.path}`", s.span, help=f"looked for {path}")
        from .parser import parse
        tree = parse(src)
        allowed = (A.FnDecl, A.ModelDecl, A.OptimizerDecl, A.ConstDecl, A.Use)
        for st in tree.body:
            if not isinstance(st, allowed):
                raise AnvilError("a file brought in with `use` may only declare functions, models, optimizers "
                               "and constants", st.span, labels=[Label(s.span, "used here")])
        self.using.add(path)
        try:
            for st in tree.body:
                self.exec_stmt(st)
        finally:
            self.using.discard(path)
        self.used_files.add(path)

    def st_ConstDecl(self, s: A.ConstDecl):
        name = s.name.id
        if self.scope is self.globals and name in self.const_overrides:
            v = self.const_overrides[name]
            self.scope.vars[name] = Binding(SVal(v) if isinstance(v, str) else CVal(dims.named(v, name)),
                                            what="const", span=s.span)
            return
        val = self.eval_top(s.value)
        if isinstance(val, CVal) and val.is_int and not isinstance(val.value, bool):
            # every integer constant names a dimension (dims.py); a derived one (`const DH = D // HEADS`)
            # keeps its formula and is shown by its own name
            if isinstance(val.value, dims.Dim) and val.value.sym not in self.dim_names:
                self.dim_names[val.value.sym] = name
            val = CVal(dims.named(val.value, name))
        if not isinstance(val, (CVal, SVal, TupleVal)):
            raise AnvilError(f"`const {name}` must be known at compile time, but this is a {val.kind}", s.value.span,
                           help="drop `const` to make it a runtime variable")
        if self.scope.vars.get(name) is not None and self.scope.vars[name].what == "const":
            raise AnvilError(f"`{name}` is already defined as a constant", s.name.span)
        self.scope.vars[name] = Binding(val, what="const", span=s.span)

    def st_FnDecl(self, s: A.FnDecl):
        self.scope.vars[s.name.id] = Binding(FnVal(s, self.scope, s.name.id), what="function", span=s.span)

    def st_ModelDecl(self, s: A.ModelDecl):
        self.scope.vars[s.name.id] = Binding(ModelDefVal(s, self.scope), what="model", span=s.span)

    def st_OptimizerDecl(self, s: A.OptimizerDecl):
        self.scope.vars[s.name.id] = Binding(OptDefVal(s, self.scope), what="optimizer", span=s.span)

    def st_ParamDecl(self, s: A.ParamDecl):
        if self.scope.kind not in ("global", "model") or self.loops or self.fn_stack:
            raise AnvilError("parameters can only be declared at the top level or inside a model", s.span)
        dtype, shape = self.eval_type(s.type, self.scope)
        name = self.scope.prefix + s.name.id
        buf = self.new_buffer(name, shape, dtype, "param", span=s.span)
        if s.sample is not None:
            dist = self.eval_top(s.sample)
            self.sample_into(buf, dist, s.sample.span)
        elif s.init is not None:
            val = self.eval_top(s.init)
            if not (isinstance(val, CVal) and val.value == 0):
                self.write_buffer(buf, val, s.init.span, fresh=True)
        self.scope.vars[s.name.id] = Binding(TVal.of(buf), ref=buf, what="param", span=s.span)

    # ------------------------------------------------------------------ assignments
    def st_Assign(self, s: A.Assign):
        if len(s.targets) == 1:
            tgt = s.targets[0]
            # model instantiation gets the variable's name
            if isinstance(s.value, A.Call):
                f = self.try_eval_callee(s.value.func)
                if isinstance(f, ModelDefVal):
                    inst = self.instantiate(f, s.value, self.scope.prefix + tgt.id)
                    self.scope.vars[tgt.id] = Binding(inst, what="model instance", span=s.span)
                    return
            if isinstance(s.value, A.ListComp):              # layers = [Layer(i) for i in range(n)]
                val = self.list_comp(s.value, model_list=tgt.id)
                if val.items and all(isinstance(x, ModelInstVal) for x in val.items):
                    self.scope.vars[tgt.id] = Binding(val, what="model instance", span=s.span)
                    return
                self.assign(tgt.id, val, s.value.span, fresh=True)
                return
            nk = len(self.block.stmts)
            val = self.eval_top(s.value)
            self.assign(tgt.id, val, s.value.span, fresh=self.fresh_since(nk, s.value))
            return
        val = self.eval_top(s.value)
        if not isinstance(val, TupleVal):
            raise AnvilError(f"cannot unpack a {val.kind} into {len(s.targets)} names", s.value.span)
        if len(val.items) != len(s.targets):
            raise AnvilError(f"expected {len(s.targets)} values, got {len(val.items)}", s.value.span)
        for t, v in zip(s.targets, val.items):
            self.assign(t.id, v, s.value.span)

    def fresh_since(self, nk: int, node) -> bool:
        return len(self.block.stmts) > nk and not isinstance(node, A.Name)

    def st_AnnAssign(self, s: A.AnnAssign):
        dtype, shape = self.eval_type(s.type, self.scope)
        if s.sample is not None:
            dist = self.eval_top(s.sample)
            buf = self.new_temp(shape, dtype, s.target.id)
            self.sample_into(buf, dist, s.sample.span)
            self.assign(s.target.id, TVal.of(buf), s.span, fresh=True)
            return
        nk = len(self.block.stmts)
        val = self.eval_top(s.value)
        val = self.coerce(val, dtype, shape, s.value.span, f"`{s.target.id}`")
        self.assign(s.target.id, val, s.value.span, fresh=self.fresh_since(nk, s.value))

    def coerce(self, val: Val, dtype, shape, span, what="value") -> Val:
        """Check/convert a value against a declared type."""
        last = getattr(self, "last_loaded", None)
        if isinstance(val, TVal) and last is not None and val.buf is last and val.is_identity() \
                and val.dtype != dtype:
            last.dtype = dtype          # the loader converts straight to the declared type
        if isinstance(val, (CVal, AffVal)) and shape == ():
            val = self.to_tensor(val, span, dtype)
        elif isinstance(val, (CVal, AffVal)) or (isinstance(val, TVal) and val.rank == 0 and shape != ()):
            e = cast(self.scalar_expr(val, span), dtype)
            val = self.emit_map(tuple(shape), lambda vs: e, span=span, label="fill")
        if isinstance(val, EVal):
            val = self.to_tensor(val, span, dtype)
        if not isinstance(val, TVal):
            raise AnvilError(f"{what} is declared as {dtype}{fmt_shape(shape)} but the value is a {val.kind}", span)
        if val.shape != tuple(shape):
            if val.numel == ir.prod(shape) and val.rank != len(shape):
                val = self.reshape(val, shape, span)
            else:
                raise AnvilError(f"{what} is declared as {dtype}{fmt_shape(shape)} but the value has shape "
                               f"{fmt_shape(val.shape)}", span)
        bad = dims.first_conflict(tuple(shape), val.shape)
        if bad:
            raise AnvilError(f"{what} is declared as {dtype}{dims.show_shape(shape)} but the value has shape "
                           f"{dims.show_shape(val.shape)}", span, notes=[dims.mismatch_note(bad[1], bad[2])],
                           help=dims.same_size_help(bad[1], bad[2]))
        if dims.has_names(shape) and isinstance(val, TVal):  # the declaration names what had no name
            val = TVal(val.buf, dims.adopt(tuple(shape), val.shape), val.tmpl, val.pvars, val.detached)
        if val.dtype != dtype:
            val = self.to_tensor(val, span, dtype)
        return val

    def st_AugAssign(self, s: A.AugAssign):
        name = s.target.id
        b = self.scope.lookup(name)
        if b is None:
            raise self.undefined(name, s.target.span)
        nk = len(self.block.stmts)
        rhs = self.eval_top(s.value)
        if s.op == "@":
            val = self.matmul(b.val, rhs, s.span)
        else:
            val = self.binary_op(BINOP[s.op], b.val, rhs, s.span)
        self.assign(name, val, s.span, fresh=len(self.block.stmts) > nk)

    def assign(self, name: str, val: Val, span, fresh: bool = False):
        b = self.scope.lookup(name)
        if b is not None and b.ref is not None:
            self.write_buffer(b.ref, val, span, fresh=fresh)
            return
        if b is not None and b.what == "const" and name in self.scope.vars:
            raise AnvilError(f"cannot assign to `{name}`: it is a constant", span)
        for lc in self.loops:
            if lc.scope is self.scope and name in lc.counter_names:
                raise AnvilError(f"cannot assign to the loop variable `{name}`", span)
        scope = self.scope
        if isinstance(val, CVal) and name in scope.runtime_names:
            val = self.to_tensor(val, span)
            fresh = True
        if isinstance(val, EVal):
            if self.foreign_vars(val.expr):
                scope.vars[name] = Binding(val, span=span)
                return
            val = self.to_tensor(val, span)
            fresh = True
        if isinstance(val, TVal) and val.is_identity() and (
                (fresh and val.buf.kind == "temp" and val.buf.name.startswith("t") and val.buf.name[1:].isdigit())
                or (val.buf.kind == "const" and val.buf.name.startswith("lit"))):
            val.buf.name = name          # name buffers after the variables that hold them
        # loop-carried bookkeeping
        lc = self.innermost_loop()
        if lc is not None and lc.scope is scope:
            if name in lc.carried:
                self.check_carried_type(name, lc.carried[name], val, span)
            elif name in lc.new_names and isinstance(val, (TVal, AffVal, CVal)):
                shape, dtype = self.value_type(val, span)
                lc.carried[name] = self.new_buffer(name, shape, dtype, "var")
            elif self.loops_in_scope() and isinstance(val, (FnVal, ModelInstVal, OptSpecVal)):
                raise AnvilError(f"cannot assign a {val.kind} to `{name}` inside a loop", span)
        scope.vars[name] = Binding(val, span=span)

    def value_type(self, val: Val, span):
        if isinstance(val, TVal):
            return val.shape, val.dtype
        if isinstance(val, AffVal):
            return (), I32
        if isinstance(val, CVal):
            return (), val.dtype
        raise AnvilError(f"a {val.kind} cannot be stored in a variable that changes at run time", span)

    def check_carried_type(self, name, buf: Buffer, val: Val, span):
        shape, dtype = self.value_type(val, span)
        if tuple(shape) != buf.shape:
            raise AnvilError(f"`{name}` changes shape inside the loop: {fmt_shape(buf.shape)} → {fmt_shape(shape)}",
                           span, help="variables that live across loop iterations must keep their shape")
        bad = dims.first_conflict(buf.dims, tuple(shape))
        if bad:
            raise AnvilError(f"`{name}` changes dimensions inside the loop: {dims.show_shape(buf.dims)} → "
                           f"{dims.show_shape(shape)}", span, notes=[dims.mismatch_note(bad[1], bad[2])],
                           help="variables that live across loop iterations must keep their shape")
        if dtype != buf.dtype and not (dtype == I32 and buf.dtype == F32):
            raise AnvilError(f"`{name}` changes type inside the loop: {buf.dtype} → {dtype}", span)

    def innermost_loop(self) -> LoopCtx | None:
        return self.loops[-1] if self.loops else None

    def loops_in_scope(self) -> bool:
        return any(lc.scope is self.scope for lc in self.loops)

    def expr_free_vars(self, e: Expr) -> set:
        return _collect_free_index_vars(e)

    def foreign_vars(self, e: Expr) -> set:
        """Index vars of an enclosing (outer) comprehension context."""
        vs = _collect_free_index_vars(e)
        own = set(self.comp.names.values()) if self.comp else set()
        if self.comp and self.comp.ellipsis:
            own |= set(self.comp.ellipsis)
        return {v for v in vs if v not in own}

    # ------------------------------------------------------------------ writing into buffers
    def write_buffer(self, buf: Buffer, val: Val, span, fresh=False, offset_vars=None):
        """buf[...] = val (broadcast). Retargets the producing kernel when it is safe."""
        if isinstance(val, EVal) and not self.foreign_vars(val.expr):
            val = self.to_tensor(val, span)
        if isinstance(val, TVal):
            if val.shape != buf.shape:
                try:
                    out = self.broadcast([buf.shape, val.shape], span)
                except AnvilError:
                    out = None
                if out != buf.shape:
                    raise AnvilError(f"cannot assign a value of shape {fmt_shape(val.shape)} to `{buf.name}` "
                                   f"of shape {fmt_shape(buf.shape)}", span)
            k = self.last_kernel()
            if fresh and k is not None and val.is_identity() and val.buf.kind == "temp" \
                    and len(k.stmts) == 1 and k.stores and k.stores[0].buf is val.buf \
                    and not k.stores[0].accumulate and val.shape == buf.shape \
                    and val.dtype == buf.dtype and self.pointwise_safe(k, buf):
                st = k.stores[0]
                st.buf = buf
                return
        elif isinstance(val, (CVal, AffVal)):
            pass
        elif not isinstance(val, EVal):
            raise AnvilError(f"cannot assign a {val.kind} to `{buf.name}`", span)
        if isinstance(val, TVal) and val.buf.root is buf.root and not val.is_identity():
            val = self.materialize(TVal(val.buf, val.shape, val.tmpl, val.pvars), span)
        shape = buf.shape
        vars_ = [Var(n, d) for n, d in zip(["i", "j", "k", "l", "m", "n"], shape)] if len(shape) <= 6 else \
            [Var(f"i{d}", n) for d, n in enumerate(shape)]

        def body():
            if isinstance(val, TVal):
                return self.bload(val, vars_, shape, span)
            return self.to_expr(val, span) if isinstance(val, EVal) else self.scalar_expr(val, span)
        e = body()
        if e.dtype != buf.dtype:
            e = cast(e, buf.dtype)
        self.emit_kernel(self.make_kernel(vars_, buf, e, span=span, label="assign"))

    def pointwise_safe(self, k: Kernel, buf: Buffer) -> bool:
        """Can kernel k write `buf` in place? (reads buf only at the store index, no reduction over it)"""
        st = k.stores[0]
        for e in k.exprs():
            for ld in ir.loads_in(e):
                if ld.buf.root is buf.root:
                    if k.red is not None:
                        return False
                    want = identity_offset(buf, k.domain)
                    if ld.offset != want or st.offset != want:
                        return False
        return True

    # ------------------------------------------------------------------ index definitions
    def st_IndexAssign(self, s: A.IndexAssign):
        names = [it for it in s.indices if isinstance(it, A.Name)]
        if names and not s.where and s.target.id in self.scope.vars \
                and any(self.scope.lookup(it.id) is not None for it in names):
            # `S[ptr] = v` where S is a variable of this scope and ptr has a value: item
            # assignment, not an index definition (which may shadow outer names, like `next[m, t]`
            # inside a function when the caller also has an `m`)
            self.setitem(s.target, s.indices, s.value, None, s.span)
            return
        if s.accumulate:
            raise AnvilError("`+=` on an index definition is not supported yet", s.span,
                           help="compute the sum on the right-hand side instead")
        name = s.target.id
        b = self.scope.lookup(name)
        val, ctx = self.eval_comprehension(s.value, s.indices, s.where, s.span)
        lhs = ctx.lhs
        if self.recorder is not None:
            for it in list(s.indices) + [w[0] for w in s.where]:
                if isinstance(it, A.Name) and it.id in ctx.names:
                    self.recorder.note(it.span, IdxVal(Affine.of(ctx.names[it.id])))
        expr = self.to_expr(val, s.value.span) if not isinstance(val, EVal) else val.expr
        if isinstance(val, TVal) and val.rank > 0:
            raise AnvilError(f"the right-hand side is a tensor of shape {fmt_shape(val.shape)}, not an element",
                           s.value.span, help="index it with the left-hand side's indices")
        shape = tuple(v.size for v in lhs)
        if b is not None and b.ref is not None:
            buf = b.ref
            if buf.shape != shape:
                raise AnvilError(f"`{name}` has shape {fmt_shape(buf.shape)} but this definition has shape "
                               f"{fmt_shape(shape)}", s.span)
            e = cast(expr, buf.dtype)
            if any(ld.buf.root is buf.root for ld in ir.loads_in(e)):
                tmp = self.new_temp(shape, e.dtype)
                self.emit_kernel(self.make_kernel(lhs, tmp, e, span=s.span))
                self.write_buffer(buf, TVal.of(tmp), s.span, fresh=True)
            else:
                self.emit_kernel(self.make_kernel(lhs, buf, e, span=s.span))
            return
        buf = self.new_temp(shape, expr.dtype, name)
        self.emit_kernel(self.make_kernel(lhs, buf, expr, span=s.span, label=self.current_label()))
        self.assign(name, TVal.of(buf), s.span, fresh=True)

    # ------------------------------------------------------------------ item assignment
    def st_SetItem(self, s: A.SetItem):
        if s.op in ("+", "-") and self.fresh_index_names(s.items):
            self.scatter(s)
            return
        if s.where:
            raise AnvilError("`where` belongs to index notation, but every subscript here has a value", s.span)
        self.setitem(s.target, s.items, s.value, s.op, s.span)

    def fresh_index_names(self, items) -> list[str]:
        """Names in subscript items that are not variables: index names (`i` in `c[labels[i]]`)."""
        out: list[str] = []

        def walk(n):
            if isinstance(n, A.Name):
                if self.scope.lookup(n.id) is None and n.id not in BUILTINS and n.id not in BUILTIN_CONSTS \
                        and n.id not in out:
                    out.append(n.id)
                return
            if isinstance(n, A.Call):
                for a in n.args:
                    walk(a)
                return
            for k, v in vars(n).items():
                if k == "span":
                    continue
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, A.Node):
                        walk(x)
        for it in items:
            walk(it)
        return out

    def scatter(self, s: A.SetItem):
        """`counts[labels[i]] += 1`, `E[ids[n], d] += g[n, d]`: index notation that adds into an
        existing tensor. The index names in the subscripts are the loop; where several of them reach
        the same element, the contributions add up."""
        fresh = self.fresh_index_names(s.items)
        if any(isinstance(it, (A.EllipsisIdx, A.Slice)) for it in s.items):
            raise AnvilError("`...` and slices cannot be mixed with index names in `+=`", s.span,
                           help="give an index expression for every dimension")
        base = self.owned_target(s.target, s.span)
        sub = A.Subscript(s.span, s.target, list(s.items))
        where_names = [w[0].id for w in s.where]
        for nm in where_names:
            if nm not in fresh:
                raise AnvilError(f"`where {nm} < …`: `{nm}` is not an index name in the subscripts", s.span)
        names, binders = self.scan_indices(A.Binary(s.span, "+", sub, s.value), fresh, where_names)
        ctx = CompCtx(names={}, binders=binders)
        ctx.lhs_items = [A.Name(s.span, n) for n in fresh]
        for nm in list(names) + fresh:
            if nm != "..." and nm not in ctx.names:
                ctx.names[nm] = Var(nm)
        for w_name, w_bound in s.where:
            v = self.eval(w_bound)
            if not isinstance(v, CVal) or not v.is_int:
                raise AnvilError("`where` bounds must be compile-time integers", w_bound.span)
            ctx.where[ctx.names[w_name.id]] = v.value if isinstance(v.value, dims.Dim) else int(v.value)
        self.comps.append(ctx)
        try:
            dest = self.subscript(base, s.items, sub)
            val = self.eval(s.value)
            ctx.lhs = self.lhs_vars(ctx)
            self.solve_ranges(ctx, s.span)
            if isinstance(val, TVal) and val.rank > 0:
                raise AnvilError(f"the right-hand side is a tensor of shape {fmt_shape(val.shape)}, not an element",
                               s.value.span, help="index it with the left-hand side's index names")
            if isinstance(val, (EVal, IdxVal)):
                expr = self.to_expr(val, s.value.span)
            else:
                expr = self.scalar_expr(val, s.value.span)
            final = self.finalize(ctx, expr, s.span)
        finally:
            self.comps.pop()
        if not isinstance(dest, EVal) or not isinstance(dest.expr, Load):
            raise AnvilError("unsupported subscript in `+=`", s.span)
        free = _collect_free_index_vars(final) - set(ctx.lhs)
        if free & set(ctx.names.values()):
            name = next(iter(free & set(ctx.names.values()))).name
            raise AnvilError(f"index `{name}` is not on the left-hand side and not inside a reduction", s.span,
                           help="reduce over it, e.g. `+= sum ...`, or use it in the subscript on the left")
        if s.op == "-":
            final = mk_binary("sub", Const(0, final.dtype), final)
        if any(ld.buf.root is base.buf.root for ld in ir.loads_in(final)):
            tmp = self.new_temp(tuple(v.size for v in ctx.lhs), final.dtype)      # read before adding
            self.emit_kernel(self.make_kernel(ctx.lhs, tmp, final, span=s.span))
            final = Load(tmp, identity_offset(tmp, ctx.lhs))
        self.emit_kernel(self.make_kernel(ctx.lhs, base.buf, final, span=s.span, label="scatter",
                                          accumulate=True, out_offset=dest.expr.offset))

    def used_outside(self, name: str, node) -> bool:
        """Does `name` occur anywhere in the current scope's code outside `node`?
        (If not, a value assigned inside `node` never needs to survive it.)"""
        stmts = self.scope_stmts.get(id(self.scope))
        if stmts is None:
            return True

        def occurs(n) -> bool:
            if n is node:
                return False
            if isinstance(n, A.Name):
                return n.id == name
            if isinstance(n, A.Param):
                return n.name == name
            if isinstance(n, (A.FnDecl, A.ModelDecl, A.OptimizerDecl)):
                return False                # separate scopes
            if isinstance(n, A.Str):
                return any(isinstance(p, A.Interp) and occurs(p.expr) for p in n.parts)
            for k, v in vars(n).items():
                if k == "span":
                    continue
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, tuple):
                        if any(isinstance(y, A.Node) and occurs(y) for y in x):
                            return True
                    elif isinstance(x, A.Node) and occurs(x):
                        return True
            return False
        return any(occurs(s) for s in stmts)

    def aliased(self, name: str, buf: Buffer) -> bool:
        """Is `buf` also reachable through a binding other than `name`?"""
        sc = self.scope
        while sc is not None:
            for nm, b in sc.vars.items():
                if nm != name and isinstance(b.val, TVal) and b.val.buf.root is buf.root:
                    return True
            sc = sc.parent
        return False

    def setitem(self, target: A.Name, items, value_node, op, span):
        """`x[...] = v` / `x[...] op= v`: write into part of an existing tensor.

        Parameters and optimizer state are updated in place. A local variable is updated in
        place when it owns its buffer (loop-carried or already copied) and nothing else
        refers to it; otherwise it is copied first, so other names keep the old value."""
        base = self.owned_target(target, span)
        sub = A.Subscript(span, target, list(items))
        view = self.subscript(base, items, sub)
        if not isinstance(view, TVal):
            raise AnvilError("an assigned element must be selected by integer values or slices", span,
                           help="to define a whole tensor elementwise use fresh index names: `y[i, j] = ...`")
        rhs = self.eval_top(value_node)
        if op is not None:
            rhs = self.matmul(view, rhs, span) if op == "@" else self.binary_op(BINOP[op], view, rhs, span)
        self.write_view(view, rhs, span)

    def owned_target(self, target: A.Name, span) -> TVal:
        """The tensor `target` names, ready to be written in place: a param or state as it is, a
        local variable once it owns its buffer (it is copied first if anything else refers to it)."""
        name = target.id
        b = self.scope.lookup(name)
        if b is None:
            raise self.undefined(name, target.span)
        if isinstance(b.val, PoisonVal):
            raise PoisonError(f"`{name}` could not be defined", target.span)
        if not isinstance(b.val, TVal):
            raise AnvilError(f"cannot assign into `{name}`: it is a {b.val.kind}, not a tensor", target.span)
        if b.ref is not None:
            return TVal.of(b.ref)
        cur = b.val
        if not cur.is_identity() or cur.buf.kind != "var" or self.aliased(name, cur.buf):
            buf = self.new_buffer(name, cur.shape, cur.dtype, "var")
            self.write_buffer(buf, cur, span)
            self.assign(name, TVal.of(buf), span)
            cur = self.scope.lookup(name).val
        return cur

    def write_view(self, view: TVal, val: Val, span):
        """Store `val` (broadcast) into the elements selected by `view`."""
        if isinstance(val, EVal):
            if self.foreign_vars(val.expr):
                raise AnvilError("cannot assign an index expression here", span)
            val = self.to_tensor(val, span)
        if isinstance(val, TVal):
            out = self.broadcast([view.shape, val.shape], span)
            if out != view.shape:
                raise AnvilError(f"cannot assign a value of shape {fmt_shape(val.shape)} into a selection of "
                               f"shape {fmt_shape(view.shape)}", span)
            if val.buf.root is view.buf.root:
                val = self.emit_map(val.shape, lambda vs: val.load([Affine.of(v) for v in vs]),
                                    span=span, label="copy")      # source overlaps destination
        elif not isinstance(val, (CVal, AffVal)):
            raise AnvilError(f"cannot assign a {val.kind} into a tensor", span)
        vars_ = [Var(n, d) for n, d in zip("ijklmnpq", view.shape)]
        e = self.bload(val, vars_, view.shape, span) if isinstance(val, TVal) else self.scalar_expr(val, span)
        if e.dtype != view.dtype:
            e = cast(e, view.dtype)
        off = view.offset([Affine.of(v) for v in vars_])
        self.emit_kernel(self.make_kernel(vars_, view.buf, e, span=span, label="setitem", out_offset=off))

    # ------------------------------------------------------------------ other statements
    def st_ExprStmt(self, s: A.ExprStmt):
        self.eval_top(s.expr)

    def st_Pass(self, s):
        pass

    def st_Return(self, s: A.Return):
        if not self.fn_stack:
            raise AnvilError("`return` outside of a function", s.span)
        if self.runtime_depth > self.fn_runtime_base[-1]:
            raise AnvilError("`return` inside a runtime `if`/loop is not supported (functions are inlined)",
                           s.span, help="compute the value into a variable and return it at the end")
        val = NoneVal() if s.value is None else self.eval_top(s.value)
        raise ReturnSignal(val)

    def st_Assert(self, s: A.Assert):
        """`assert cond, "message {x}"`: checked by the compiler when cond is known, otherwise when the
        program runs. A failure prints the message (with the file and line) and exits with status 1."""
        cond = self.eval_top(s.cond)
        where = f"{os.path.basename(s.span.file.path)}:{s.span.line}" if s.span is not None else "?"
        if isinstance(cond, CVal):
            if not cond.value:
                raise AnvilError("this assertion is false (already at compile time)", s.cond.span)
            return
        cbuf = self.runtime_cond(cond, s.cond.span)
        fail = Block()
        self.blocks.append(fail)
        try:
            items = [ir.PrintItem("text", f"anvil: assertion failed at {where}")]
            if s.message is not None:
                msg = self.eval_top(s.message)
                items.append(ir.PrintItem("text", ": "))
                parts = msg.parts if isinstance(msg, FStrVal) else [msg]
                for p in parts:
                    got = [ir.PrintItem("text", p)] if isinstance(p, str) else \
                        self.print_value(*p) if isinstance(p, tuple) else self.print_value(p, "", s.message.span)
                    for it in got:
                        if it.kind in ("tensor", "chars"):
                            raise AnvilError("an assertion message can show numbers and strings, not whole tensors",
                                           s.message.span)
                        if it.kind == "scalar" and not it.fmt:
                            it.fmt = "%d" if it.buf.dtype == I32 else "%g"
                        items.append(it)
            self.emit(ir.Print(items, "\n", err=True))
            self.emit(ir.RTCall("exit", {"value": 1}, span=s.span))
        finally:
            self.blocks.pop()
        self.emit(ir.If(cbuf, Block(), fail, span=s.span))

    def st_Minimize(self, s: A.Minimize):
        from .autodiff import minimize
        minimize(self, s)

    # ------------------------------------------------------------------ control flow
    def runtime_cond(self, val: Val, span) -> Buffer:
        if isinstance(val, AffVal):
            val = self.to_tensor(val, span)
        if isinstance(val, EVal):
            val = self.to_tensor(val, span)
        if not isinstance(val, TVal) or val.rank != 0:
            raise AnvilError(f"a condition must be a scalar, found {describe(val)}", span,
                           help="reduce it, e.g. `any`-style: `sum(mask) > 0`")
        if not val.is_identity():
            val = self.materialize(val, span)
        return val.buf

    def st_If(self, s: A.If):
        cond = self.eval_top(s.cond)
        if isinstance(cond, CVal):
            self.exec_block(s.body if cond.value else s.orelse)
            return
        cbuf = self.runtime_cond(cond, s.cond.span)
        scope = self.scope
        snapshot = dict(scope.vars)
        then_b, else_b = Block(), Block()
        self.blocks.append(then_b)
        self.runtime_depth += 1
        try:
            self.exec_block(s.body)
        finally:
            self.blocks.pop()
        then_vars = scope.vars
        scope.vars = dict(snapshot)
        self.blocks.append(else_b)
        try:
            self.exec_block(s.orelse)
        finally:
            self.blocks.pop()
            self.runtime_depth -= 1
        else_vars = scope.vars
        merged = {}
        for name in list(dict.fromkeys(list(then_vars) + list(else_vars))):
            tb, eb = then_vars.get(name), else_vars.get(name)
            if tb is eb:
                merged[name] = tb
                continue
            if not self.used_outside(name, s):
                if name in snapshot:               # only used inside this `if`: nothing to merge
                    merged[name] = snapshot[name]
                continue
            tv = tb.val if tb is not None else None
            ev = eb.val if eb is not None else None
            datas = [v for v in (tv, ev) if v is not None]
            if any(not isinstance(v, (TVal, CVal, AffVal)) for v in datas):
                if isinstance(tv, EVal) or isinstance(ev, EVal):
                    raise AnvilError(f"`{name}` is an index expression assigned inside a runtime `if`", s.span)
                raise AnvilError(f"`{name}` is assigned a {datas[0].kind} inside a runtime `if`", s.span)
            shapes = {self.value_type(v, s.span) for v in datas}
            if len({sh for sh, _ in shapes}) > 1:
                raise AnvilError(f"`{name}` has different shapes in the two branches of this `if`", s.span)
            shape, dtype = next(iter(shapes))
            if len({dt for _, dt in shapes}) > 1:
                dtype = F32
            canon = self.new_buffer(name, shape, dtype, "var")
            for blk, v in ((then_b, tv), (else_b, ev)):
                if v is not None:
                    self.blocks.append(blk)
                    try:
                        self.write_buffer(canon, v, s.span)
                    finally:
                        self.blocks.pop()
            merged[name] = Binding(TVal.of(canon), span=s.span)
            lc = self.innermost_loop()
            if lc is not None and lc.scope is scope and name in lc.new_names and name not in lc.carried:
                lc.carried[name] = self.new_buffer(name, shape, dtype, "var")
        scope.vars = merged
        self.emit(ir.If(cbuf, then_b, else_b, span=s.span))

    def setup_loop(self, body, targets: list[str], span) -> LoopCtx:
        scope = self.scope
        assigned = list(dict.fromkeys(A.walk_assigned_names(body)))
        carried: dict[str, Buffer] = {}
        new_names: set[str] = set()
        for name in assigned:
            if name in targets:
                continue
            b = scope.vars.get(name)
            if b is None:
                ob = scope.lookup(name)
                if ob is not None and ob.ref is not None:
                    continue
                if self.used_outside(name, self.current_loop_node):
                    new_names.add(name)            # first assigned in the loop, read after it
                continue
            if b.ref is not None:
                continue
            if isinstance(b.val, TVal) and b.val.is_identity() and b.val.buf.kind == "var" \
                    and not self.aliased(name, b.val.buf):
                carried[name] = b.val.buf          # already its own slot: no copy on loop entry
            elif isinstance(b.val, (TVal, CVal, AffVal)):
                shape, dtype = self.value_type(b.val, span)
                canon = self.new_buffer(name, shape, dtype, "var")
                self.write_buffer(canon, b.val, span)
                scope.vars[name] = Binding(TVal.of(canon), span=b.span)
                carried[name] = canon
            elif isinstance(b.val, EVal):
                raise AnvilError(f"`{name}` is an index expression and cannot be reassigned in a loop", span)
            else:
                new_names.add(name)
        return LoopCtx(scope, carried, new_names, set(targets))

    def write_backs(self, lc: LoopCtx, span):
        for name, canon in lc.carried.items():
            b = lc.scope.vars.get(name)
            if b is None or b.ref is not None:
                continue
            if isinstance(b.val, TVal) and b.val.buf is canon and b.val.is_identity():
                continue
            if isinstance(b.val, (TVal, CVal, AffVal)):
                self.write_buffer(canon, b.val, span)

    def finish_loop(self, lc: LoopCtx):
        for name, canon in lc.carried.items():
            lc.scope.vars[name] = Binding(TVal.of(canon))

    def static_for(self, s: A.For):
        """`static for t in range(T)`: the body is elaborated once per value, with t a compile-time
        constant, so there is no runtime loop: values flow from one step to the next as ordinary
        variables, and gradients flow back through all the steps (backpropagation through time)."""
        it = self.eval_top(s.iter)
        names = [t.id for t in s.targets]
        if len(names) != 1:
            raise AnvilError("a `static for` takes one loop variable", s.span)
        values = self.unrolled_values(it, s.iter.span, "a `static for`")
        if len(values) > 1000:
            raise AnvilError(f"a `static for` of {len(values)} steps is too long to unroll", s.iter.span)
        self.static_marks.append(len(self.loops))
        try:
            for v in values:
                self.scope.vars[names[0]] = Binding(v, what="loop variable")
                if self.recorder is not None:
                    self.recorder.note(s.targets[0].span, v, definition=True)
                self.exec_block(s.body)
        finally:
            self.static_marks.pop()

    def unrolled_values(self, it: Val, span, what: str) -> list:
        """The values a `static for` or a list comprehension goes through, all known at compile time."""
        if isinstance(it, RangeVal):
            bounds = []
            for b in (it.start, it.stop):
                if not isinstance(b, CVal) or not b.is_int:
                    raise AnvilError(f"{what} needs a range known at compile time", span,
                                   help="use a plain `for` for a loop whose length is only known when it runs")
                bounds.append(int(b.value))
            return [CVal(v) for v in range(bounds[0], bounds[1], it.step)]
        if isinstance(it, TupleVal):
            return list(it.items)
        if isinstance(it, TVal) and it.rank > 0:
            rest = [("slice", None, None, None, span)] * (it.rank - 1)
            return [self.subscript_value(it, [CVal(k)] + rest, span) for k in range(it.shape[0])]
        raise AnvilError(f"{what} loops over a range, a tuple or a tensor, not {describe(it)}", span)

    def list_comp(self, n: A.ListComp, model_list: str | None = None) -> TupleVal:
        """[elt for var in iter]. With model_list, the models it creates are named `model_list.0`, `.1`, ..."""
        values = self.unrolled_values(self.eval(n.iter), n.iter.span, "a list comprehension")
        if len(values) > 1000:
            raise AnvilError(f"a list comprehension of {len(values)} items is too long", n.iter.span)
        saved = self.scope.vars.get(n.var.id)
        out = []
        try:
            for k, v in enumerate(values):
                self.scope.vars[n.var.id] = Binding(v, what="loop variable")
                f = self.try_eval_callee(n.elt.func) if isinstance(n.elt, A.Call) else None
                if isinstance(f, ModelDefVal):
                    out.append(self.instantiate(f, n.elt, f"{self.scope.prefix}{model_list or 'models'}.{k}"))
                else:
                    out.append(self.eval(n.elt))
        finally:
            if saved is None:
                self.scope.vars.pop(n.var.id, None)
            else:
                self.scope.vars[n.var.id] = saved
        return TupleVal(out)

    def ev_ListComp(self, n: A.ListComp):
        return self.list_comp(n)

    def st_For(self, s: A.For):
        if s.static:
            return self.static_for(s)
        self.current_loop_node = s
        it = self.eval_top(s.iter)
        names = [t.id for t in s.targets]
        if isinstance(it, TVal):
            if len(names) != 1:
                raise AnvilError("iterating over a tensor yields one row at a time", s.span)
            if it.rank == 0:
                raise AnvilError("cannot iterate over a scalar", s.iter.span)
            it_tensor = it
            it = RangeVal(CVal(0), CVal(it.shape[0]), 1)
        else:
            it_tensor = None
        if isinstance(it, BatchesVal):
            return self.for_batches(s, it, names)
        if not isinstance(it, RangeVal):
            raise AnvilError(f"cannot loop over a {it.kind}", s.iter.span,
                           help="use `range(n)`, `batches(x, y, size=...)`, or a tensor")
        if len(names) != 1 and it_tensor is None:
            raise AnvilError("`range` yields one value per iteration", s.span)
        counter = self.new_buffer(names[0] if it_tensor is None else f"{names[0]}_idx", (), I32, "var")
        start = self.loop_bound(it.start, s.iter.span)
        stop = self.loop_bound(it.stop, s.iter.span)
        lc = self.setup_loop(s.body, names, s.span)
        body = Block(is_loop=True)
        self.blocks.append(body)
        self.loops.append(lc)
        self.runtime_depth += 1
        try:
            if it_tensor is None:
                self.scope.vars[names[0]] = Binding(AffVal(Affine.of(ScalarRef(counter))), what="loop variable")
            else:
                row = self.subscript_value(it_tensor, [AffVal(Affine.of(ScalarRef(counter)))], s.iter.span)
                self.scope.vars[names[0]] = Binding(row, what="loop variable")
            if self.recorder is not None:
                self.recorder.note(s.targets[0].span, self.scope.vars[names[0]].val, definition=True)
            self.exec_block(s.body)
            self.write_backs(lc, s.span)
        finally:
            self.blocks.pop()
            self.loops.pop()
            self.runtime_depth -= 1
        self.finish_loop(lc)
        self.emit(ir.For(counter, start, stop, it.step, body, span=s.span))

    def loop_bound(self, v: Val, span):
        if isinstance(v, CVal):
            if not v.is_int:
                raise AnvilError("loop bounds must be integers", span)
            return int(v.value)
        if isinstance(v, AffVal):
            if v.affine.is_const():
                return v.affine.const
            t = self.to_tensor(v, span, I32)
            return t.buf
        if isinstance(v, TVal) and v.rank == 0 and v.dtype == I32:
            snap = self.new_buffer("bound", (), I32, "var")
            self.write_buffer(snap, v, span)
            return snap
        raise AnvilError(f"loop bounds must be integers, found {describe(v)}", span)

    def for_batches(self, s: A.For, it: BatchesVal, names):
        if len(names) != len(it.tensors):
            raise AnvilError(f"`batches` yields {len(it.tensors)} value(s) per step but the loop unpacks "
                           f"{len(names)}", s.span)
        n = it.tensors[0].shape[0]
        nb = n // it.size
        counter = self.new_buffer(f"{names[0]}_batch", (), I32, "var")
        perm = None
        if it.shuffle:
            perm = self.new_buffer("perm", (n,), I32, "state")
            self.emit(ir.RTCall("iota_once", {"buf": perm}, span=s.span))
            self.emit(ir.RTCall("shuffle", {"buf": perm}, span=s.span))
        lc = self.setup_loop(s.body, names, s.span)
        body = Block(is_loop=True)
        self.blocks.append(body)
        self.loops.append(lc)
        self.runtime_depth += 1
        try:
            base = Affine(0, {ScalarRef(counter): it.size})
            for nm, t in zip(names, it.tensors):
                if perm is None:
                    def mapping(pv, t=t):
                        return [base + Affine.of(pv[0])] + [Affine.of(p) for p in pv[1:]]
                else:
                    def mapping(pv, t=t):
                        g = Gather(Load(perm, base + Affine.of(pv[0])), n, "shuffled batch index")
                        return [Affine(0, {g: 1})] + [Affine.of(p) for p in pv[1:]]
                v = self.view(t, (it.size,) + t.shape[1:], mapping, s.span)
                if perm is not None:
                    # gather the shuffled rows once per step into a contiguous batch, so every
                    # kernel that uses it (forward and backward) reads plain rows
                    v = self.materialize(v, s.span, name=nm)
                self.scope.vars[nm] = Binding(v, what="loop variable")
                if self.recorder is not None:
                    self.recorder.note(next(t for t in s.targets if t.id == nm).span, v, definition=True)
            self.exec_block(s.body)
            self.write_backs(lc, s.span)
        finally:
            self.blocks.pop()
            self.loops.pop()
            self.runtime_depth -= 1
        self.finish_loop(lc)
        self.emit(ir.For(counter, 0, nb, 1, body, span=s.span))

    def st_While(self, s: A.While):
        self.current_loop_node = s
        lc = self.setup_loop(s.body, [], s.span)
        cond_block = Block()
        self.blocks.append(cond_block)
        try:
            cond = self.eval_top(s.cond)
            if isinstance(cond, CVal):
                cond = self.to_tensor(CVal(1.0 if cond.value else 0.0), s.cond.span)
            cbuf = self.runtime_cond(cond, s.cond.span)
        finally:
            self.blocks.pop()
        body = Block(is_loop=True)
        self.blocks.append(body)
        self.loops.append(lc)
        self.runtime_depth += 1
        try:
            self.exec_block(s.body)
            self.write_backs(lc, s.span)
        finally:
            self.blocks.pop()
            self.loops.pop()
            self.runtime_depth -= 1
        self.finish_loop(lc)
        self.emit(ir.While(cond_block, cbuf, body, span=s.span))

    def loop_jump(self, s, kind):
        if self.static_marks and self.static_marks[-1] == len(self.loops):
            raise AnvilError(f"`{kind}` inside a `static for` is not supported (the loop is unrolled)", s.span)
        lc = self.innermost_loop()
        if lc is None or lc.scope is not self.scope:
            raise AnvilError(f"`{kind}` outside of a loop", s.span)
        self.write_backs(lc, s.span)
        self.emit(ir.Break() if kind == "break" else ir.Continue())

    def st_Break(self, s):
        self.loop_jump(s, "break")

    def st_Continue(self, s):
        self.loop_jump(s, "continue")

    # ------------------------------------------------------------------ types
    def eval_type(self, t: A.TypeExpr, scope: Scope, shape_vars: dict | None = None):
        dtype = t.dtype or F32
        if t.dims is None:
            return dtype, ()
        dims = []
        for d in t.dims:
            v = self.eval_dim(d, scope, shape_vars)
            dims.append(v)
        return dtype, tuple(dims)

    def eval_dim(self, d: A.Expr, scope: Scope, shape_vars: dict | None = None) -> int:
        saved = self.scope
        self.scope = scope
        try:
            if shape_vars and isinstance(d, A.Name) and d.id in shape_vars:
                return shape_vars[d.id]
            v = self.eval(d)
        finally:
            self.scope = saved
        if not isinstance(v, CVal) or not v.is_int or isinstance(v.value, bool):
            raise AnvilError(f"shape dimensions must be compile-time integers, found {describe(v)}", d.span)
        if v.value < 0:
            raise AnvilError(f"negative dimension {v.value}", d.span)
        return v.value if isinstance(v.value, dims.Dim) else int(v.value)

    # ------------------------------------------------------------------ expressions
    def undefined(self, name, span) -> AnvilError:
        cands = self.scope.all_names() | set(BUILTINS) | set(BUILTIN_CONSTS)
        s = suggest(name, cands)
        return AnvilError(f"`{name}` is not defined", span, help=f"did you mean `{s}`?" if s else None)

    def lookup_name(self, name, span) -> Val:
        if self.comp is not None:
            ctx = self.comp
            if name in ctx.names:
                return IdxVal(Affine.of(ctx.names[name]))
        b = self.scope.lookup(name)
        if b is not None:
            if isinstance(b.val, PoisonVal):
                raise PoisonError(f"`{name}` could not be defined", span)
            return b.val
        if name in BUILTIN_CONSTS:
            return CVal(BUILTIN_CONSTS[name])
        if name in BUILTINS:
            return BuiltinVal(name)
        raise self.undefined(name, span)

    def try_eval_callee(self, node):
        if isinstance(node, A.Name):
            b = self.scope.lookup(node.id)
            if b is not None:
                return b.val
        return None

    def eval(self, node: A.Expr) -> Val:
        meth = getattr(self, "ev_" + type(node).__name__, None)
        if meth is None:
            raise AnvilError(f"unsupported expression {type(node).__name__}", node.span)
        return meth(node)

    def ev_Name(self, n: A.Name):
        v = self.lookup_name(n.id, n.span)
        if self.recorder is not None:
            self.recorder.note(n.span, v)
        return v

    def ev_Num(self, n: A.Num):
        return CVal(n.value)

    def ev_Bool(self, n: A.Bool):
        return CVal(n.value)

    def ev_Str(self, n: A.Str):
        if n.is_plain:
            return SVal(n.plain)
        parts = []
        for p in n.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append((self.eval(p.expr), p.spec, p.span))
        # only strings and integers known at compile time: the text is known too ("encoder.layer.{k}.")
        if all(isinstance(p, str) or isinstance(p[0], SVal) and not p[1]
               or isinstance(p[0], CVal) and p[0].is_int and not isinstance(p[0].value, bool)
               and (not p[1] or p[1].endswith("d")) for p in parts):
            return SVal("".join(p if isinstance(p, str) else p[0].value if isinstance(p[0], SVal)
                                else format(int(p[0].value), p[1] or "d") for p in parts))
        return FStrVal(parts)

    def ev_TupleLit(self, n: A.TupleLit):
        return TupleVal([self.eval(x) for x in n.items])

    def ev_ListLit(self, n: A.ListLit):
        return self.tensor_literal(n)

    def ev_Unary(self, n: A.Unary):
        v = self.eval(n.operand)
        if n.op == "-":
            if isinstance(v, CVal) and not isinstance(v.value, bool):
                return CVal(-v.value)
            return self.unary_op("neg", v, n.span)
        if n.op == "√":
            return self.unary_op("sqrt", v, n.span)
        if n.op == "not":
            if isinstance(v, CVal):
                return CVal(not v.value)
            return self.unary_op("not", v, n.span)
        raise AnvilError(f"unknown operator {n.op}", n.span)

    def ev_Binary(self, n: A.Binary):
        if n.op in ("and", "or"):
            a = self.eval(n.left)
            if isinstance(a, CVal):
                if n.op == "and" and not a.value:
                    return CVal(False)
                if n.op == "or" and a.value:
                    return CVal(True)
                b = self.eval(n.right)
                return CVal(bool(b.value)) if isinstance(b, CVal) else self.binary_op("ne", b, CVal(0), n.span)
            b = self.eval(n.right)
            return self.binary_op(n.op, a, b, n.span)
        a = self.eval(n.left)
        b = self.eval(n.right)
        if n.op == "@":
            return self.matmul(a, b, n.span)
        if isinstance(a, SVal) or isinstance(b, SVal):
            if n.op == "+" and isinstance(a, SVal) and isinstance(b, SVal):
                return SVal(a.value + b.value)
            if n.op == "*" and isinstance(a, SVal) != isinstance(b, SVal):
                k = b if isinstance(a, SVal) else a
                if isinstance(k, CVal) and k.is_int:
                    return SVal((a if isinstance(a, SVal) else b).value * int(k.value))
            if n.op in ("==", "!=") and isinstance(a, SVal) and isinstance(b, SVal):
                return CVal((a.value == b.value) == (n.op == "=="))        # e.g. `if OPT == "adam":`
            raise AnvilError("strings support `+` (joining), `*` by a constant integer (repeating), and "
                           "`==` / `!=`", n.span)
        if isinstance(a, TupleVal) or isinstance(b, TupleVal):
            if n.op == "==" and isinstance(a, TupleVal) and isinstance(b, TupleVal):
                return CVal(all(isinstance(x, CVal) and isinstance(y, CVal) and x.value == y.value
                                for x, y in zip(a.items, b.items)) and len(a.items) == len(b.items))
            raise AnvilError(f"cannot use `{n.op}` on a tuple", n.span)
        for v, side in ((a, n.left), (b, n.right)):
            if not isinstance(v, (CVal, TVal, AffVal, EVal, IdxVal)):
                raise AnvilError(f"cannot use `{n.op}` on a {v.kind}", side.span)
        return self.binary_op(BINOP[n.op], a, b, n.span)

    def ev_Cond(self, n: A.Cond):
        c = self.eval(n.cond)
        if isinstance(c, CVal):
            return self.eval(n.then if c.value else n.orelse)
        a = self.eval(n.then)
        b = self.eval(n.orelse)
        return self.select_op(c, a, b, n.span)

    def ev_Attr(self, n: A.Attr):
        base = self.eval(n.value)
        name = n.name
        if isinstance(base, ModelInstVal):
            b = base.scope.vars.get(name)
            if b is None:
                s = suggest(name, base.scope.vars)
                raise AnvilError(f"model `{base.name}` has no member `{name}`", n.span,
                               help=f"did you mean `{s}`?" if s else None)
            return b.val
        if isinstance(base, TVal):
            if name == "T":
                return self.transpose(base, list(reversed(range(base.rank))), n.span)
            if name == "shape":
                return TupleVal([CVal(d) for d in base.shape])
            if name in ("ndim", "rank"):
                return CVal(base.rank)
            if name in ("size", "numel"):
                return CVal(base.numel)
            if name == "dtype":
                return SVal(base.dtype)
        if isinstance(base, (CVal, AffVal, EVal, IdxVal, TVal, TupleVal)):
            # uniform function call syntax: x.f(args) == f(x, args)
            f = self.scope.lookup(name)
            fv = f.val if f is not None else (BuiltinVal(name) if name in BUILTINS else None)
            if fv is not None and isinstance(fv, (FnVal, BuiltinVal)):
                return MethodVal(fv, base)
        raise AnvilError(f"{base.kind} has no attribute `{name}`", n.span)

    def ev_Subscript(self, n: A.Subscript):
        base = self.eval(n.value)
        if isinstance(base, TupleVal):
            if len(n.items) != 1:
                raise AnvilError("tuples take a single index", n.span)
            i = self.eval(n.items[0])
            if not isinstance(i, CVal) and base.items and all(isinstance(x, SVal) for x in base.items):
                return self.pick_string(base, i, n.items[0].span)
            if not isinstance(i, CVal) or not i.is_int:
                raise AnvilError("tuple index must be a constant integer", n.items[0].span)
            try:
                return base.items[i.value]
            except IndexError:
                raise AnvilError(f"tuple index {i.value} out of range", n.span)
        if not isinstance(base, TVal):
            raise AnvilError(f"cannot index a {base.kind}", n.span)
        return self.subscript(base, n.items, n)

    def pick_string(self, base: TupleVal, i: Val, span) -> PickVal:
        """`NAMES[k]` for a list of strings and a run-time integer k (for printing)."""
        if isinstance(i, EVal):
            raise AnvilError("a list of strings cannot be indexed inside index notation", span)
        if not isinstance(i, (AffVal, TVal)) or (isinstance(i, TVal) and (i.rank != 0 or i.dtype != I32)):
            raise AnvilError(f"a list of strings is indexed by an integer, not {describe(i)}", span)
        t = self.to_tensor(i, span, I32)
        if not t.is_identity():
            t = self.materialize(t, span)
        return PickVal([x.value for x in base.items], t)

    def ev_Reduce(self, n: A.Reduce):
        return self.reduction(n.op, n.operand, n, n.span)

    def ev_Call(self, n: A.Call):
        if isinstance(n.func, A.Name) and n.func.id in REDUCTIONS and len(n.args) == 1 and not n.kwargs \
                and self.scope.lookup(n.func.id) is None:
            return self.reduction(n.func.id, n.args[0], n, n.span)
        f = self.eval(n.func)
        if isinstance(f, ModelDefVal):
            return self.instantiate(f, n, None)
        args = [self.eval(a) for a in n.args]
        kwargs = {k.name: self.eval(k.value) for k in n.kwargs}
        return self.call(f, args, kwargs, n)

    def call(self, f: Val, args, kwargs, node) -> Val:
        if isinstance(f, MethodVal):
            return self.call(f.func, [f.self_val] + args, kwargs, node)
        if isinstance(f, BuiltinVal):
            return self.call_builtin(f.name, args, kwargs, node)
        if isinstance(f, FnVal):
            return self.call_fn(f, args, kwargs, node)
        if isinstance(f, ModelInstVal):
            b = f.scope.vars.get("forward")
            if b is None:
                raise AnvilError(f"model `{f.name}` has no `forward` method, so it cannot be called", node.span)
            return self.call(b.val, args, kwargs, node)
        if isinstance(f, OptDefVal):
            return self.optimizer_spec(f, args, kwargs, node)
        raise AnvilError(f"a {f.kind} is not callable", node.span)

    # ------------------------------------------------------------------ functions
    def bind_args(self, params: list[A.Param], args, kwargs, node, what: str, defaults_scope: Scope):
        out = {}
        if len(args) > len(params):
            raise AnvilError(f"{what} takes {len(params)} argument(s) but {len(args)} were given", node.span)
        for p, a in zip(params, args):
            out[p.name] = a
        for k, v in kwargs.items():
            if not any(p.name == k for p in params):
                s = suggest(k, [p.name for p in params])
                raise AnvilError(f"{what} has no parameter `{k}`", node.span, help=f"did you mean `{s}`?" if s else None)
            if k in out:
                raise AnvilError(f"argument `{k}` given twice", node.span)
            out[k] = v
        for p in params:
            if p.name not in out:
                if p.default is None:
                    raise AnvilError(f"{what} is missing argument `{p.name}`", node.span)
                saved = self.scope
                self.scope = defaults_scope
                try:
                    out[p.name] = self.eval_top(p.default)
                finally:
                    self.scope = saved
        return out

    def call_fn(self, f: FnVal, args, kwargs, node) -> Val:
        decl: A.FnDecl = f.decl
        if self.fn_stack.count(f.name) > 0 and len(self.fn_stack) > 64:
            raise AnvilError(f"recursive call to `{f.name}` (functions are inlined; recursion is not supported)",
                           node.span)
        if f.name in self.fn_stack and self.fn_stack.count(f.name) >= 8:
            raise AnvilError(f"recursive call to `{f.name}` (functions are inlined; recursion is not supported)",
                           node.span)
        bound = self.bind_args(decl.params, args, kwargs, node, f"`{f.name}`", f.scope)
        scope = Scope(parent=f.scope, kind="fn", prefix=f.scope.prefix)
        shape_vars: dict[str, int] = {}
        argvals = {}
        for p in decl.params:
            v = bound[p.name]
            if p.type is not None:
                v = self.check_param_type(p, v, f, shape_vars, node)
            argvals[p.name] = v
            scope.vars[p.name] = Binding(v, what="argument")
            if self.recorder is not None:
                self.recorder.note(Span(p.span.file, p.span.start, p.span.start + len(p.name)), v, definition=True)
        for k, val in shape_vars.items():
            scope.vars.setdefault(k, Binding(CVal(val), what="const"))
        body = decl.body
        saved_scope = self.scope
        self.scope = scope
        self.fn_stack.append(f.name)
        self.comps.append(None)        # index names don't leak into the callee
        self.fn_runtime_base.append(self.runtime_depth)
        try:
            if isinstance(body, A.Expr):
                result = self.eval_top(body)
            else:
                scope.runtime_names = compute_runtime_names(body)
                self.scope_stmts[id(scope)] = body
                result = NoneVal()
                try:
                    for i, st in enumerate(body):
                        if i == len(body) - 1 and isinstance(st, A.ExprStmt):
                            self.span_stack.append(st.span)
                            try:
                                result = self.eval_top(st.expr)
                            finally:
                                self.span_stack.pop()
                        else:
                            self.exec_stmt(st)
                except ReturnSignal as r:
                    result = r.value
        except AnvilError as e:
            if not isinstance(e, PoisonError):
                self.add_call_site(e, f, node)
            raise
        finally:
            self.scope = saved_scope
            self.fn_stack.pop()
            self.comps.pop()
            self.fn_runtime_base.pop()
        if decl.ret is not None:
            result = self.check_return_type(decl.ret, result, f, shape_vars, node)
        if self.recorder is not None:
            from .ide import CallSig, short
            shown = [p for p in decl.params if not isinstance(argvals[p.name], (FnVal, ModelInstVal))]
            sig = f"{f.name}(" + ", ".join(f"{p.name}: {short(argvals[p.name])}" for p in shown) + f") -> {short(result)}"
            callee = getattr(node, "func", None)
            self.recorder.note(callee.span if callee is not None else node.span, CallSig(sig))
        return result

    def add_call_site(self, e: AnvilError, f: FnVal, node):
        """An error inside an inlined function also shows the calls that led to it, so an error in
        the standard library points back at the line of the program that caused it."""
        if node is None or node.span is None or e.span is None or len(e.labels) >= MAX_CALL_LABELS:
            return
        if node.span.file is e.span.file and node.span.start <= e.span.start < node.span.end:
            return                                         # the call is already what the error shows
        if any(lab.span == node.span for lab in e.labels):
            return
        e.labels.append(Label(node.span, f"in this call to `{f.name}`"))

    def check_param_type(self, p: A.Param, v: Val, f: FnVal, shape_vars, node) -> Val:
        t = p.type
        dtype = t.dtype or F32
        if t.dims is None:
            if isinstance(v, (CVal, AffVal, EVal, IdxVal)):
                return v
            if isinstance(v, TVal) and v.rank == 0:
                return v
            raise AnvilError(f"argument `{p.name}` of `{f.name}` must be a scalar, found {describe(v)}", node.span)
        if isinstance(v, EVal):
            return v
        if isinstance(v, (CVal, AffVal)) and len(t.dims) == 0:
            return v
        if not isinstance(v, TVal):
            raise AnvilError(f"argument `{p.name}` of `{f.name}` must be a tensor, found {describe(v)}", node.span)
        self.unify_shape(t, v.shape, f, shape_vars, node, f"argument `{p.name}`")
        if v.dtype != dtype:
            if dtype == F32 and v.dtype == I32 and t.dtype is None:
                return self.to_tensor(v, node.span, F32)
            raise AnvilError(f"argument `{p.name}` of `{f.name}` must be {dtype}, found {v.dtype}", node.span)
        return v

    def unify_shape(self, t: A.TypeExpr, shape, f: FnVal, shape_vars, node, what):
        if len(t.dims) != len(shape):
            raise AnvilError(f"{what} of `{f.name}` must have rank {len(t.dims)} "
                           f"({self.fmt_type(t)}), but has shape {fmt_shape(shape)}", node.span)
        for d_ast, actual in zip(t.dims, shape):
            if isinstance(d_ast, A.Name) and not self.fixed_dim(d_ast.id, f.scope):
                nm = d_ast.id
                if nm in shape_vars and shape_vars[nm] != actual:
                    raise AnvilError(f"shape mismatch in call to `{f.name}`: `{nm}` is {shape_vars[nm]} from an "
                                   f"earlier argument but {actual} in {what}", node.span,
                                   notes=[f"`{f.name}` expects {what} : {self.fmt_type(t)}"])
                if nm in shape_vars and dims.conflict(shape_vars[nm], actual):
                    raise AnvilError(f"shape mismatch in call to `{f.name}`: `{nm}` is `{dims.show(shape_vars[nm])}` "
                                   f"from an earlier argument but `{dims.show(actual)}` in {what}", node.span,
                                   notes=[dims.mismatch_note(shape_vars[nm], actual),
                                          f"`{f.name}` expects {what} : {self.fmt_type(t)}"])
                if nm not in shape_vars or isinstance(actual, dims.Dim):
                    shape_vars[nm] = actual
            else:
                want = self.eval_dim(d_ast, f.scope, shape_vars)
                if want != actual:
                    raise AnvilError(f"shape mismatch in call to `{f.name}`: {what} should be "
                                   f"{self.fmt_type(t, shape_vars)} but has shape {fmt_shape(shape)}", node.span)
                if dims.conflict(want, actual):
                    raise AnvilError(f"shape mismatch in call to `{f.name}`: {what} should be "
                                   f"{self.fmt_type(t)} but has shape {dims.show_shape(shape)}", node.span,
                                   notes=[dims.mismatch_note(want, actual)], help=dims.same_size_help(want, actual))

    @staticmethod
    def fixed_dim(name: str, scope: Scope) -> bool:
        """In a signature, a dimension name is fixed only if it names a constant (`const`,
        a model argument, or an enclosing shape variable); otherwise it is a shape variable."""
        b = scope.lookup(name)
        return b is not None and b.what == "const"

    def fmt_type(self, t: A.TypeExpr, shape_vars=None) -> str:
        def dim(d):
            if isinstance(d, A.Name):
                if shape_vars and d.id in shape_vars:
                    return str(shape_vars[d.id])
                return d.id
            if isinstance(d, A.Num):
                return str(d.value)
            return d.span.text
        dims = "" if t.dims is None else "[" + ", ".join(dim(d) for d in t.dims) + "]"
        return (t.dtype or "") + dims if (t.dtype or dims) else "f32"

    def check_return_type(self, t: A.TypeExpr, v: Val, f: FnVal, shape_vars, node) -> Val:
        dtype = t.dtype or F32
        if t.dims is None or len(t.dims) == 0:
            if isinstance(v, TVal) and v.rank != 0:
                raise AnvilError(f"`{f.name}` should return a scalar but returns shape {fmt_shape(v.shape)}", node.span)
            return v
        if not isinstance(v, TVal):
            if isinstance(v, EVal):
                return v
            raise AnvilError(f"`{f.name}` should return {self.fmt_type(t)} but returns a {v.kind}", node.span)
        self.unify_shape(t, v.shape, f, shape_vars, node, "the return value")
        if v.dtype != dtype and not (t.dtype is None):
            raise AnvilError(f"`{f.name}` should return {dtype} but returns {v.dtype}", node.span)
        return v

    # ------------------------------------------------------------------ models
    def model_body(self, stmts):
        """A model's declarations; an `if` on a compile-time condition picks which ones
        (`if DEPTH == 2: l2 = Linear(H, H)`)."""
        for st in stmts:
            if isinstance(st, (A.ParamDecl, A.FnDecl, A.ConstDecl, A.Assign, A.ModelDecl, A.Pass)):
                self.exec_stmt(st)
            elif isinstance(st, A.If):
                cond = self.eval_top(st.cond)
                if not isinstance(cond, CVal):
                    raise AnvilError("an `if` in a model body needs a condition known at compile time", st.cond.span,
                                   help="it decides which layers the model has, e.g. `if DEPTH == 2:`")
                self.model_body(st.body if cond.value else st.orelse)
            else:
                raise AnvilError("a model body may only contain params, fns, consts, sub-models, and `if` on a "
                               "compile-time condition", st.span)

    def instantiate(self, m: ModelDefVal, node: A.Call, name: str | None) -> ModelInstVal:
        decl: A.ModelDecl = m.decl
        if self.loops or self.fn_stack:
            raise AnvilError("models can only be created at the top level (not inside a loop or function)", node.span,
                           help="create it once at the top level and reuse it")
        if name is None:
            name = f"{self.scope.prefix}{decl.name.id}{len(self.instances) + 1}"
        args = [self.eval(a) for a in node.args]
        kwargs = {k.name: self.eval(k.value) for k in node.kwargs}
        bound = self.bind_args(decl.params, args, kwargs, node, f"model `{decl.name.id}`", m.scope)
        scope = Scope(parent=m.scope, kind="model", prefix=name + ".")
        for p in decl.params:
            v = bound[p.name]
            if not isinstance(v, (CVal, SVal, TupleVal, FnVal, BuiltinVal, ModelDefVal)):
                raise AnvilError(f"model arguments must be known at compile time; `{p.name}` is a {v.kind}", node.span)
            scope.vars[p.name] = Binding(v, what="const")
        scope.runtime_names = compute_runtime_names(decl.body)
        inst = ModelInstVal(name, scope, decl)
        saved = self.scope
        self.scope = scope
        try:
            self.model_body(decl.body)
        finally:
            self.scope = saved
        self.instances.append(inst)
        return inst

    def model_params(self, inst: ModelInstVal) -> list[Buffer]:
        out = []
        for b in inst.scope.vars.values():
            if b.ref is not None and b.what == "param":
                out.append(b.ref)
            elif isinstance(b.val, ModelInstVal) and b.val is not inst:
                out.extend(self.model_params(b.val))
            elif is_model_list(b.val):
                for m in b.val.items:
                    out.extend(self.model_params(m))
        return out

    def params_of(self, v: Val) -> list[Buffer] | None:
        """The parameters of a model or of a list of models; None for anything else."""
        if isinstance(v, ModelInstVal):
            return self.model_params(v)
        if is_model_list(v):
            return [b for m in v.items for b in self.model_params(m)]
        return None

    # ------------------------------------------------------------------ optimizers
    def optimizer_spec(self, f: OptDefVal, args, kwargs, node) -> OptSpecVal:
        decl: A.OptimizerDecl = f.decl
        bound = self.bind_args(decl.params, args, kwargs, node, f"optimizer `{decl.name.id}`", f.scope)
        return OptSpecVal(f, bound, node.span)

    # ------------------------------------------------------------------ index notation
    def scan_indices(self, node: A.Expr, lhs_names: list[str], where_names: list[str]):
        """Find index names (unbound names used in subscripts) and bind each to the
        innermost reduction enclosing all of its uses."""
        candidates = dict.fromkeys(list(lhs_names) + list(where_names))   # in order of appearance:
        # the order of a reduction's indices decides its loop order, which must not vary from run to run

        def is_unbound(name):
            if name in lhs_names or name in where_names:
                return True
            if self.comp is not None and name in self.comp.names:
                return False
            return self.scope.lookup(name) is None and name not in BUILTINS and name not in BUILTIN_CONSTS

        def find(n, in_sub):
            if isinstance(n, A.Name):
                if in_sub and is_unbound(n.id):
                    candidates[n.id] = None
                return
            if isinstance(n, A.EllipsisIdx):
                if in_sub:
                    candidates["..."] = None
                return
            if isinstance(n, A.Subscript):
                find(n.value, in_sub)
                for it in n.items:
                    find(it, True)
                return
            if isinstance(n, A.Slice):
                for x in (n.start, n.stop, n.step):
                    if x is not None:
                        find(x, in_sub)
                return
            if isinstance(n, A.Str):
                for p in n.parts:
                    if isinstance(p, A.Interp):
                        find(p.expr, in_sub)
                return
            for k, v in vars(n).items():
                if k == "span":
                    continue
                if isinstance(v, A.Node):
                    find(v, in_sub)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, A.Node):
                            find(x, in_sub)
        find(node, False)
        if not candidates:
            return {}, {}
        # occurrences with reduction stacks
        occ: dict[str, list[tuple]] = {c: [] for c in candidates}

        def is_reduction(n):
            if isinstance(n, A.Reduce):
                return True
            return (isinstance(n, A.Call) and isinstance(n.func, A.Name) and n.func.id in REDUCTIONS
                    and len(n.args) == 1 and not n.kwargs and self.scope.lookup(n.func.id) is None)

        def walk(n, stack):
            if isinstance(n, A.Name):
                if n.id in occ:
                    occ[n.id].append(tuple(stack))
                return
            if isinstance(n, A.EllipsisIdx):
                if "..." in occ:
                    occ["..."].append(tuple(stack))
                return
            if is_reduction(n):
                operand = n.operand if isinstance(n, A.Reduce) else n.args[0]
                walk(operand, stack + [id(n)])
                return
            if isinstance(n, A.Str):
                for p in n.parts:
                    if isinstance(p, A.Interp):
                        walk(p.expr, stack)
                return
            for k, v in vars(n).items():
                if k == "span":
                    continue
                if isinstance(v, A.Node):
                    walk(v, stack)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, A.Node):
                            walk(x, stack)
        walk(node, [])
        binders: dict[int, list[str]] = {}
        lhs_set = set(lhs_names)
        for name, stacks in occ.items():
            if name in lhs_set:
                continue
            if not stacks:
                continue
            common = list(stacks[0])
            for st in stacks[1:]:
                k = 0
                while k < len(common) and k < len(st) and common[k] == st[k]:
                    k += 1
                common = common[:k]
            if not common:
                if name == "...":
                    raise AnvilError("`...` is used in an index expression but is not on the left-hand side",
                                   node.span, help="write `y[...] = ...` or reduce over it with `sum`")
                raise AnvilError(f"index `{name}` is not bound", self.find_name_span(node, name) or node.span,
                               label="not on the left-hand side and not inside a reduction",
                               help=f"put it on the left (`y[..., {name}] = ...`) or reduce over it "
                                    f"(`sum ...`)" + (self.did_you_mean(name)))
            binders.setdefault(common[-1], []).append(name)
        return {c: None for c in candidates}, binders

    def did_you_mean(self, name):
        s = suggest(name, self.scope.all_names())
        return f"; or did you mean the variable `{s}`?" if s else ""

    def find_name_span(self, node, name):
        if isinstance(node, A.Name) and node.id == name:
            return node.span
        for k, v in vars(node).items():
            if k == "span":
                continue
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                if isinstance(x, A.Node):
                    r = self.find_name_span(x, name)
                    if r is not None:
                        return r
        return None

    def eval_top(self, node: A.Expr) -> Val:
        """Evaluate an expression statement-level: sets up index notation if needed."""
        names, binders = self.scan_indices(node, [], [])
        if not names:
            return self.eval(node)
        val, ctx = self.eval_comprehension(node, [], [], node.span, names_binders=(names, binders))
        if isinstance(val, EVal):
            return val
        return val

    def eval_comprehension(self, value: A.Expr, lhs_items, where, span, names_binders=None):
        lhs_names = [it.id for it in lhs_items if isinstance(it, A.Name)]
        has_lhs_ellipsis = any(isinstance(it, A.EllipsisIdx) for it in lhs_items)
        where_names = [w[0].id for w in where]
        if names_binders is None:
            names, binders = self.scan_indices(value, lhs_names + (["..."] if has_lhs_ellipsis else []),
                                               where_names)
        else:
            names, binders = names_binders
        ctx = CompCtx(names={}, binders=binders)
        ctx.lhs_items = lhs_items
        for nm in list(names) + lhs_names + where_names:
            if nm == "..." or nm in ctx.names:
                continue
            ctx.names[nm] = Var(nm)
        ctx.ellipsis_is_index = "..." in names or has_lhs_ellipsis
        ctx.ellipsis_on_lhs = has_lhs_ellipsis
        for w_name, w_bound in where:
            v = self.eval(w_bound)
            if not isinstance(v, CVal) or not v.is_int:
                raise AnvilError("`where` bounds must be compile-time integers", w_bound.span)
            ctx.where[ctx.names[w_name.id]] = v.value if isinstance(v.value, dims.Dim) else int(v.value)
        self.comps.append(ctx)
        try:
            val = self.eval(value)
            if has_lhs_ellipsis and ctx.ellipsis is None:
                raise AnvilError("the left-hand side uses `...` but the right-hand side never does", span)
            ctx.lhs = self.lhs_vars(ctx)
            self.solve_ranges(ctx, span)
            expr = None
            if isinstance(val, (EVal, IdxVal)):
                expr = self.to_expr(val, span)
            elif isinstance(val, (CVal, AffVal)) or (isinstance(val, TVal) and val.rank == 0):
                expr = self.scalar_expr(val, span)
            final = self.finalize(ctx, expr, span)
        finally:
            self.comps.pop()
        if expr is not None:
            free = _collect_free_index_vars(final) - set(ctx.lhs)
            own = set(ctx.names.values()) | set(ctx.ellipsis or [])
            bad = [v for v in free if v in own]
            if bad:
                raise AnvilError(f"index `{bad[0].name}` is not bound by the left-hand side or a reduction", span)
            return EVal(final), ctx
        return val, ctx

    def lhs_vars(self, ctx: CompCtx) -> list:
        out = []
        for it in ctx.lhs_items:
            if isinstance(it, A.EllipsisIdx):
                out.extend(ctx.ellipsis or [])
            else:
                out.append(ctx.names[it.id])
        return out

    def reduction(self, op: str, operand: A.Expr, node, span) -> Val:
        ctx = self.comp
        names = ctx.binders.get(id(node), []) if ctx is not None else []
        if not names:
            v = self.eval(operand)
            if isinstance(v, (EVal, IdxVal)):
                raise AnvilError(f"`{op}` has no index to reduce over", span,
                               help="every index inside it is already bound by the left-hand side "
                                    "(or refers to an existing variable)")
            return self.reduce_tensor(op, v, None, False, span)
        rvars = [ctx.names[nm] for nm in names if nm != "..."]
        ell_reduced = "..." in names
        mark = len(ctx.bound)
        ctx.bound.extend(rvars)
        if ell_reduced:
            ctx.ellipsis_reducer = id(node)
        try:
            body_val = self.eval(operand)
        finally:
            added = ctx.bound[mark:]
            del ctx.bound[mark:]
        if ell_reduced:
            if not ctx.ellipsis:
                raise AnvilError("`...` inside this reduction does not stand for any dimension", span)
            rvars = rvars + [v for v in added if v in ctx.ellipsis]
        if isinstance(body_val, TVal) and body_val.rank > 0:
            raise AnvilError(f"cannot reduce over indices of a whole tensor of shape {fmt_shape(body_val.shape)}",
                           span, help="index the tensor with the reduction's indices")
        body = self.to_expr(body_val, span)
        if op in ("argmax", "argmin") and len(rvars) != 1:
            raise AnvilError(f"`{op}` must reduce exactly one index", span)
        body_vars = _collect_free_index_vars(body)
        order = self.lhs_vars(ctx) + ctx.bound
        domain = [v for v in order if v in body_vars and v not in rvars]
        domain = list(dict.fromkeys(domain))
        foreign = [v for v in body_vars if v not in rvars and v not in order]
        if foreign:
            raise AnvilError("a reduction here uses an index from an enclosing expression", span,
                           help="move the reduction into its own definition")
        if op == "mean" and body.dtype == I32:
            body = cast(body, F32)
        dtype = I32 if op in ("argmax", "argmin") else (F32 if op == "mean" else body.dtype)
        temp = PendingTemp(domain, rvars, op, body, dtype, span, self.current_label())
        ctx.pending.append(temp)
        return EVal(PendingLoad(temp, [Affine.of(v) for v in domain]))

    def solve_ranges(self, ctx: CompCtx, span):
        own = list(ctx.names.values()) + list(ctx.ellipsis or [])
        own_set = set(own)
        exact: dict[Var, list] = {}
        for (aff, size, sp, desc) in ctx.constraints:
            v = aff.single_var()
            if v is not None and v in own_set:
                exact.setdefault(v, []).append((size, sp, desc))
        for v in own:
            if v.extent is not None:
                continue
            if v in ctx.where:
                set_extent(v, ctx.where[v])
                for (size, sp, desc) in exact.get(v, []):
                    if size != v.extent:
                        raise AnvilError(f"index `{v.name}` is declared `< {v.extent}` but indexes a dimension of "
                                       f"size {size}", sp, label=desc)
                    if dims.conflict(ctx.where[v], size):
                        raise AnvilError(f"index `{v.name}` is declared `< {dims.show(ctx.where[v])}` but indexes a "
                                       f"dimension of size `{dims.show(size)}`", sp, label=desc,
                                       notes=[dims.mismatch_note(ctx.where[v], size)],
                                       help=dims.same_size_help(ctx.where[v], size))
                continue
            if v in exact:
                (size0, sp0, desc0) = exact[v][0]
                for (size, sp, desc) in exact[v][1:]:
                    if size != size0:
                        raise AnvilError(f"index `{v.name}` ranges over {size0} in one place but {size} in another",
                                       sp, label=f"{desc}: size {size}",
                                       labels=[ir_label(sp0, f"{desc0}: size {size0}")],
                                       notes=["every use of an index must agree on its range (this is the shape check)"])
                    if dims.conflict(size0, size):
                        raise AnvilError(f"index `{v.name}` ranges over `{dims.show(size0)}` in one place but "
                                       f"`{dims.show(size)}` in another", sp,
                                       label=f"{desc}: size {dims.show(size)}",
                                       labels=[ir_label(sp0, f"{desc0}: size {dims.show(size0)}")],
                                       notes=[dims.mismatch_note(size0, size)],
                                       help=dims.same_size_help(size0, size))
                # the range takes a name from any use that has one
                set_extent(v, next((sz for sz, _, _ in exact[v] if isinstance(sz, dims.Dim)), size0))
        # bounded (affine) constraints
        changed = True
        while changed:
            changed = False
            for v in own:
                if v.extent is not None:
                    continue
                best = None
                for (aff, size, sp, desc) in ctx.constraints:
                    c = aff.coef(v)
                    if c <= 0:
                        continue
                    rest = aff - Affine(0, {v: c})
                    if rest.scalars() or rest.gathers():
                        continue
                    if any(u.extent is None for u in rest.direct_vars()):
                        continue
                    lo, hi = rest.var_bounds()
                    if lo < 0:
                        continue
                    m = (size - 1 - hi) // c + 1
                    best = m if best is None else min(best, m)
                if best is not None:
                    if best <= 0:
                        raise AnvilError(f"index `{v.name}` has an empty range here", span)
                    set_extent(v, best)
                    changed = True
        for v in own:
            if v.extent is None:
                raise AnvilError(f"cannot infer the range of index `{v.name}`", span,
                               help=f"use it as a plain subscript somewhere, or add `where {v.name} < N`")
        # verify all constraints
        for (aff, size, sp, desc) in ctx.constraints:
            if any(u.extent is None for u in aff.direct_vars()):
                continue
            lo, hi = aff.var_bounds()
            scal = aff.scalars()
            if aff.gathers():
                continue
            if scal:
                part = Affine(0, {k: c for k, c in aff.terms.items() if isinstance(k, ScalarRef)})
                ctx.checks.append(ir.Check(part, -lo, size - 1 - hi,
                                           f"index out of bounds in {desc} (size {size})", sp))
                continue
            if lo < 0 or hi > size - 1:
                bad = lo if lo < 0 else hi
                raise AnvilError(f"index out of bounds: `{ir.fmt_affine(aff)}` can be {bad}, but the dimension has size "
                               f"{size}", sp, label=desc,
                               help="restrict the index range with `where`, or pad the input")

    def finalize(self, ctx: CompCtx, expr: Expr | None, span):
        for c in ctx.checks:
            self.emit(c)

        def resolve(e: Expr) -> Expr:
            def f(x):
                if isinstance(x, PendingLoad):
                    buf = x.temp.buf
                    off = Affine(0)
                    for a, s in zip(x.idx, row_major_strides(buf.shape)):
                        off = off + a * s
                    return Load(buf, off)
                return x
            return ir.map_expr(e, f)

        for t in ctx.pending:
            shape = tuple(v.size for v in t.domain)
            body = resolve(t.body)
            op = t.op
            if op == "mean":
                n = ir.prod(v.extent for v in t.red_vars)
                body = mk_binary("mul", Const(1.0 / n), cast(body, F32))
                op = "sum"
            buf = self.new_temp(shape, t.dtype)
            t.buf = buf
            self.emit_kernel(self.make_kernel(t.domain, buf, body, red_vars=t.red_vars, red_op=op,
                                              span=self.current_span(), label=t.label or op))
        return resolve(expr) if expr is not None else None

    # ------------------------------------------------------------------ subscripts
    def subscript(self, base: TVal, items, node) -> Val:
        ctx = self.comp
        # expand ellipsis
        n_ell = sum(1 for it in items if isinstance(it, A.EllipsisIdx))
        if n_ell > 1:
            raise AnvilError("only one `...` is allowed in a subscript", node.span)
        explicit = [it for it in items if not isinstance(it, A.EllipsisIdx)]
        if len(explicit) > base.rank:
            raise AnvilError(f"too many indices: tensor has rank {base.rank} ({fmt_shape(base.shape)}) but "
                           f"{len(explicit)} were given", node.span)
        vals: list = []
        for it in items:
            if isinstance(it, A.EllipsisIdx):
                k = base.rank - len(explicit)
                if ctx is not None and ctx.ellipsis_is_index:
                    if ctx.ellipsis is None:
                        ctx.ellipsis = [Var(f"e{d}") for d in range(k)]
                        if not ctx.ellipsis_on_lhs:
                            ctx.bound.extend(ctx.ellipsis)     # bound by the enclosing reduction
                    if len(ctx.ellipsis) != k:
                        raise AnvilError(f"`...` stands for {len(ctx.ellipsis)} dimension(s) elsewhere but "
                                       f"{k} here", it.span)
                    vals.extend(IdxVal(Affine.of(v)) for v in ctx.ellipsis)
                else:
                    vals.extend(("slice", None, None, None, it.span) for _ in range(k))
            elif isinstance(it, A.Slice):
                vals.append(("slice", it.start, it.stop, it.step, it.span))
            else:
                vals.append(self.eval(it))
        while len(vals) < base.rank:
            vals.append(("slice", None, None, None, node.span))
        return self.subscript_value(base, vals, node.span, item_nodes=items)

    def subscript_value(self, base: TVal, vals, span, item_nodes=None) -> Val:
        """vals: per-dimension Val or ('slice', start, stop, step, span)."""
        index_level = any(isinstance(v, (IdxVal, EVal)) for v in vals)
        if index_level and any(isinstance(v, tuple) for v in vals):
            raise AnvilError("mixing index variables and slices in one subscript is not supported", span,
                           help="give an index for every dimension, or use `...` for the leading ones")
        ctx = self.comp
        if index_level:
            idx = []
            desc = self.describe_subscript(base, item_nodes)
            for d, v in enumerate(vals):
                size = base.shape[d]
                a = self.index_affine(v, size, span, desc)
                if a is None:
                    raise AnvilError("unsupported index expression", span)
                idx.append(a)
                if ctx is not None and (a.direct_vars() or a.scalars()) and not a.gathers():
                    ctx.constraints.append((a, size, span, f"`{desc}` dimension {d}"))
            return EVal(base.load(idx))
        # tensor-level: build a view
        new_shape = []
        maps = []      # per base dim: ('var', new_dim_index, start Affine, step) | ('fixed', Affine)
        gather_dims = []
        for d, v in enumerate(vals):
            size = base.shape[d]
            if isinstance(v, tuple):
                _, start, stop, step, sp = v
                st, ln, stp = self.slice_bounds(start, stop, step, size, sp)
                maps.append(("var", len(new_shape), st, stp))
                new_shape.append(ln)
            elif isinstance(v, CVal):
                if not v.is_int or isinstance(v.value, bool):
                    raise AnvilError(f"indices must be integers, found {v.value!r}", span)
                i = v.value + size if v.value < 0 else v.value
                if not 0 <= i < size:
                    raise AnvilError(f"index {v.value} is out of bounds for a dimension of size {size}", span)
                maps.append(("fixed", Affine(i)))
            elif isinstance(v, (AffVal,)) or (isinstance(v, TVal) and v.rank == 0 and v.dtype == I32):
                a = self.runtime_index(v, span)
                self.emit(ir.Check(a, 0, size - 1, f"index out of bounds (size {size})", span))
                maps.append(("fixed", a))
            elif isinstance(v, TVal) and v.dtype == I32:
                if gather_dims:
                    raise AnvilError("only one integer-tensor index is supported per subscript", span)
                gather_dims.append(d)
                maps.append(("gather", len(new_shape), v))
                new_shape.extend(v.shape)
            elif isinstance(v, TVal):
                raise AnvilError(f"indices must be integers, found a {v.dtype} tensor", span)
            else:
                raise AnvilError(f"cannot index with a {v.kind}", span)
        pv = [Var(f"p{d}", n) for d, n in enumerate(new_shape)]
        idx = []
        for m in maps:
            if m[0] == "var":
                _, nd, st, stp = m
                idx.append(st + Affine(0, {pv[nd]: stp}))
            elif m[0] == "fixed":
                idx.append(m[1])
            else:
                _, nd, iv = m
                g = Gather(Load(iv.buf, iv.offset([Affine.of(p) for p in pv[nd:nd + iv.rank]])),
                           base.shape[len(idx)], "integer-tensor index")
                idx.append(Affine(0, {g: 1}))
        tmpl = base.offset(idx)
        return TVal(base.buf, tuple(new_shape), tmpl, pv, base.detached)

    def describe_subscript(self, base, item_nodes):
        name = base.buf.name
        if item_nodes:
            return f"{name}[{', '.join(it.span.text for it in item_nodes)}]"
        return name

    def index_affine(self, v, size, span, desc) -> Affine | None:
        if isinstance(v, IdxVal):
            return v.affine
        if isinstance(v, CVal):
            if not v.is_int or isinstance(v.value, bool):
                raise AnvilError(f"indices must be integers, found {v.value!r}", span)
            i = v.value + size if v.value < 0 else v.value
            if not 0 <= i < size:
                raise AnvilError(f"index {v.value} is out of bounds for a dimension of size {size}", span)
            return Affine(i)
        if isinstance(v, AffVal):
            return v.affine
        if isinstance(v, TVal) and v.rank == 0 and v.dtype == I32:
            return self.runtime_index(v, span)
        if isinstance(v, EVal):
            e = v.expr
            if e.dtype != I32:
                raise AnvilError("index expressions must be integers", span,
                               help="convert with `i32(...)`")
            if isinstance(e, Load):
                return Affine(0, {Gather(e, size, desc): 1})
            if isinstance(e, Index):
                return e.affine
            split = _wrap_gather(split_gather(e))
            if split is not None:                    # `data[starts[b] + t]`: a gathered value plus an offset
                ld, rest = split
                if rest.scalars() or rest.gathers():
                    raise AnvilError("an index can add index names and constants to one tensor element", span)
                return Affine(0, {Gather(ld, size, desc, rest): 1}) + rest
            raise AnvilError("an index must be an affine expression of indices or a single integer tensor "
                           "element", span)
        if isinstance(v, TVal):
            raise AnvilError(f"cannot use a tensor of shape {fmt_shape(v.shape)} as an index inside index notation",
                           span)
        return None

    def runtime_index(self, v, span) -> Affine:
        if isinstance(v, AffVal):
            return v.affine
        if isinstance(v, TVal):
            if not v.is_identity():
                v = self.materialize(v, span)
            return Affine.of(ScalarRef(v.buf))
        raise AnvilError("expected an integer index", span)

    def slice_bounds(self, start, stop, step, size, span):
        named = {}                                    # the bounds' values, with their names (dims.py)

        def ev(x):
            if x is None:
                return None
            v = self.eval(x)
            if isinstance(v, CVal):
                if not v.is_int:
                    raise AnvilError("slice bounds must be integers", x.span)
                named[id(x)] = v.value
                return Affine(int(v.value))
            if isinstance(v, AffVal):
                return v.affine
            if isinstance(v, TVal) and v.rank == 0 and v.dtype == I32:
                return self.runtime_index(v, x.span)
            raise AnvilError(f"slice bounds must be integers, found {describe(v)}", x.span)
        st = 1
        if step is not None:
            sv = self.eval(step)
            if not isinstance(sv, CVal) or not sv.is_int or sv.value <= 0:
                raise AnvilError("slice steps must be positive compile-time integers", step.span)
            st = int(sv.value)
        a = ev(start)
        b = ev(stop)
        if a is not None and a.is_const() and a.const < 0:
            a = Affine(a.const + size)
        if b is not None and b.is_const() and b.const < 0:
            b = Affine(b.const + size)
        if a is None:
            a = Affine(0)
        if b is None:
            b = Affine(size)
        if a.is_const() and b.is_const():
            lo = min(max(a.const, 0), size)
            hi = min(max(b.const, 0), size)
            ln = max(0, (hi - lo + st - 1) // st)
            if st == 1 and 0 <= a.const <= b.const <= size:  # `x[:, 1:T + 1]` has length T
                sym = (named.get(id(stop), size) if stop is not None else size) - (named.get(id(start), 0) if start is not None else 0)
                if isinstance(sym, dims.Dim) and int(sym) == ln:
                    ln = sym
            return Affine(lo), ln, st
        diff = b - a
        if not diff.is_const():
            raise AnvilError("the length of a slice must be known at compile time", span,
                           help="write the stop as `start + length`, e.g. `x[s : s + 64]`")
        ln = max(0, (diff.const + st - 1) // st)
        self.emit(ir.Check(a, 0, size - ln * st if ln else size, f"slice out of bounds (size {size})", span))
        return a, ln, st

    # ------------------------------------------------------------------ literals
    def tensor_literal(self, n: A.ListLit) -> Val:
        if n.items and all(isinstance(x, A.Str) for x in n.items):
            vals = [self.eval(x) for x in n.items]
            if all(isinstance(v, SVal) for v in vals):
                return TupleVal(vals)               # a list of strings (e.g. glyphs for `show`)
        try:
            return self.const_literal(n)
        except _NotConstant:
            return self.runtime_literal(n)

    def runtime_literal(self, n: A.ListLit) -> Val:
        """`[a, b, c]` with run-time values: one kernel that stores each element."""
        leaves = []          # (position tuple, Val)

        def walk(x, pos):
            if isinstance(x, A.ListLit):
                shapes = set()
                for k, item in enumerate(x.items):
                    shapes.add(walk(item, pos + (k,)))
                if len(shapes) > 1:
                    raise AnvilError("ragged tensor literal: all rows must have the same shape", x.span)
                return (len(x.items),) + (next(iter(shapes)) if shapes else ())
            v = self.eval(x)
            if isinstance(v, EVal):
                v = self.to_tensor(v, x.span)
            if not isinstance(v, (CVal, AffVal, TVal)):
                raise AnvilError(f"a tensor literal cannot contain a {v.kind}", x.span)
            leaves.append((pos, v, x.span))
            return self.value_type(v, x.span)[0] if not isinstance(v, TVal) else v.shape
        shape = walk(n, ())
        outer = len(leaves[0][0]) if leaves else 0
        inner = shape[outer:]
        dtype = F32 if any((v.dtype if not isinstance(v, CVal) else v.dtype) == F32 for _, v, _ in leaves) else I32
        buf = self.new_temp(shape, dtype)
        strides = row_major_strides(shape)
        vars_ = [Var(f"i{d}", s) for d, s in enumerate(inner)]
        stmts = []
        for pos, v, sp in leaves:
            base = sum(p * s for p, s in zip(pos, strides))
            off = Affine(base) + Affine(0, {vv: s for vv, s in zip(vars_, strides[outer:])})
            e = self.bload(v, vars_, inner, sp) if isinstance(v, TVal) else self.scalar_expr(v, sp)
            stmts.append(ir.Store(buf, off, cast(e, dtype)))
        k = self.make_kernel(vars_, buf, stmts[0].value, span=n.span, label="literal", out_offset=stmts[0].offset)
        m = {old: Affine.of(new) for old, new in zip(vars_, k.domain)}
        k.stmts = [ir.Store(st.buf, st.offset.subst(m), ir.subst_expr(st.value, m)) for st in stmts]
        self.emit_kernel(k)
        return TVal.of(buf)

    def const_literal(self, n: A.ListLit) -> Val:
        def walk(x):
            if isinstance(x, A.ListLit):
                items = [walk(i) for i in x.items]
                shapes = {s for s, _ in items}
                if len(shapes) > 1:
                    raise AnvilError("ragged tensor literal: all rows must have the same shape", x.span)
                inner = next(iter(shapes)) if items else ()
                return (len(items),) + inner, [v for _, vs in items for v in vs]
            v = self.eval(x)
            if not isinstance(v, CVal):
                raise _NotConstant()
            return (), [v.value]
        shape, flat = walk(n)
        is_int = all(isinstance(v, (int, bool)) and not isinstance(v, float) for v in flat)
        dtype = I32 if is_int and flat else F32
        self.temp_counter += 1
        buf = self.new_buffer(f"lit{self.temp_counter}", shape, dtype, "const",
                              init=[int(v) if dtype == I32 else float(v) for v in flat], span=n.span)
        return TVal.of(buf)


def split_gather(e: Expr):
    """`ids[i] + t - 1` -> (the load ids[i], the affine part t - 1), or None."""
    if isinstance(e, Load) and e.dtype == I32:
        return e, Affine(0)
    if isinstance(e, Index):
        return None, e.affine
    if isinstance(e, Const) and e.dtype == I32:
        return None, Affine(int(e.value))
    if isinstance(e, Binary) and e.op in ("add", "sub"):
        a, b = split_gather(e.a), split_gather(e.b)
        if a is None or b is None or (a[0] is not None and b[0] is not None):
            return None
        if e.op == "sub":
            if b[0] is not None:
                return None
            return a[0], a[1] - b[1]
        return a[0] if a[0] is not None else b[0], a[1] + b[1]
    if isinstance(e, Binary) and e.op == "mul":
        for x, c in ((e.a, e.b), (e.b, e.a)):
            if isinstance(c, Const) and c.dtype == I32:
                s = split_gather(x)
                if s is not None and s[0] is None:
                    return None, s[1] * int(c.value)
    return None


def _wrap_gather(split):
    return split if split is not None and split[0] is not None else None


class _NotConstant(Exception):
    pass


def ir_label(span, msg):
    from .diagnostics import Label
    return Label(span, msg)
