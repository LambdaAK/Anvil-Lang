"""Anvil from Python: compile an Anvil function once per argument shape and call it on NumPy arrays.

    import anvil, numpy as np

    attention = anvil.function('''
    fn attention(q: [b, t, d], k: [b, t, d], v: [b, t, d]):
        s[n, i, j] = sum q[n, i, c] * k[n, j, c] / sqrt(d) + (0.0 if j <= i else -1e9)
        return softmax(s) @ v
    ''')
    out = attention(q, k, v)          # float32 arrays in, a float32 array out

The first call with a new combination of shapes compiles the function to a shared library of
native code (cached in ~/.cache/anvil, so a later process skips the compiler). The arrays are not
copied: the compiled code reads the arguments and writes the results in place, through pointers.

Arguments:
  - NumPy arrays (and lists) are tensors: floats become f32, integers and bools i32.
  - Python ints, floats, bools and strings are compile-time constants, baked into the code, so
    they can be shapes or loop bounds (`fn f(x, k) = ...` with `f(x, 3)`); a new value compiles
    a new version. Pass `np.float32(x)` for a run-time scalar instead.

The result is an array, a tuple of arrays (`return a, b`), or a Python number if the function's
result is known at compile time. The rest of the source (other functions, models, constants) is
available to the function. Top-level statements run on every call, so keep them to declarations.
A run-time error in the compiled code (an index out of bounds) ends the process, as it does in an
`anvil run` program.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
import threading

import numpy as np

from . import ast as A
from . import ir
from .diagnostics import AnvilError
from .ir import F32, I32, Affine, Buffer, KernelStmt
from .parser import parse
from .source import SourceFile


class Function:
    """An Anvil function callable from Python; see the module docstring."""

    def __init__(self, source: str, name: str | None = None, seed: int = 0, optimize: bool = True,
                 consts: dict | None = None):
        if "\n" not in source and source.endswith(".anvil") and os.path.exists(source):
            self.path = os.path.abspath(source)
            with open(source, encoding="utf-8") as f:
                source = f.read()
        else:
            self.path = os.path.join(os.getcwd(), "<python>.anvil")
        self.source = source if source.endswith("\n") else source + "\n"
        tree = parse(SourceFile(self.path, self.source))            # syntax errors show up right away
        fns = [s.name.id for s in tree.body if isinstance(s, A.FnDecl)]
        if name is None:
            if not fns:
                raise AnvilError("the source defines no function (`fn name(args) = …`)", None)
            name = fns[-1]
        elif name not in fns:
            raise AnvilError(f"the source has no function `{name}`", None,
                           help=f"its functions: {', '.join(fns)}" if fns else None)
        self.name = name
        self.seed = seed
        self.optimize = optimize
        self.consts = consts
        self.versions: dict[tuple, Compiled] = {}

    def __repr__(self):
        return f"<anvil.function {self.name}, {len(self.versions)} compiled version(s)>"

    def __call__(self, *args):
        tensors, sig = arguments(args)
        fn = self.versions.get(sig)
        if fn is None:
            fn = self.versions[sig] = self.compile_for(sig)
        return fn.call(tensors)

    def lower(self, *args) -> "Compiled":
        """The compiled version for these arguments (its .ir and .asm show what it does)."""
        _, sig = arguments(args)
        fn = self.versions.get(sig)
        if fn is None:
            fn = self.versions[sig] = self.compile_for(sig)
        return fn

    def compile_for(self, sig) -> "Compiled":
        lowered = lower(self.source, self.path, self.name, sig, seed=self.seed, optimize=self.optimize,
                        consts=self.consts)
        return Compiled(self.name, lowered, seed=self.seed)


BAKE_LIMIT = 64 << 20       # values baked into a compiled function (256 MB of f32)


class Lowered:
    """A function as a program of its own: what it reads from the rest of the source is baked in."""

    def __init__(self, prog, inputs, outputs, consts, tuple_result, name, arg_names, out_names):
        self.prog = prog
        self.inputs = inputs                 # input buffers, in argument order (tensors only)
        self.outputs = outputs               # output buffers (None for a compile-time result)
        self.consts = consts                 # the compile-time results
        self.tuple_result = tuple_result
        self.name = name
        self.arg_names = arg_names           # the tensor arguments' names in the source
        self.out_names = out_names


def lower(source: str, path: str, name: str, sig, seed: int = 0, optimize: bool = True,
          consts: dict | None = None) -> Lowered:
    """Compile `name` for the arguments in sig. The rest of the source (models, weights loaded with
    `load`, constants) runs once, here, on the reference interpreter; every tensor the function
    reads from it becomes constant data. The result computes only the function."""
    import io
    from .driver import load_prelude
    from .elaborate import Elaborator
    from .interp import Interpreter
    from .optimize import block_reads
    from .values import Binding, TupleVal, TVal
    names, inputs = [], []
    for i, a in enumerate(sig):
        if a[0] == "tensor":
            names.append(f"__anvil_arg{i}")
            inputs.append((f"__anvil_arg{i}", a[1], a[2]))
        else:
            names.append(a[1])                                   # a constant, as Anvil source
    src = SourceFile(path, source + f"__anvil_result = {name}({', '.join(names)})\n")
    tree = parse(src)
    call = tree.body.pop()                                       # the program first, then the call
    elab = Elaborator(source=src, seed=seed, consts=consts)
    in_bufs = []
    for k, (var, shape, dtype) in enumerate(inputs):
        b = elab.new_buffer(f"input{k}", shape, dtype, kind="input")
        elab.globals.vars[var] = Binding(TVal.of(b), what="argument")
        in_bufs.append(b)
    prog = elab.run(tree, load_prelude())
    n_init = len(prog.main.stmts)
    elab.exec_block([call])
    if elab.errors:
        from .diagnostics import AnvilErrors
        raise elab.errors[0] if len(elab.errors) == 1 else AnvilErrors(elab.errors)
    result = elab.globals.vars["__anvil_result"].val
    items = result.items if isinstance(result, TupleVal) else [result]
    outputs, consts = [], []
    for item in items:
        outputs.append(output_buffer(elab, prog, item, len([o for o in outputs if o is not None]), consts, name))
    init = ir.Block(prog.main.stmts[:n_init])
    body = ir.Block(prog.main.stmts[n_init:])
    fn_prog = ir.Program(body, prog.buffers, prog.source)
    check_inputs_unchanged(fn_prog, in_bufs, inputs)
    # run the program's own statements once, and bake what the function reads
    # what the function reads that comes from the program: written by its statements (or by nothing),
    # not the function's own temporaries
    from .optimize import block_writes
    made_by_program, made_by_function = block_writes(init), block_writes(body)
    needed = [b for b in block_reads(body) if isinstance(b, ir.Buffer)
              and b.kind not in ir.EXTERN_KINDS and b.kind != "const"
              and (b in made_by_program or b not in made_by_function)]
    if needed:
        it = Interpreter(ir.Program(init, prog.buffers, prog.source), out=io.StringIO(), seed=seed,
                         inp=io.StringIO(""))
        it.run()
        total = sum(b.numel for b in needed)
        if total > BAKE_LIMIT:
            raise AnvilError(f"`{name}` reads {total:,} values computed outside it (more than {BAKE_LIMIT:,}): "
                           f"pass large data as arguments instead", None)
        for b in needed:
            b.init = it.arr(b)[:b.numel].tolist()
            b.kind = "const"
    if optimize:
        from .optimize import optimize_program
        optimize_program(fn_prog)
    decl = next(s for s in tree.body if isinstance(s, A.FnDecl) and s.name.id == name)
    arg_names = [p.name for p, a in zip(decl.params, sig) if a[0] == "tensor"]
    out_names = [f"out{k}" if len(items) > 1 else "out" for k, o in enumerate(outputs) if o is not None]
    return Lowered(fn_prog, in_bufs, outputs, consts, isinstance(result, TupleVal), name, arg_names, out_names)


def output_buffer(elab, prog, val, k, consts, name):
    """A kernel copying the result into the caller's array (fused into whatever computes it)."""
    from .values import CVal, TVal
    if isinstance(val, CVal):
        consts.append(val.value)
        return None
    if not isinstance(val, TVal):
        raise AnvilError(f"`{name}` returns a {val.kind}; a compiled function must return tensors or numbers", None)
    out = elab.new_buffer(f"output{k}", val.shape, val.dtype, kind="output")
    vs = [ir.Var(f"i{d}", n) for d, n in enumerate(val.shape)]
    body = val.load([Affine.of(v) for v in vs])
    kern = elab.make_kernel(vs, out, body, label="result")
    kern.name = f"k{kern.id}"
    prog.main.stmts.append(KernelStmt(kern))
    return out


def check_inputs_unchanged(prog: ir.Program, in_bufs, inputs):
    """The arguments are the caller's arrays: the function must not write to them."""
    roots = {b.root.id: name for b, (name, _, _) in zip(in_bufs, inputs)}
    for k in ir.all_kernels(prog):
        for st in k.stores:
            if st.buf.root.id in roots:
                n = int(roots[st.buf.root.id].removeprefix("__anvil_arg"))
                raise AnvilError(f"the function writes to its argument {n + 1}, which is the caller's array",
                               k.span, help="copy it first, e.g. `y = copy(x)`")


def arguments(args):
    """NumPy arrays for the tensor arguments, and the signature that picks the compiled version."""
    tensors, sig = [], []
    for a in args:
        if isinstance(a, (bool, int, float, str)) and not isinstance(a, np.generic):
            text = ("true" if a else "false") if isinstance(a, bool) else \
                (f'"{a}"' if isinstance(a, str) else repr(a))
            if isinstance(a, float) and not np.isfinite(a):
                text = "inf" if a > 0 else ("-inf" if a < 0 else "nan")
            sig.append(("const", text))
            continue
        arr = np.asarray(a)
        if arr.dtype.kind == "f":
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            dtype = F32
        elif arr.dtype.kind in "iub":
            arr = np.ascontiguousarray(arr, dtype=np.int32)
            dtype = I32
        else:
            raise TypeError(f"anvil: cannot pass an array of {arr.dtype} (use floats or integers)")
        tensors.append(arr)
        sig.append(("tensor", tuple(int(d) for d in arr.shape), dtype))
    return tensors, tuple(sig)


class Compiled:
    """One version of a Function: a shared library for one combination of argument shapes."""

    def __init__(self, name, lowered: Lowered, seed: int):
        from .backend.aarch64 import generate
        from .cli import cache_dir
        from .backend.toolchain import find_cc
        import subprocess
        self.name = name
        self.prog = lowered.prog
        self.inputs = lowered.inputs
        self.outputs = lowered.outputs
        self.consts = lowered.consts
        self.tuple_result = lowered.tuple_result
        self.asm, gen = generate(self.prog, seed=seed, library=True)
        key = hashlib.sha256(self.asm.encode()).hexdigest()[:16]
        lib = os.path.join(cache_dir(), f"fn-{key}.dylib")
        if not os.path.exists(lib):
            with tempfile.TemporaryDirectory() as d:
                s = os.path.join(d, "fn.s")
                with open(s, "w") as f:
                    f.write(self.asm)
                tmp = os.path.join(d, "fn.dylib")
                cmd = [find_cc(), "-arch", "arm64", "-shared", "-x", "assembler", s, "-o", tmp, "-lz",
                       "-framework", "Accelerate"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"anvil: linking the function failed:\n{r.stderr}")
                os.replace(tmp, lib)
        self.lib = ctypes.CDLL(lib)
        self.entry = self.lib.anvil_entry
        self.entry.restype = ctypes.c_int
        self.entry.argtypes = []
        used = {b.id for b in gen.used.values()}
        # an argument the function never reads has no pointer slot
        self.in_slots = [ctypes.c_void_p.in_dll(self.lib, f"anvil_{b.name}") if b.id in used else None
                         for b in self.inputs]
        self.out_slots = [ctypes.c_void_p.in_dll(self.lib, f"anvil_{b.name}") if b is not None and b.id in used else None
                          for b in self.outputs]
        self.lock = threading.Lock()                    # the library's buffers are shared by every call

    @property
    def ir(self) -> str:
        return ir.fmt_program(self.prog)

    def call(self, arrays):
        results = []
        for b in self.outputs:
            results.append(None if b is None else np.empty(b.shape, np.float32 if b.dtype == F32 else np.int32))
        with self.lock:
            for slot, a in zip(self.in_slots, arrays):
                if slot is not None:
                    slot.value = a.ctypes.data
            for slot, r in zip(self.out_slots, results):
                if slot is not None:
                    slot.value = r.ctypes.data
            self.entry()
            for slot in self.in_slots + self.out_slots:
                if slot is not None:
                    slot.value = None
        consts = iter(self.consts)
        out = [next(consts) if r is None else (r[()] if r.ndim == 0 else r) for r in results]
        return tuple(out) if self.tuple_result else out[0]


def function(source: str, name: str | None = None, seed: int = 0) -> Function:
    """An Anvil function callable on NumPy arrays: `source` is Anvil code (or a path to a .anvil file),
    and `name` picks the function in it (by default the last one defined)."""
    return Function(source, name=name, seed=seed)
