"""Built-in functions."""
from __future__ import annotations

import gzip
import math
import os
import re
import struct

from . import ir
from .diagnostics import AnvilError, fmt_shape
from .dims import Dim
from .ir import F32, I32, Affine, Const, Index, Rand, row_major_strides
from .simplify import cast, mk_binary, mk_select, mk_unary
from .values import (AffVal, BatchesVal, CVal, DistVal, EVal, FStrVal, IdxVal, ModelInstVal, NoneVal,
                     PickVal, RangeVal, SVal, TextVal, TupleVal, TVal)

BUILTIN_CONSTS = {"inf": math.inf, "pi": math.pi, "nan": math.nan}

UNARY_MATH = ["exp", "log", "sqrt", "abs", "tanh", "sigmoid", "sin", "cos", "floor", "ceil", "round",
              "sign", "rsqrt"]

BUILTINS = set(UNARY_MATH) | {
    "detach", "f32", "i32", "max", "min", "sum", "mean", "prod", "argmax", "argmin", "pow", "where",
    "zeros", "ones", "full", "arange", "eye", "linspace", "randn", "rand", "reshape", "flatten",
    "transpose", "len", "copy", "print", "range", "batches", "idx", "seed", "clock", "grad",
    "normal", "uniform", "bernoulli", "randint", "shape", "float", "int", "show", "sleep", "nonzero", "stack",
    "input", "save", "load", "csv", "bytes", "decode", "npy", "save_npy", "safetensors",
}

CSV_NUMBER = re.compile(r"\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*")   # what a csv field may hold

IDX_TYPES = {0x08: ("u8", 1), 0x09: ("i8", 1), 0x0B: ("i16", 2), 0x0C: ("i32", 4), 0x0D: ("f32", 4),
             0x0E: ("f64", 8)}


# NumPy element types Anvil reads: code for the run-time conversion, and bytes per element
NPY_KINDS = {"f4": (0, 4), "f8": (1, 8), "f2": (2, 2), "i4": (3, 4), "u4": (3, 4), "i8": (4, 8), "u8": (4, 8),
             "u1": (5, 1), "b1": (5, 1), "i1": (6, 1), "u2": (7, 2), "i2": (8, 2)}


def read_npy_header(path: str):
    """(descr, shape, offset of the data) of a .npy file."""
    import ast as pyast
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not a .npy file")
        major = f.read(2)[0]
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        header = f.read(hlen).decode("latin1")
        offset = f.tell()
    d = pyast.literal_eval(header)
    if d.get("fortran_order"):
        raise ValueError("the array is in Fortran order (save it with np.ascontiguousarray first)")
    return d["descr"], tuple(int(x) for x in d["shape"]), offset


def npy_header(descr: str, shape) -> bytes:
    """The header of a version 1.0 .npy file holding a C-ordered array (padded to 64 bytes)."""
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {tuple(int(d) for d in shape)!r}, }}"
    pad = 64 - (10 + len(text) + 1) % 64
    text = text + " " * (pad % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(text).to_bytes(2, "little") + text.encode("latin1")


# safetensors element types, as the NumPy codes of NPY_KINDS
SAFETENSORS_TYPES = {"F32": "f4", "F64": "f8", "F16": "f2", "I32": "i4", "U32": "u4", "I64": "i8", "U64": "u8",
                     "I16": "i2", "U16": "u2", "I8": "i1", "U8": "u1", "BOOL": "b1"}


def read_safetensors_header(path: str) -> dict:
    """{tensor name: (dtype, shape, offset of its data in the file)} of a .safetensors file."""
    import json
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 2 <= n <= (100 << 20):
            raise ValueError("not a safetensors file")
        header = json.loads(f.read(n).decode("utf-8"))
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = (info["dtype"], tuple(int(x) for x in info["shape"]), 8 + n + start, end - start)
    return out


def read_idx_header(path: str):
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rb") as f:
        head = f.read(4)
        if len(head) < 4 or head[0] != 0 or head[1] != 0:
            raise ValueError("not an IDX file (bad magic number)")
        code, ndim = head[2], head[3]
        if code not in IDX_TYPES:
            raise ValueError(f"unsupported IDX element type 0x{code:02x}")
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    return IDX_TYPES[code][0], tuple(dims), 4 + 4 * ndim


def _is_gzip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def parse_spec(spec: str, dtype: str, span):
    """Python-ish format spec -> (printf conversion, scale)."""
    s = spec
    flags = ""
    if s and s[0] in "<>^":
        if s[0] == "<":
            flags += "-"
        s = s[1:]
    if s[:1] == "+":
        flags += "+"
        s = s[1:]
    if s[:1] == "0":
        flags += "0"
        s = s[1:]
    width = ""
    while s[:1].isdigit():
        width += s[0]
        s = s[1:]
    prec = ""
    if s[:1] == ".":
        s = s[1:]
        while s[:1].isdigit():
            prec += s[0]
            s = s[1:]
        prec = "." + prec
    typ = s or ""
    if typ not in ("", "f", "e", "g", "d", "%"):
        raise AnvilError(f"unsupported format spec `{spec}`", span, help="use [width][.precision][f|e|g|d|%]")
    scale = 1.0
    if typ == "%":
        return f"%{flags}{width}{prec or '.0'}f%%", 100.0, F32
    if typ == "d" or (typ == "" and dtype == I32 and not prec):
        return f"%{flags}{width}d", 1.0, I32
    if typ == "" and not prec and not width and not flags:
        return "", 1.0, F32          # automatic format (like tensor elements)
    if typ == "":
        typ = "f" if prec else "g"
    return f"%{flags}{width}{prec}{typ}", scale, F32


def describe_kind(v) -> str:
    return f"a {v.kind}" if not isinstance(v, TVal) else "a scalar"


class BuiltinsMixin:
    def call_builtin(self, name, args, kwargs, node):
        if name in UNARY_MATH:
            self.expect_args(name, args, kwargs, node, 1)
            return self.unary_op(name, args[0], node.span, label=name)
        fn = getattr(self, "bi_" + name, None)
        if fn is None:
            raise AnvilError(f"`{name}` cannot be called this way", node.span)
        return fn(args, kwargs, node)

    def data_dir(self, node) -> str:
        """Where a file name in a call is relative to: the .anvil file the call is written in (a file
        brought in with `use` reads its own data files), else the program's file, else here."""
        f = getattr(getattr(node, "span", None), "file", None)
        path = getattr(f, "path", "")
        if path and not path.startswith("<") and os.path.basename(path) != "prelude.anvil" and os.path.exists(path):
            return os.path.dirname(os.path.abspath(path))
        return self.source.dir if self.source is not None else os.getcwd()

    def expect_args(self, name, args, kwargs, node, n_min, n_max=None, allowed_kw=()):
        n_max = n_min if n_max is None else n_max
        if not n_min <= len(args) <= n_max:
            want = f"{n_min}" if n_min == n_max else f"{n_min}–{n_max}"
            raise AnvilError(f"`{name}` takes {want} argument(s), got {len(args)}", node.span)
        for k in kwargs:
            if k not in allowed_kw:
                raise AnvilError(f"`{name}` has no keyword argument `{k}`", node.span)

    def const_int(self, v, what, span) -> int:
        if not isinstance(v, CVal) or not v.is_int or isinstance(v.value, bool):
            raise AnvilError(f"{what} must be a compile-time integer, found {v.kind}", span)
        return v.value if isinstance(v.value, Dim) else int(v.value)

    def shape_args(self, args, name, span) -> tuple:
        if len(args) == 1 and isinstance(args[0], TupleVal):
            args = args[0].items
        if len(args) == 1 and isinstance(args[0], TVal) and args[0].buf.kind == "const":
            return tuple(int(x) for x in args[0].buf.init)
        return tuple(self.const_int(a, f"the shape given to `{name}`", span) for a in args)

    # ------------------------------------------------------------------ math
    def bi_detach(self, args, kwargs, node):
        self.expect_args("detach", args, kwargs, node, 1)
        v = args[0]
        if isinstance(v, TVal):
            return v.with_detached()
        if isinstance(v, EVal):
            return EVal(mk_unary("detach", v.expr))
        return v

    def bi_f32(self, args, kwargs, node):
        self.expect_args("f32", args, kwargs, node, 1)
        v = args[0]
        if isinstance(v, CVal):
            return CVal(float(v.value))
        return self.elementwise(lambda es: cast(es[0], F32), [v], node.span, "f32")

    bi_float = bi_f32

    def bi_i32(self, args, kwargs, node):
        self.expect_args("i32", args, kwargs, node, 1)
        v = args[0]
        if isinstance(v, CVal):
            return CVal(int(v.value))
        if isinstance(v, (AffVal, IdxVal)):
            return v
        return self.elementwise(lambda es: cast(es[0], I32), [v], node.span, "i32")

    bi_int = bi_i32

    def bi_pow(self, args, kwargs, node):
        self.expect_args("pow", args, kwargs, node, 2)
        return self.binary_op("pow", args[0], args[1], node.span)

    def bi_where(self, args, kwargs, node):
        self.expect_args("where", args, kwargs, node, 3)
        return self.select_op(args[0], args[1], args[2], node.span)

    def _minmax(self, op, args, kwargs, node):
        if len(args) == 2 and not kwargs:
            return self.binary_op(op, args[0], args[1], node.span, label=op)
        self.expect_args(op, args, kwargs, node, 1, 1, ("axis", "keepdims"))
        return self._reduce(op, args, kwargs, node)

    def bi_max(self, args, kwargs, node):
        return self._minmax("max", args, kwargs, node)

    def bi_min(self, args, kwargs, node):
        return self._minmax("min", args, kwargs, node)

    def _reduce(self, op, args, kwargs, node):
        self.expect_args(op, args, kwargs, node, 1, 1, ("axis", "keepdims"))
        axes = None
        if "axis" in kwargs:
            a = kwargs["axis"]
            if isinstance(a, TupleVal):
                axes = [self.const_int(x, "axis", node.span) for x in a.items]
            else:
                axes = [self.const_int(a, "axis", node.span)]
        keep = bool(kwargs["keepdims"].value) if "keepdims" in kwargs else False
        return self.reduce_tensor(op, args[0], axes, keep, node.span)

    def bi_sum(self, args, kwargs, node): return self._reduce("sum", args, kwargs, node)
    def bi_mean(self, args, kwargs, node): return self._reduce("mean", args, kwargs, node)
    def bi_prod(self, args, kwargs, node): return self._reduce("prod", args, kwargs, node)
    def bi_argmax(self, args, kwargs, node): return self._reduce("argmax", args, kwargs, node)
    def bi_argmin(self, args, kwargs, node): return self._reduce("argmin", args, kwargs, node)

    # ------------------------------------------------------------------ creation
    def bi_zeros(self, args, kwargs, node):
        shape = self.shape_args(args, "zeros", node.span)
        return self.emit_map(shape, lambda vs: Const(0.0), span=node.span, label="zeros")

    def bi_ones(self, args, kwargs, node):
        shape = self.shape_args(args, "ones", node.span)
        return self.emit_map(shape, lambda vs: Const(1.0), span=node.span, label="ones")

    def bi_full(self, args, kwargs, node):
        if len(args) < 2:
            raise AnvilError("`full(shape, value)` takes a shape and a value", node.span)
        shape = self.shape_args(args[:-1], "full", node.span)
        v = args[-1]
        return self.emit_map(shape, lambda vs: self.scalar_expr(v, node.span), span=node.span, label="full")

    def bi_arange(self, args, kwargs, node):
        self.expect_args("arange", args, kwargs, node, 1, 2)
        if len(args) == 1:
            a, b = 0, self.const_int(args[0], "arange's bound", node.span)
        else:
            a = self.const_int(args[0], "arange's start", node.span)
            b = self.const_int(args[1], "arange's stop", node.span)
        return self.emit_map((max(0, b - a),), lambda vs: Index(Affine(a) + Affine.of(vs[0])), span=node.span,
                             label="arange")

    def bi_eye(self, args, kwargs, node):
        self.expect_args("eye", args, kwargs, node, 1)
        n = self.const_int(args[0], "eye's size", node.span)
        return self.emit_map((n, n), lambda vs: mk_select(mk_binary("eq", Index(Affine.of(vs[0])),
                                                                    Index(Affine.of(vs[1]))), Const(1.0), Const(0.0)),
                             span=node.span, label="eye")

    def bi_linspace(self, args, kwargs, node):
        self.expect_args("linspace", args, kwargs, node, 3)
        n = self.const_int(args[2], "linspace's count", node.span)
        a = self.scalar_expr(args[0], node.span)
        b = self.scalar_expr(args[1], node.span)
        step = mk_binary("div", mk_binary("sub", b, a), Const(float(max(n - 1, 1))))
        return self.emit_map((n,), lambda vs: mk_binary("add", cast(a, F32),
                                                        mk_binary("mul", step, cast(Index(Affine.of(vs[0])), F32))),
                             span=node.span, label="linspace")

    def bi_randn(self, args, kwargs, node):
        shape = self.shape_args(args, "randn", node.span)
        buf = self.new_temp(shape, F32)
        self.sample_into(buf, DistVal("normal", {"mean": CVal(0.0), "std": CVal(1.0)}), node.span)
        return TVal.of(buf)

    def bi_rand(self, args, kwargs, node):
        shape = self.shape_args(args, "rand", node.span)
        buf = self.new_temp(shape, F32)
        self.sample_into(buf, DistVal("uniform", {"lo": CVal(0.0), "hi": CVal(1.0)}), node.span)
        return TVal.of(buf)

    # ------------------------------------------------------------------ shapes
    def bi_reshape(self, args, kwargs, node):
        if len(args) < 2:
            raise AnvilError("`reshape(x, dims...)` needs a tensor and a shape", node.span)
        x = args[0]
        if not isinstance(x, TVal):
            raise AnvilError(f"cannot reshape a {x.kind}", node.span)
        dims = args[1:]
        if len(dims) == 1 and isinstance(dims[0], TupleVal):
            dims = dims[0].items
        shape = []
        for d in dims:
            if isinstance(d, CVal) and d.value == -1:
                shape.append(-1)
            else:
                shape.append(self.const_int(d, "a reshape dimension", node.span))
        return self.reshape(x, shape, node.span)

    def bi_flatten(self, args, kwargs, node):
        self.expect_args("flatten", args, kwargs, node, 1, 2)
        x = args[0]
        if not isinstance(x, TVal):
            raise AnvilError(f"cannot flatten a {x.kind}", node.span)
        start = self.const_int(args[1], "flatten's start dim", node.span) if len(args) > 1 else 0
        lead = list(x.shape[:start])
        return self.reshape(x, lead + [-1] if x.rank > start else lead, node.span)

    def bi_transpose(self, args, kwargs, node):
        x = args[0]
        if not isinstance(x, TVal):
            raise AnvilError(f"cannot transpose a {x.kind}", node.span)
        perm = [self.const_int(a, "a permutation entry", node.span) for a in args[1:]] or list(reversed(range(x.rank)))
        return self.transpose(x, perm, node.span)

    def bi_stack(self, args, kwargs, node):
        """stack([a, b, c], axis=k): join same-shaped tensors along a new axis k (as in NumPy).
        A list literal already stacks along axis 0, so this moves that axis to k (a view)."""
        self.expect_args("stack", args, kwargs, node, 1, 1, ("axis",))
        x = args[0]
        if not isinstance(x, TVal) or x.rank == 0:
            raise AnvilError("`stack` takes a list of tensors", node.span, help="e.g. `stack([a, b], axis=-1)`")
        axis = self.const_int(kwargs["axis"], "the axis", node.span) if "axis" in kwargs else 0
        if not -x.rank <= axis < x.rank:
            raise AnvilError(f"axis {axis} is out of range for stacking tensors of rank {x.rank - 1}", node.span)
        axis %= x.rank
        perm = list(range(1, axis + 1)) + [0] + list(range(axis + 1, x.rank))
        return self.transpose(x, perm, node.span)

    def bi_len(self, args, kwargs, node):
        self.expect_args("len", args, kwargs, node, 1)
        x = args[0]
        if isinstance(x, TVal):
            if x.rank == 0:
                raise AnvilError("a scalar has no length", node.span)
            return CVal(x.shape[0])
        if isinstance(x, TupleVal):
            return CVal(len(x.items))
        raise AnvilError(f"a {x.kind} has no length", node.span)

    def bi_shape(self, args, kwargs, node):
        self.expect_args("shape", args, kwargs, node, 1)
        x = args[0]
        if not isinstance(x, TVal):
            return TupleVal([])
        return TupleVal([CVal(d) for d in x.shape])

    def bi_copy(self, args, kwargs, node):
        self.expect_args("copy", args, kwargs, node, 1)
        x = args[0]
        if not isinstance(x, TVal):
            return self.to_tensor(x, node.span)
        return self.emit_map(x.shape, lambda vs: x.load([Affine.of(v) for v in vs]), span=node.span, label="copy")

    def bi_nonzero(self, args, kwargs, node):
        """nonzero(mask, size=K, fill_value=0): the indices of the nonzero elements in row-major
        order, one i32[K] tensor per dimension (as in NumPy and JAX). Shapes are static, so the
        result is padded with fill_value, or cut off after K indices."""
        self.expect_args("nonzero", args, kwargs, node, 1, 1, ("size", "fill_value"))
        x = args[0]
        if not isinstance(x, TVal) or x.rank == 0:
            raise AnvilError(f"`nonzero` takes a tensor, found {describe_kind(x)}", node.span)
        if "size" not in kwargs:
            raise AnvilError("`nonzero` needs a static `size`", node.span,
                           help="e.g. `nonzero(mask, size=32)` gives 32 indices: padded, or cut off")
        size = self.const_int(kwargs["size"], "the size", node.span)
        if size <= 0:
            raise AnvilError("the size must be positive", node.span)
        fill = self.const_int(kwargs["fill_value"], "fill_value", node.span) if "fill_value" in kwargs else 0
        flags = self.emit_map(x.shape, lambda vs: cast(mk_binary("ne", x.load([Affine.of(v) for v in vs]),
                                                                 Const(0, x.dtype)), I32),
                              span=node.span, label="nonzero")
        flat = self.new_temp((size,), I32)          # flat indices, then -1s
        self.emit(ir.RTCall("nonzero", {"src": flags.buf, "buf": flat}, span=node.span))
        out = []
        for n, st in zip(x.shape, row_major_strides(x.shape)):
            def body(vs, n=n, st=st):
                f = TVal.of(flat).load([Affine.of(vs[0])])
                i = mk_binary("mod", mk_binary("idiv", f, Const(st, I32)), Const(n, I32)) if x.rank > 1 else f
                return mk_select(mk_binary("lt", f, Const(0, I32)), Const(fill, I32), i)
            out.append(self.emit_map((size,), body, span=node.span, label="nonzero"))
        return TupleVal(out)

    # ------------------------------------------------------------------ iteration
    def bi_range(self, args, kwargs, node):
        self.expect_args("range", args, kwargs, node, 1, 3)
        for a in args:
            if isinstance(a, CVal) and not a.is_int:
                raise AnvilError("`range` takes integers", node.span,
                               help="use `//` for integer division")
            if isinstance(a, TVal) and a.dtype != I32:
                raise AnvilError("`range` takes integers", node.span)
        if len(args) == 1:
            return RangeVal(CVal(0), args[0], 1)
        step = 1
        if len(args) == 3:
            step = self.const_int(args[2], "the range step", node.span)
            if step <= 0:
                raise AnvilError("range steps must be positive", node.span)
        return RangeVal(args[0], args[1], step)

    def bi_batches(self, args, kwargs, node):
        for k in kwargs:
            if k not in ("size", "shuffle"):
                raise AnvilError(f"`batches` has no keyword argument `{k}`", node.span)
        tensors = list(args)
        size = None
        if tensors and isinstance(tensors[-1], CVal):
            size = self.const_int(tensors.pop(), "the batch size", node.span)
        if "size" in kwargs:
            size = self.const_int(kwargs["size"], "the batch size", node.span)
        if size is None or size <= 0:
            raise AnvilError("`batches` needs a positive `size`", node.span, help="e.g. `batches(x, y, size=64)`")
        if not tensors:
            raise AnvilError("`batches` needs at least one tensor", node.span)
        n = None
        for t in tensors:
            if not isinstance(t, TVal) or t.rank == 0:
                raise AnvilError("`batches` takes tensors whose first dimension is the example index", node.span)
            if n is not None and t.shape[0] != n:
                raise AnvilError(f"`batches` tensors disagree on the number of examples ({n} vs {t.shape[0]})",
                               node.span)
            n = t.shape[0]
        if size > n:
            raise AnvilError(f"batch size {size} is larger than the dataset ({n})", node.span)
        shuffle = bool(kwargs["shuffle"].value) if "shuffle" in kwargs else False
        return BatchesVal(tensors, size, shuffle, node.span)

    # ------------------------------------------------------------------ data
    def bi_idx(self, args, kwargs, node):
        self.expect_args("idx", args, kwargs, node, 1)
        p = args[0]
        if not isinstance(p, SVal):
            raise AnvilError("`idx` takes a file path string", node.span)
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        if not os.path.exists(path):
            raise AnvilError(f"data file not found: {path}", node.span,
                           help="paths are relative to the .anvil file; see examples/README for getting MNIST")
        try:
            etype, dims, header = read_idx_header(path)
        except (OSError, ValueError, struct.error) as e:
            raise AnvilError(f"cannot read IDX header of {path}: {e}", node.span)
        if etype not in ("u8",):
            raise AnvilError(f"IDX element type {etype} is not supported yet (only u8)", node.span)
        name = os.path.basename(path).split(".")[0].replace("-", "_")
        buf = self.new_buffer(name, dims, F32, "data", span=node.span)
        self.emit(ir.RTCall("load_idx", {"buf": buf, "path": path, "dims": list(dims), "header": header,
                                         "etype": etype, "gz": _is_gzip(path)}, span=node.span))
        self.last_loaded = buf
        return TVal.of(buf)

    def bi_npy(self, args, kwargs, node):
        """npy("w.npy"): an array saved by NumPy (np.save, or a PyTorch tensor's .numpy()). Its
        shape comes from the file, when the program is compiled. Floating-point files become f32
        tensors and integer or boolean files i32 tensors."""
        self.expect_args("npy", args, kwargs, node, 1)
        p = args[0]
        if not isinstance(p, SVal):
            raise AnvilError("`npy` takes a file path string", node.span)
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        if not os.path.exists(path):
            raise AnvilError(f"data file not found: {path}", node.span, help="paths are relative to the .anvil file")
        try:
            descr, shape, offset = read_npy_header(path)
        except (OSError, ValueError, SyntaxError) as e:
            raise AnvilError(f"cannot read {os.path.basename(path)} as a NumPy file: {e}", node.span)
        kind = NPY_KINDS.get(descr.lstrip("<|="))
        if kind is None or descr.startswith(">"):
            raise AnvilError(f"{os.path.basename(path)} holds {descr} values, which Anvil cannot read", node.span,
                           help="save it as float32, float64, float16, or an integer or boolean type, little-endian")
        n = 1
        for d in shape:
            n *= d
        if 4 * n > (1 << 34):
            raise AnvilError(f"{os.path.basename(path)} is too large ({n:,} values)", node.span)
        dtype = I32 if descr.lstrip("<|=")[0] in "iub" else F32
        name = os.path.basename(path).rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
        buf = self.new_buffer(name, shape, dtype, "data", span=node.span)
        self.emit(ir.RTCall("load_npy", {"buf": buf, "path": path, "offset": offset, "kind": kind[0],
                                         "size": kind[1]}, span=node.span))
        self.last_loaded = buf
        return TVal.of(buf)

    def bi_safetensors(self, args, kwargs, node):
        """safetensors("model.safetensors", "encoder.layer.0.attention.self.query.weight"): one tensor of a
        safetensors file (the format of Hugging Face models). Like `npy`, its shape comes from the file when
        the program is compiled, and the numbers are read when it runs."""
        self.expect_args("safetensors", args, kwargs, node, 2)
        p, key = args
        if not isinstance(p, SVal) or not isinstance(key, SVal):
            raise AnvilError("`safetensors` takes a file path and a tensor name", node.span,
                           help='e.g. `safetensors("model.safetensors", "embeddings.word_embeddings.weight")`')
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        if not os.path.exists(path):
            raise AnvilError(f"model file not found: {path}", node.span, help="paths are relative to the .anvil file")
        cache = self.__dict__.setdefault("_safetensors_headers", {})
        if path not in cache:
            try:
                cache[path] = read_safetensors_header(path)
            except (OSError, ValueError, KeyError) as e:
                raise AnvilError(f"cannot read {os.path.basename(path)} as a safetensors file: {e}", node.span)
        tensors = cache[path]
        if key.value not in tensors:
            import difflib
            close = difflib.get_close_matches(key.value, list(tensors), n=1, cutoff=0.6)
            raise AnvilError(f"{os.path.basename(path)} has no tensor `{key.value}`", node.span,
                           help=f"did you mean `{close[0]}`?" if close else f"it has {len(tensors)} tensors")
        st_dtype, shape, offset, nbytes = tensors[key.value]
        descr = SAFETENSORS_TYPES.get(st_dtype)
        if descr is None:
            raise AnvilError(f"`{key.value}` holds {st_dtype} values, which Anvil cannot read", node.span,
                           help="convert the model to float32 or float16 first")
        kind = NPY_KINDS[descr]
        n = 1
        for d in shape:
            n *= d
        if n * kind[1] != nbytes:
            raise AnvilError(f"`{key.value}` in {os.path.basename(path)} has {nbytes} bytes, not {n * kind[1]}", node.span)
        dtype = I32 if descr[0] in "iub" else F32
        buf = self.new_buffer(key.value.replace(".", "_"), shape, dtype, "data", span=node.span)
        self.emit(ir.RTCall("load_npy", {"buf": buf, "path": path, "offset": offset, "kind": kind[0] | 16,
                                         "size": kind[1]}, span=node.span))
        self.last_loaded = buf
        return TVal.of(buf)

    def bi_save_npy(self, args, kwargs, node):
        """save_npy(x, "x.npy"): write a tensor as a NumPy file (np.load reads it back)."""
        self.expect_args("save_npy", args, kwargs, node, 2)
        x, p = args
        if not isinstance(p, SVal):
            raise AnvilError("the second argument of `save_npy` is a file name", node.span)
        if isinstance(x, CVal):
            x = self.to_tensor(x, node.span)
        if not isinstance(x, TVal):
            raise AnvilError(f"`save_npy` writes a tensor, not {describe_kind(x)}", node.span)
        if not x.is_identity():
            x = self.materialize(x, node.span)
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        header = npy_header("<i4" if x.dtype == I32 else "<f4", x.shape)
        self.emit(ir.RTCall("save_npy", {"buf": x.buf, "path": path, "header": header}, span=node.span))
        return NoneVal()

    def checkpoint_args(self, name, args, kwargs, node):
        self.expect_args(name, args, kwargs, node, 2)
        x, p = args
        if not isinstance(p, SVal):
            raise AnvilError(f"the second argument of `{name}` is a file name", node.span)
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        if self.params_of(x) is not None:
            bufs = self.params_of(x)
        elif isinstance(x, TVal) and x.buf.kind == "param" and x.is_identity():
            bufs = [x.buf]
        else:
            raise AnvilError(f"`{name}` takes a model or a `param`, not {describe_kind(x)}", node.span,
                           help=f'e.g. `{name}(net, "net.weights")`')
        return bufs, path

    def bi_save(self, args, kwargs, node):
        """save(model, "file"): write a model's parameters (or one `param`) to a file. The path is
        relative to the program."""
        bufs, path = self.checkpoint_args("save", args, kwargs, node)
        self.emit(ir.RTCall("save", {"bufs": bufs, "path": path}, span=node.span))
        return NoneVal()

    def bi_load(self, args, kwargs, node):
        """load(model, "file"): read parameters written by `save`. 1 if they were loaded; 0 if the
        file is missing or holds tensors of other sizes, and then nothing changes."""
        bufs, path = self.checkpoint_args("load", args, kwargs, node)
        ok = self.new_temp((), F32, "loaded")
        self.emit(ir.RTCall("load", {"bufs": bufs, "path": path, "buf": ok}, span=node.span))
        return TVal.of(ok)

    def bi_csv(self, args, kwargs, node):
        """csv("file.csv", sep=","): a table of numbers as f32[rows, cols]. The file is read at compile
        time for its shape (a first line that is not all numbers is a header, and skipped) and again
        when the program runs."""
        self.expect_args("csv", args, kwargs, node, 1, 1, ("sep",))
        p = args[0]
        if not isinstance(p, SVal):
            raise AnvilError("`csv` takes a file path string", node.span)
        sep = kwargs["sep"].value if "sep" in kwargs and isinstance(kwargs["sep"], SVal) else ","
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().split("\n")
        except OSError:
            raise AnvilError(f"data file not found: {path}", node.span, help="paths are relative to the .anvil file")

        def fields(line):
            return line.split() if sep == " " else line.split(sep)
        skip, rows, cols = 0, 0, None
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                if rows == 0:
                    skip = lineno
                continue
            fs = fields(line)
            bad = [x for x in fs if not CSV_NUMBER.fullmatch(x)]
            if bad and rows == 0 and skip == lineno - 1 and cols is None:
                skip, cols = lineno, len(fs)              # a header
                continue
            if bad:
                raise AnvilError(f"{os.path.basename(path)}, line {lineno}: `{bad[0].strip()}` is not a number",
                               node.span, help=f"the fields are separated by {sep!r}; pass `sep=` if that is wrong"
                               if len(fs) == 1 else "every field after the header must be a number")
            if cols is not None and len(fs) != cols:
                raise AnvilError(f"{os.path.basename(path)}, line {lineno}: {len(fs)} fields, but the table has "
                               f"{cols} columns", node.span)
            cols = len(fs)
            rows += 1
        if rows == 0:
            raise AnvilError(f"{os.path.basename(path)} has no rows of numbers", node.span)
        name = os.path.basename(path).split(".")[0].replace("-", "_")
        buf = self.new_buffer(name, (rows, cols), F32, "data", span=node.span)
        self.emit(ir.RTCall("load_csv", {"buf": buf, "path": path, "skip": skip}, span=node.span))
        return TVal.of(buf)

    def bi_bytes(self, args, kwargs, node):
        """bytes("file"): a file's bytes as i32[n], values 0–255 (n is its size at compile time)."""
        self.expect_args("bytes", args, kwargs, node, 1)
        p = args[0]
        if not isinstance(p, SVal):
            raise AnvilError("`bytes` takes a file path string", node.span)
        base = self.data_dir(node)
        path = os.path.normpath(os.path.join(base, os.path.expanduser(p.value)))
        if not os.path.isfile(path):
            raise AnvilError(f"data file not found: {path}", node.span, help="paths are relative to the .anvil file")
        n = os.path.getsize(path)
        if n == 0:
            raise AnvilError(f"{os.path.basename(path)} is empty", node.span)
        name = os.path.basename(path).split(".")[0].replace("-", "_")
        buf = self.new_buffer(name, (n,), I32, "data", span=node.span)
        self.emit(ir.RTCall("load_bytes", {"buf": buf, "path": path}, span=node.span))
        return TVal.of(buf)

    def bi_decode(self, args, kwargs, node):
        """decode(codes): print an i32 tensor of byte values as text: `print(decode(sample))`."""
        self.expect_args("decode", args, kwargs, node, 1)
        x = args[0]
        if not isinstance(x, TVal) or x.rank == 0:
            raise AnvilError("`decode` takes a tensor of byte values", node.span)
        if x.dtype != I32 or not x.is_identity():
            x = self.emit_map(x.shape, lambda vs: cast(x.load([Affine.of(v) for v in vs]), I32),
                              span=node.span, label="decode")
        return TextVal(x.buf)

    def bi_seed(self, args, kwargs, node):
        self.expect_args("seed", args, kwargs, node, 1)
        v = self.const_int(args[0], "the seed", node.span)
        self.emit(ir.RTCall("seed", {"value": v & 0xFFFFFFFF}, span=node.span))
        return NoneVal()

    def bi_clock(self, args, kwargs, node):
        self.expect_args("clock", args, kwargs, node, 0)
        buf = self.new_temp((), F32, "clock")
        self.emit(ir.RTCall("clock", {"buf": buf}, span=node.span))
        return TVal.of(buf)

    # ------------------------------------------------------------------ distributions
    def bi_normal(self, args, kwargs, node):
        b = self.bind_dist(["mean", "std"], [CVal(0.0), CVal(1.0)], args, kwargs, node, "normal")
        return DistVal("normal", b, node.span)

    def bi_uniform(self, args, kwargs, node):
        b = self.bind_dist(["lo", "hi"], [CVal(0.0), CVal(1.0)], args, kwargs, node, "uniform")
        return DistVal("uniform", b, node.span)

    def bi_randint(self, args, kwargs, node):
        b = self.bind_dist(["lo", "hi"], [None, None], args, kwargs, node, "randint")
        return DistVal("randint", b, node.span)

    def bi_bernoulli(self, args, kwargs, node):
        b = self.bind_dist(["p"], [None], args, kwargs, node, "bernoulli")
        return DistVal("bernoulli", b, node.span)

    def bind_dist(self, names, defaults, args, kwargs, node, what):
        if len(args) > len(names):
            raise AnvilError(f"`{what}` takes at most {len(names)} arguments", node.span)
        out = dict(zip(names, defaults))
        for n, a in zip(names, args):
            out[n] = a
        for k, v in kwargs.items():
            if k not in names:
                raise AnvilError(f"`{what}` has no parameter `{k}`", node.span)
            out[k] = v
        for k, v in out.items():
            if v is None:
                raise AnvilError(f"`{what}` needs `{k}`", node.span)
        return out

    def sample_into(self, buf, dist, span):
        if not isinstance(dist, DistVal):
            raise AnvilError(f"expected a distribution after `~` (normal, uniform, bernoulli), found {dist.kind}", span)
        shape = buf.shape
        vars_ = [ir.Var(n, d) for n, d in zip(["i", "j", "k", "l", "m", "n", "p", "q"], shape)]
        a = dist.args

        def arg(name):
            v = a[name]
            if isinstance(v, TVal):
                try:
                    out = self.broadcast([shape, v.shape], span)
                except AnvilError:
                    out = None
                if out != tuple(shape):
                    raise AnvilError(f"`{name}` of shape {fmt_shape(v.shape)} does not broadcast to the sampled "
                                   f"shape {fmt_shape(shape)}", span)
                return self.bload(v, vars_, shape, span)
            if isinstance(v, (CVal, AffVal)):
                return cast(self.scalar_expr(v, span), F32)
            raise AnvilError(f"`{name}` must be a number or tensor", span)
        def noise(z, what):
            """The noise of a sample whose parameters are tensors (z ~ normal(mu, sigma), the
            reparameterization trick) goes in its own buffer, so the gradient with respect to a
            parameter (d z / d sigma = the noise) reads the noise the sample used: the gradient's
            kernel would otherwise draw new random numbers."""
            if not any(isinstance(v, TVal) for v in a.values()):
                return z
            nb = self.new_buffer(f"{what}_noise", shape, F32, "temp", span=span)
            self.emit_kernel(self.make_kernel(vars_, nb, z, span=span, label=f"{what} noise"))
            strides = row_major_strides(shape)
            return ir.Load(nb, sum((Affine.of(v) * st for v, st in zip(vars_, strides)), Affine(0)))
        if dist.dist == "normal":
            u1 = Rand(0, open_low=True)
            u2 = Rand(1)
            r = mk_unary("sqrt", mk_binary("mul", Const(-2.0), mk_unary("log", u1)))
            z = noise(mk_binary("mul", r, mk_unary("cos", mk_binary("mul", Const(2 * math.pi), u2))), "normal")
            body = mk_binary("add", arg("mean"), mk_binary("mul", arg("std"), z))
        elif dist.dist == "uniform":
            lo, hi = arg("lo"), arg("hi")
            body = mk_binary("add", lo, mk_binary("mul", mk_binary("sub", hi, lo), noise(Rand(0), "uniform")))
        elif dist.dist == "bernoulli":
            body = mk_binary("lt", Rand(0), arg("p"))
        elif dist.dist == "randint":
            lo, hi = arg("lo"), arg("hi")
            body = mk_binary("add", lo, mk_unary("floor", mk_binary("mul", mk_binary("sub", hi, lo), Rand(0))))
        else:
            raise AnvilError(f"unknown distribution {dist.dist}", span)
        if buf.dtype != F32:
            body = cast(body, buf.dtype)
        self.emit_kernel(self.make_kernel(vars_, buf, body, span=span, label=dist.dist))

    # ------------------------------------------------------------------ output
    def bi_print(self, args, kwargs, node):
        for k in kwargs:
            if k not in ("end", "sep"):
                raise AnvilError(f"`print` has no keyword argument `{k}`", node.span)
        end = kwargs["end"].value if "end" in kwargs and isinstance(kwargs["end"], SVal) else "\n"
        sep = kwargs["sep"].value if "sep" in kwargs and isinstance(kwargs["sep"], SVal) else " "
        items: list[ir.PrintItem] = []
        for i, a in enumerate(args):
            if i:
                items.append(ir.PrintItem("text", sep))
            if isinstance(a, SVal):
                items.append(ir.PrintItem("text", a.value))
            elif isinstance(a, FStrVal):
                for p in a.parts:
                    if isinstance(p, str):
                        items.append(ir.PrintItem("text", p))
                    else:
                        v, spec, sp = p
                        items.extend(self.print_value(v, spec, sp))
            else:
                items.extend(self.print_value(a, "", node.span))
        # merge adjacent text
        merged = []
        for it in items:
            if it.kind == "text" and merged and merged[-1].kind == "text":
                merged[-1] = ir.PrintItem("text", merged[-1].text + it.text)
            else:
                merged.append(it)
        self.emit(ir.Print(merged, end))
        return NoneVal()

    def print_value(self, v, spec, span) -> list:
        if isinstance(v, SVal):
            return [ir.PrintItem("text", v.value)]
        if isinstance(v, CVal):
            val = v.value
            if isinstance(val, bool):
                return [ir.PrintItem("text", "true" if val else "false")]
            if spec:
                try:
                    return [ir.PrintItem("text", format(val, spec))]
                except ValueError:
                    raise AnvilError(f"bad format spec `{spec}` for {val!r}", span)
            return [ir.PrintItem("text", str(val) if isinstance(val, int) else f"{val:g}")]
        if isinstance(v, TupleVal):
            out = [ir.PrintItem("text", "(")]
            for i, x in enumerate(v.items):
                if i:
                    out.append(ir.PrintItem("text", ", "))
                out.extend(self.print_value(x, spec, span))
            out.append(ir.PrintItem("text", ")"))
            return out
        if isinstance(v, ModelInstVal):
            return [ir.PrintItem("text", f"<{v.decl.name.id} {v.name}>")]
        if isinstance(v, PickVal):
            return [ir.PrintItem("pick", buf=v.index.buf, choices=v.choices)]
        if isinstance(v, TextVal):
            return [ir.PrintItem("chars", buf=v.buf)]
        if isinstance(v, FStrVal):                # e.g. a string returned by a function
            out = []
            for p in v.parts:
                out.extend([ir.PrintItem("text", p)] if isinstance(p, str) else self.print_value(*p))
            return out
        if isinstance(v, AffVal) and not spec and v.affine.const == 0 and len(v.affine.terms) == 1:
            (t, c), = v.affine.terms.items()
            if isinstance(t, ir.ScalarRef) and c == 1:
                return [ir.PrintItem("scalar", buf=t.buf, fmt="%d")]
        if isinstance(v, (AffVal, EVal)):
            v = self.to_tensor(v, span)
        if not isinstance(v, TVal):
            return [ir.PrintItem("text", f"<{v.kind}>")]
        if v.rank == 0:
            fmt, scale, want = parse_spec(spec, v.dtype, span)
            if scale != 1.0 or want != v.dtype:
                v = self.emit_map((), lambda vs: mk_binary("mul", Const(scale), cast(v.load([]), F32))
                                  if scale != 1.0 else cast(v.load([]), want), span=span, label="format")
            elif not v.is_identity():
                v = self.materialize(v, span)
            return [ir.PrintItem("scalar", buf=v.buf, fmt=fmt)]
        if not v.is_identity():
            v = self.materialize(v, span)
        return [ir.PrintItem("tensor", buf=v.buf)]

    def bi_show(self, args, kwargs, node):
        """show(grid, glyphs): draw a 1-D or 2-D tensor of small integers as text, one glyph per
        value (glyphs: a string of single characters, or a list of strings)."""
        self.expect_args("show", args, kwargs, node, 2)
        grid, pal = args
        if not isinstance(grid, TVal) or grid.rank not in (1, 2):
            raise AnvilError("`show` draws a 1-D or 2-D tensor", node.span)
        if isinstance(pal, SVal):
            glyphs = list(pal.value)
        elif isinstance(pal, TupleVal) and all(isinstance(g, SVal) for g in pal.items):
            glyphs = [g.value for g in pal.items]
        else:
            raise AnvilError("the second argument of `show` is a string of glyphs or a list of strings", node.span)
        if not glyphs:
            raise AnvilError("`show` needs at least one glyph", node.span)
        if grid.dtype != I32 or not grid.is_identity():
            grid = self.emit_map(grid.shape, lambda vs: cast(grid.load([Affine.of(v) for v in vs]), I32),
                                 span=node.span, label="show")
        rows, cols = (1, grid.shape[0]) if grid.rank == 1 else grid.shape
        self.emit(ir.RTCall("show", {"buf": grid.buf, "rows": rows, "cols": cols, "glyphs": glyphs}, span=node.span))
        return NoneVal()

    def bi_input(self, args, kwargs, node):
        """input(prompt=""): print the prompt, read a line, and return the first (decimal) number
        in it as an f32, or nan if there is none. At the end of the input the program ends."""
        self.expect_args("input", args, kwargs, node, 0, 1)
        if args:
            if not isinstance(args[0], (SVal, FStrVal, PickVal)):
                raise AnvilError("the prompt of `input` is a string", node.span)
            self.bi_print(args, {"end": SVal("")}, node)
        buf = self.new_temp((), F32, "input")
        self.emit(ir.RTCall("input", {"buf": buf}, span=node.span))
        return TVal.of(buf)

    def bi_sleep(self, args, kwargs, node):
        self.expect_args("sleep", args, kwargs, node, 1)
        v = args[0]
        if isinstance(v, CVal):
            v = self.to_tensor(CVal(float(v.value)), node.span, F32)
        elif not (isinstance(v, TVal) and v.rank == 0):
            raise AnvilError("`sleep` takes a number of seconds", node.span)
        v = self.to_tensor(v, node.span, F32)
        if not v.is_identity():
            v = self.materialize(v, node.span)
        self.emit(ir.RTCall("sleep", {"buf": v.buf}, span=node.span))
        return NoneVal()

    # ------------------------------------------------------------------ autodiff
    def bi_grad(self, args, kwargs, node):
        if len(args) < 2:
            raise AnvilError("`grad(loss, x, ...)` needs a loss and at least one tensor", node.span)
        from .autodiff import grad
        targets = []
        for a, an in zip(args[1:], node.args[1:]):
            if self.params_of(a) is not None:
                targets.extend(TVal.of(b) for b in self.params_of(a))
            elif isinstance(a, TVal):
                targets.append((a, an))
            else:
                raise AnvilError(f"can only differentiate with respect to tensors, not a {a.kind}", an.span)
        grads = grad(self, args[0], targets, node)
        return grads[0] if len(grads) == 1 else TupleVal(grads)
