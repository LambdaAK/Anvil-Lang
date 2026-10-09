"""NumPy reference interpreter for the kernel IR.

It defines the semantics that the native backend must reproduce (including the
random number generator, the shuffle and the tensor print format), and is used for
differential testing and finite-difference gradient checks."""
from __future__ import annotations

import gzip
import io
import re
import struct
import sys
import time

import numpy as np

from . import ir
from .ir import I32, Acc, Affine, Binary, Const, Gather, Index, LetRef, Load, Rand, ScalarRef, Select, Unary

M32 = np.uint64(0xFFFFFFFF)
GOLD = 0x9E3779B1
STREAM_MUL = 0x85EBCA77
SALT_MUL = 0xC2B2AE3D
SHUFFLE_SALT = 7
CHUNK = 1 << 22
CKPT_HEAD = struct.Struct("<4sIQ")          # save/load: magic, version, number of tensors
CKPT_MAGIC = b"EINW"
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")    # what `input` reads (as strtof)


class AnvilRuntimeError(Exception):
    pass


class BreakSignal(Exception):
    pass


class ProgramExit(Exception):
    """The program ended itself (a failed `assert`) with this exit status."""

    def __init__(self, code: int):
        super().__init__(f"exit status {code}")
        self.code = code


class EndOfInput(Exception):
    """`input` found the end of the input: the program ends."""


class ContinueSignal(Exception):
    pass


def lowbias32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x = (x.astype(np.uint64) * np.uint64(0x7FEB352D) & M32).astype(np.uint32)
    x ^= x >> np.uint32(15)
    x = (x.astype(np.uint64) * np.uint64(0x846CA68B) & M32).astype(np.uint32)
    x ^= x >> np.uint32(16)
    return x


MIX = 0x632BE5AB      # keeps the hash away from its fixed point at 0


def rand_bits(idx, stream: int, salt: int, seed: int) -> np.ndarray:
    k = (stream * STREAM_MUL + salt * SALT_MUL + seed + MIX) & 0xFFFFFFFF
    h0 = (np.asarray(idx, dtype=np.uint64) * np.uint64(GOLD) + np.uint64(k)) & M32
    return lowbias32(lowbias32(h0))


def fmt_f32(v) -> str:
    """Default float format: fixed-point, or scientific for very large/small magnitudes."""
    v = float(v)
    a = abs(v)
    if v == 0 or v != v or 1e-4 <= a < 1e6:
        return "%.4f" % v
    return "%.4e" % v


def fmt_elem(v, dtype) -> str:
    if dtype == I32:
        return "%d" % int(v)
    return fmt_f32(v)


def format_tensor(flat, shape, dtype) -> str:
    rank = len(shape)
    if rank == 0:
        return fmt_elem(flat[0], dtype)
    numel = int(np.prod(shape))
    summarize = numel > 1000
    strides = ir.row_major_strides(shape)

    def rec(off, d):
        n = shape[d]
        if summarize and n > 6:
            idxs = [0, 1, 2, None, n - 3, n - 2, n - 1]
        else:
            idxs = list(range(n))
        parts = []
        for i in idxs:
            if i is None:
                parts.append("...")
            elif d == rank - 1:
                parts.append(fmt_elem(flat[off + i], dtype))
            else:
                parts.append(rec(off + i * strides[d], d + 1))
        sep = ", " if d == rank - 1 else ",\n" + " " * (d + 1)
        return "[" + sep.join(parts) + "]"
    if numel == 0:
        return "[]"
    return rec(0, 0)


class Interpreter:
    def __init__(self, prog: ir.Program, out=None, float_dtype=np.float32, seed: int = 0, inp=None,
                 check: str | None = None):
        self.prog = prog
        self.check = check                # "nan" | "inf": stop at the first nan (or infinity) a kernel makes
        self.out = out or sys.stdout
        self.inp = inp or sys.stdin
        self.fdt = float_dtype
        self.mem: dict[int, np.ndarray] = {}
        for b in prog.buffers:
            r = b.root
            if r.id in self.mem:
                continue
            dt = np.int32 if r.dtype == I32 else self.fdt
            arr = np.zeros(max(r.numel, 1), dtype=dt)
            if r.init is not None:
                arr[:len(r.init)] = np.asarray(r.init, dtype=dt)
            self.mem[r.id] = arr
        self.seed = seed & 0xFFFFFFFF
        self.stream = 0
        self.t0 = time.perf_counter()
        self.kernel_count = 0

    def arr(self, b: ir.Buffer) -> np.ndarray:
        return self.mem[b.root.id]

    def run(self):
        try:
            self.exec_block(self.prog.main)
        except EndOfInput:
            pass
        self.out.flush() if hasattr(self.out, "flush") else None

    # ------------------------------------------------------------------ statements
    def exec_block(self, block: ir.Block):
        for st in block.stmts:
            self.exec_stmt(st)

    def exec_stmt(self, st):
        if isinstance(st, ir.KernelStmt):
            self.exec_kernel(st.kernel)
        elif isinstance(st, ir.For):
            self.exec_for(st)
        elif isinstance(st, ir.While):
            while True:
                self.exec_block(st.cond_block)
                if self.arr(st.cond)[0] == 0:
                    break
                try:
                    self.exec_block(st.body)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        elif isinstance(st, ir.If):
            if self.arr(st.cond)[0] != 0:
                self.exec_block(st.then)
            else:
                self.exec_block(st.orelse)
        elif isinstance(st, ir.Break):
            raise BreakSignal()
        elif isinstance(st, ir.Continue):
            raise ContinueSignal()
        elif isinstance(st, ir.Print):
            self.exec_print(st)
        elif isinstance(st, ir.Check):
            v = self.eval_affine(st.value, {}, ())
            v = int(np.asarray(v).reshape(-1)[0])
            if not st.lo <= v <= st.hi:
                loc = st.span.location() if st.span is not None else "?"
                raise AnvilRuntimeError(f"{loc}: {st.message} (value {v})")
        elif isinstance(st, ir.RTCall):
            self.exec_rt(st)
        else:
            raise NotImplementedError(type(st).__name__)

    def bound(self, x) -> int:
        if isinstance(x, int):
            return x
        return int(self.arr(x)[0])

    def exec_for(self, st: ir.For):
        c = self.arr(st.counter)
        i = self.bound(st.start)
        stop = self.bound(st.stop)
        c[0] = i
        while c[0] < stop:
            try:
                self.exec_block(st.body)
            except BreakSignal:
                break
            except ContinueSignal:
                pass
            c[0] += st.step

    def exec_print(self, st: ir.Print):
        parts = []
        for it in st.items:
            if it.kind == "text":
                parts.append(it.text)
            elif it.kind == "scalar":
                v = self.arr(it.buf)[0]
                if not it.fmt:
                    parts.append(fmt_elem(v, it.buf.dtype))
                elif it.buf.dtype == I32 or it.fmt.endswith("d"):
                    parts.append(it.fmt % int(v))
                else:
                    parts.append(it.fmt % float(v))
            elif it.kind == "chars":
                codes = self.arr(it.buf)[:it.buf.numel].astype(np.int64) & 0xFF
                parts.append(codes.astype(np.uint8).tobytes().decode("utf-8", errors="replace"))
            elif it.kind == "pick":
                k = int(self.arr(it.buf)[0])
                parts.append(it.choices[k] if 0 <= k < len(it.choices) else "?")
            else:
                b = it.buf
                parts.append(format_tensor(self.arr(b)[:b.numel], b.shape, b.dtype))
        if st.err:
            self.out.flush() if hasattr(self.out, "flush") else None
            sys.stderr.write("".join(parts) + st.end)
            return
        self.out.write("".join(parts) + st.end)

    def exec_rt(self, st: ir.RTCall):
        a = st.args
        if st.name == "load_idx":
            b = a["buf"]
            opener = gzip.open if a["gz"] else open
            with opener(a["path"], "rb") as f:
                f.read(a["header"])
                raw = np.frombuffer(f.read(b.numel), dtype=np.uint8)
            if raw.size != b.numel:
                raise AnvilRuntimeError(f"{a['path']}: file is truncated")
            self.arr(b)[:b.numel] = raw.astype(self.arr(b).dtype)
        elif st.name == "exit":
            self.out.flush() if hasattr(self.out, "flush") else None
            raise ProgramExit(a["value"])
        elif st.name == "skip_rand":
            self.stream += 1                  # a removed kernel's random numbers
        elif st.name == "load_bytes":
            b = a["buf"]
            try:
                with open(a["path"], "rb") as f:
                    raw = f.read()
            except OSError:
                raise AnvilRuntimeError(f"cannot open data file {a['path']}")
            if len(raw) != b.numel:
                raise AnvilRuntimeError(f"data file {a['path']} does not have the shape this program was compiled for")
            self.arr(b)[:b.numel] = np.frombuffer(raw, dtype=np.uint8)
        elif st.name == "load_npy":
            b = a["buf"]
            try:
                if a["kind"] & 16:                          # a tensor in the middle of a .safetensors file
                    dt = {(0, 4): "<f4", (1, 8): "<f8", (2, 2): "<f2", (3, 4): "<i4", (4, 8): "<i8", (5, 1): "u1",
                          (6, 1): "i1", (7, 2): "<u2", (8, 2): "<i2"}[(a["kind"] & 15, a["size"])]
                    data = np.fromfile(a["path"], dtype=dt, count=b.numel, offset=a["offset"])
                else:
                    data = np.load(a["path"], allow_pickle=False)
            except (OSError, ValueError):
                raise AnvilRuntimeError(f"cannot open data file {a['path']}")
            if data.size != b.numel:
                raise AnvilRuntimeError(f"data file {a['path']} does not have the shape this program was compiled for")
            self.arr(b)[:b.numel] = data.reshape(-1).astype(self.arr(b).dtype)
        elif st.name == "save_npy":
            b = a["buf"]
            try:
                with open(a["path"], "wb") as f:
                    f.write(a["header"])
                    f.write(self.arr(b)[:b.numel].astype(np.int32 if b.dtype == I32 else np.float32).tobytes())
            except OSError:
                sys.stderr.write(f"anvil: cannot write {a['path']}\n")
        elif st.name == "load_csv":
            b = a["buf"]
            try:
                with open(a["path"], encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                raise AnvilRuntimeError(f"cannot open data file {a['path']}")
            pos = 0
            for _ in range(a["skip"]):
                pos = text.index("\n", pos) + 1
            vals = NUMBER.findall(text, pos)
            if len(vals) < b.numel:
                raise AnvilRuntimeError(f"data file {a['path']} does not have the shape this program was compiled for")
            self.arr(b)[:b.numel] = np.array(vals[:b.numel], dtype=np.float32)
        elif st.name == "seed":
            self.seed = a["value"] & 0xFFFFFFFF
            self.stream = 0
        elif st.name in ("iota", "iota_once"):
            b = a["buf"]
            self.arr(b)[:b.numel] = np.arange(b.numel, dtype=np.int32)
        elif st.name == "shuffle":
            b = a["buf"]
            p = self.arr(b)
            n = b.numel
            stream = self.stream
            self.stream += 1
            if n > 1:
                h = rand_bits(np.arange(n - 1), stream, SHUFFLE_SALT, self.seed)
                for k in range(n - 1):
                    i = n - 1 - k
                    j = int(h[k]) % (i + 1)
                    p[i], p[j] = p[j], p[i]
        elif st.name == "clock":
            self.arr(a["buf"])[0] = time.perf_counter() - self.t0
        elif st.name == "show":
            g = a["glyphs"]
            data = self.arr(a["buf"])
            lines = []
            for r in range(a["rows"]):
                row = data[r * a["cols"]:(r + 1) * a["cols"]]
                lines.append("".join(g[v] if 0 <= v < len(g) else "?" for v in row))
            self.out.write("\n".join(lines) + "\n")
        elif st.name == "sleep":
            time.sleep(max(0.0, float(self.arr(a["buf"])[0])))
        elif st.name == "input":
            self.out.flush() if hasattr(self.out, "flush") else None
            line = self.inp.readline()
            if not line:
                self.out.write("\n")
                raise EndOfInput()
            m = NUMBER.search(line)
            self.arr(a["buf"])[0] = float(m.group(0)) if m else np.nan
        elif st.name == "save":
            bufs = a["bufs"]
            try:
                with open(a["path"], "wb") as f:
                    f.write(CKPT_HEAD.pack(CKPT_MAGIC, 2, len(bufs)))
                    f.write(struct.pack(f"<{len(bufs)}Q", *[ir.ckpt_key(b.root) for b in bufs]))
                    for b in bufs:
                        f.write(self.arr(b)[:b.numel].astype(np.int32 if b.dtype == I32 else np.float32).tobytes())
            except OSError:
                sys.stderr.write(f"anvil: cannot write {a['path']}\n")
        elif st.name == "load":
            bufs = a["bufs"]
            ok = 0.0
            try:
                with open(a["path"], "rb") as f:
                    raw = f.read()
                sizes = [b.numel for b in bufs]
                head = CKPT_HEAD.size + 8 * len(bufs)
                magic, version, n = CKPT_HEAD.unpack_from(raw) if len(raw) >= CKPT_HEAD.size else (None, 0, 0)
                # version 2 stores ir.ckpt_key (count and shape hash); version 1 only the counts
                want = [ir.ckpt_key(b.root) for b in bufs] if version == 2 else sizes
                if len(raw) == head + 4 * sum(sizes) and magic == CKPT_MAGIC and version in (1, 2) \
                        and n == len(bufs) and list(struct.unpack_from(f"<{len(bufs)}Q", raw, CKPT_HEAD.size)) == want:
                    pos = head
                    for b in bufs:
                        data = np.frombuffer(raw, np.int32 if b.dtype == I32 else np.float32, b.numel, pos)
                        self.arr(b)[:b.numel] = data
                        pos += 4 * b.numel
                    ok = 1.0
            except OSError:
                pass
            self.arr(a["buf"])[0] = ok
        elif st.name == "nonzero":
            src, b = a["src"], a["buf"]
            idx = np.flatnonzero(self.arr(src)[:src.numel])[:b.numel]
            out = self.arr(b)
            out[:b.numel] = -1
            out[:len(idx)] = idx
        else:
            raise NotImplementedError(st.name)

    # ------------------------------------------------------------------ kernels
    def exec_kernel(self, k: ir.Kernel):
        self.kernel_count += 1
        stream = None
        if k.uses_rand():
            stream = self.stream
            self.stream += 1
        D = list(k.domain)
        R = list(k.red.vars) if k.red else []
        total = ir.prod(v.extent for v in D + R)
        if total == 0:
            return
        if total <= CHUNK or (not D and not R):
            self.run_kernel(k, D, R, None, stream)
            self.check_kernel(k)
            return
        # chunk over the first domain var, or partially reduce over the first reduction var
        if D:
            v0 = D[0]
            per = max(1, CHUNK // max(1, total // v0.extent))
            for s in range(0, v0.extent, per):
                self.run_kernel(k, D, R, (v0, s, min(v0.extent, s + per)), stream)
        else:
            self.run_kernel(k, D, R, None, stream, chunk_red=True)
        self.check_kernel(k)

    def check_kernel(self, k: ir.Kernel):
        """--check: the first nan (or infinity) in a tensor k wrote ends the program."""
        if not self.check:
            return
        written = []
        for st in k.stores:
            if st.buf.root.dtype != I32 and st.buf.root not in written:
                written.append(st.buf.root)
        for b in written:
            a = self.arr(b)[:b.numel]
            bad = np.isnan(a) if self.check == "nan" else ~np.isfinite(a)
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                where = f" at {[int(v) for v in np.unravel_index(i, b.shape)]}" if b.shape else ""
                from .backend.aarch64 import check_description
                raise AnvilRuntimeError(f"--check: {a[i]} in `{b.name}`{where}\n{check_description(k)}")

    def grid(self, vars_, ranges):
        env = {}
        n = len(vars_)
        for ax, v in enumerate(vars_):
            lo, hi = ranges.get(v, (0, v.extent))
            shape = [1] * n
            shape[ax] = hi - lo
            env[v] = np.arange(lo, hi, dtype=np.int64).reshape(shape)
        return env

    def run_kernel(self, k, D, R, chunk, stream, chunk_red=False):
        ranges = {}
        if chunk is not None:
            v0, lo, hi = chunk
            ranges[v0] = (lo, hi)
        allv = D + R
        env = self.grid(allv, ranges)
        dshape = tuple((ranges.get(v, (0, v.extent))[1] - ranges.get(v, (0, v.extent))[0]) for v in D)
        fullshape = dshape + tuple(v.extent for v in R)
        ctx = {"env": env, "ndim": len(allv), "stream": stream, "D": D, "shape": fullshape, "lets": {}}
        acc = None
        if k.red is not None:
            if chunk_red and R:
                acc = self.reduce_chunked(k, D, R, ctx, env, fullshape)
            else:
                body = np.broadcast_to(self.eval(k.red.body, ctx), fullshape)
                acc = self.reduce(k.red.op, body, len(D))
        # stmts are evaluated over the domain only
        denv = {v: env[v].reshape(env[v].shape[:len(D)]) if len(D) else env[v] for v in D}
        sctx = {"env": denv, "ndim": len(D), "stream": stream, "D": D, "shape": dshape, "lets": {}, "acc": acc}
        for st in k.stmts:
            if isinstance(st, ir.Let):
                sctx["lets"][id(st)] = self.eval(st.value, sctx)
                continue
            val = np.broadcast_to(self.eval(st.value, sctx), dshape)
            off = np.broadcast_to(self.eval_affine(st.offset, denv, dshape, sctx), dshape)
            a = self.arr(st.buf)
            val = val.astype(a.dtype, copy=False)
            if st.accumulate:
                np.add.at(a, off.reshape(-1), val.reshape(-1))
            else:
                a[off.reshape(-1)] = val.reshape(-1)

    def reduce(self, op, body, nd):
        axes = tuple(range(nd, body.ndim))
        if op == "sum":
            return body.sum(axis=axes, dtype=body.dtype)
        if op == "max":
            return body.max(axis=axes)
        if op == "min":
            return body.min(axis=axes)
        if op == "prod":
            return body.prod(axis=axes, dtype=body.dtype)
        if op in ("argmax", "argmin"):
            flat = body.reshape(body.shape[:nd] + (-1,))
            r = flat.argmax(axis=-1) if op == "argmax" else flat.argmin(axis=-1)
            return r.astype(np.int32)
        raise NotImplementedError(op)

    def reduce_chunked(self, k, D, R, ctx, env, fullshape):
        r0 = R[0]
        per = max(1, CHUNK // max(1, ir.prod(v.extent for v in R) // r0.extent))
        acc = None
        best_idx = None
        for s in range(0, r0.extent, per):
            e = min(r0.extent, s + per)
            env2 = self.grid(D + R, {r0: (s, e)})
            shape2 = tuple(v.extent for v in D) + tuple((e - s) if v is r0 else v.extent for v in R)
            c2 = dict(ctx, env=env2, shape=shape2)
            body = np.broadcast_to(self.eval(k.red.body, c2), shape2)
            part = self.reduce(k.red.op if k.red.op not in ("argmax", "argmin") else
                               ("max" if k.red.op == "argmax" else "min"), body, len(D))
            if k.red.op in ("argmax", "argmin"):
                pidx = self.reduce(k.red.op, body, len(D)) + s
                if acc is None:
                    acc, best_idx = part, pidx
                else:
                    better = part > acc if k.red.op == "argmax" else part < acc
                    acc = np.where(better, part, acc)
                    best_idx = np.where(better, pidx, best_idx)
                continue
            if acc is None:
                acc = part
            elif k.red.op == "sum":
                acc = acc + part
            elif k.red.op == "max":
                acc = np.maximum(acc, part)
            elif k.red.op == "min":
                acc = np.minimum(acc, part)
            elif k.red.op == "prod":
                acc = acc * part
        return best_idx.astype(np.int32) if k.red.op in ("argmax", "argmin") else acc

    # ------------------------------------------------------------------ expressions
    def eval_affine(self, a: Affine, env, shape, ctx=None):
        v = np.int64(a.const)
        for t, c in a.terms.items():
            if isinstance(t, ir.Var):
                v = v + c * env[t]
            elif isinstance(t, ScalarRef):
                v = v + c * np.int64(self.arr(t.buf)[0])
            elif isinstance(t, Gather):
                g = self.eval(t.load, ctx or {"env": env, "ndim": len(shape), "lets": {}}).astype(np.int64)
                lo, hi = t.check_range()
                if np.any((g < lo) | (g >= hi)):
                    bad = int(np.asarray(g)[(g < lo) | (g >= hi)].reshape(-1)[0])
                    raise AnvilRuntimeError(f"index {bad} out of bounds for a dimension of size {t.bound} "
                                          f"({t.where})")
                v = v + c * g
        return v

    def eval(self, e, ctx):
        fdt = self.fdt
        if isinstance(e, Const):
            return np.asarray(e.value, dtype=np.int32 if e.dtype == I32 else fdt)
        if isinstance(e, Load):
            off = self.eval_affine(e.offset, ctx["env"], ctx.get("shape", ()), ctx)
            return self.arr(e.buf)[off]
        if isinstance(e, Index):
            return np.asarray(self.eval_affine(e.affine, ctx["env"], ctx.get("shape", ()), ctx), dtype=np.int32)
        if isinstance(e, Acc):
            return ctx["acc"]
        if isinstance(e, LetRef):
            return ctx["lets"][id(e.let)]
        if isinstance(e, Rand):
            idx = np.int64(0)
            D = ctx["D"]
            strides = ir.row_major_strides([v.extent for v in D])
            for v, s in zip(D, strides):
                idx = idx + ctx["env"][v].reshape(ctx["env"][v].shape[:len(D)] + (1,) * (ctx["ndim"] - len(D))) * s
            bits = rand_bits(idx, ctx["stream"], e.salt, self.seed)
            hi = (bits >> np.uint32(8)).astype(np.float64)
            if e.open_low:
                hi = hi + 1.0
            return (hi * (2.0 ** -24)).astype(fdt)
        if isinstance(e, Unary):
            a = self.eval(e.a, ctx)
            return self.unary(e.op, a, e)
        if isinstance(e, Binary):
            a = self.eval(e.a, ctx)
            b = self.eval(e.b, ctx)
            return self.binary(e.op, a, b, e)
        if isinstance(e, Select):
            c = self.eval(e.c, ctx)
            a = self.eval(e.a, ctx)
            b = self.eval(e.b, ctx)
            dt = np.int32 if e.dtype == I32 else fdt
            return np.where(c != 0, a, b).astype(dt)
        raise NotImplementedError(type(e).__name__)

    def unary(self, op, a, e):
        fdt = self.fdt
        with np.errstate(all="ignore"):
            if op == "neg": return -a
            if op == "abs": return np.abs(a)
            if op == "exp": return np.exp(a.astype(fdt))
            if op == "log": return np.log(a.astype(fdt))
            if op == "sqrt": return np.sqrt(a.astype(fdt))
            if op == "rsqrt": return (1 / np.sqrt(a.astype(fdt))).astype(fdt)
            if op == "recip": return (1 / a.astype(fdt)).astype(fdt)
            if op == "tanh": return np.tanh(a.astype(fdt))
            if op == "sigmoid": return (1 / (1 + np.exp(-a.astype(fdt)))).astype(fdt)
            if op == "sin": return np.sin(a.astype(fdt))
            if op == "cos": return np.cos(a.astype(fdt))
            if op == "floor": return np.floor(a)
            if op == "ceil": return np.ceil(a)
            if op == "round": return np.round(a)
            if op == "sign": return np.sign(a).astype(a.dtype)
            if op == "not": return (a == 0).astype(fdt)
            if op == "f32": return a.astype(fdt)
            if op == "i32": return np.trunc(a).astype(np.int32) if a.dtype != np.int32 else a
            if op == "detach": return a
        raise NotImplementedError(op)

    def binary(self, op, a, b, e):
        fdt = self.fdt
        isint = e.dtype == I32
        with np.errstate(all="ignore"):
            if op == "add": r = a + b
            elif op == "sub": r = a - b
            elif op == "mul": r = a * b
            elif op == "div": r = a.astype(fdt) / b.astype(fdt)
            elif op == "idiv":
                if isint:
                    bb = np.where(b == 0, 1, b)
                    r = np.floor_divide(a, bb)
                else:
                    r = np.floor(a / b)
            elif op == "mod":
                if isint:
                    bb = np.where(b == 0, 1, b)
                    r = np.mod(a, bb)
                else:
                    r = np.mod(a, b)
            elif op == "max": r = np.maximum(a, b)
            elif op == "min": r = np.minimum(a, b)
            elif op == "pow": r = np.power(a.astype(fdt), b.astype(fdt))
            elif op == "lt": r = (a < b)
            elif op == "le": r = (a <= b)
            elif op == "gt": r = (a > b)
            elif op == "ge": r = (a >= b)
            elif op == "eq": r = (a == b)
            elif op == "ne": r = (a != b)
            elif op == "and": r = (a != 0) & (b != 0)
            elif op == "or": r = (a != 0) | (b != 0)
            else:
                raise NotImplementedError(op)
        return np.asarray(r).astype(np.int32 if isint else fdt)


def run_program(prog: ir.Program, out=None, float_dtype=np.float32, seed=0, check=None) -> Interpreter:
    it = Interpreter(prog, out=out, float_dtype=float_dtype, seed=seed, check=check)
    it.run()
    return it


def run_to_string(prog: ir.Program, **kw) -> str:
    buf = io.StringIO()
    run_program(prog, out=buf, **kw)
    return buf.getvalue()
