"""Lowering of one kernel to AArch64/NEON machine IR.

Schedules
  vec_out  vectorize the innermost output dimension: 4 lanes × U column groups, and
           (for reductions) R row groups of the next-outer dimension. That gives R×U
           accumulators held in registers across the whole reduction, the classic GEMM
           micro-kernel; a lane-uniform operand is loaded once per row with `ldr s`
           and multiplied in with by-element `fmla`.
  vec_red  vectorize the innermost reduction dimension; horizontal reduce at the end
  scalar   one element at a time (lane 0 of the vector registers)

Every distinct memory access gets a pointer register that is advanced by its stride in
each loop latch (and rewound on exit). Accesses whose address depends on gathered data
are recomputed at the loop level of that dependency (one pointer per row group when the
gather depends on the tiled row). Sub-expressions are evaluated at the outermost loop
level they depend on (loop-invariant code motion), once per register group they vary in.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .. import ir
from ..ir import (F32, I32, Acc, Affine, Binary, Const, Gather, Index, Kernel, Let, LetRef, Load, Rand, ScalarRef,
                  Select, Store, Unary)
from .mir import Ins, Label, LoopMark, Seq, VReg
from .vecmath import VecMath

GOLD = 0x9E3779B1
STREAM_MUL = 0x85EBCA77
SALT_MUL = 0xC2B2AE3D
MIX = 0x632BE5AB      # keeps the hash away from its fixed point at 0


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


_FMOV_IMMS = {(1 + m / 16) * 2.0 ** e for e in range(-3, 5) for m in range(16)}


def _fmov_encodable(x: float) -> bool:
    return x != 0 and math.isfinite(x) and abs(x) in _FMOV_IMMS


@dataclass
class Config:
    mode: str = "scalar"      # vec_out | vec_red | scalar
    U: int = 1                # column groups of 4 lanes
    R: int = 1                # row groups (vec_out with a reduction)
    hoist: bool = True
    parallel: bool = False    # called as kernel(start, end) over the outermost loop

    def describe(self) -> str:
        if self.mode == "vec_out":
            tile = f"{self.R}×{4 * self.U} register tile" if self.R > 1 else f"{4 * self.U} lanes per step"
            return f"vectorized over the output's last index, {tile}" + (
                "; parallel over the outermost loop" if self.parallel else "")
        par = "; parallel over the outermost loop" if self.parallel else ""
        if self.mode == "vec_red":
            rows = f"{self.R} rows × " if self.R > 1 else ""
            return f"vectorized over the reduction, {rows}{4 * self.U} lanes per step + horizontal reduce{par}"
        return "scalar" + par


class Access:
    def __init__(self, buf, offset: Affine, base_level: int, per_row: bool):
        self.buf = buf
        self.offset = offset
        self.base_level = base_level
        self.per_row = per_row     # one pointer per row group (gather depends on the tiled row)
        self.ptr: VReg | None = None
        self.ptrs: dict[int, VReg] = {}
        self.inited = None         # (level, generation) of the loop emission that set the pointer(s)

    def all_ptrs(self):
        if self.per_row:
            return list(self.ptrs.values())
        return [self.ptr] if self.ptr is not None else []


class Lowering(VecMath):
    def __init__(self, k: Kernel, cfg: Config, sym, fail_msg):
        self.k = k
        self.cfg = cfg
        self.sym = sym
        self.fail_msg = fail_msg
        self.D = list(k.domain)
        self.R = list(k.red.vars) if k.red else []
        self.n = len(self.D)
        self.m = len(self.R)
        self.order = self.D + self.R
        self.lvl_of = {v: i + 1 for i, v in enumerate(self.order)}
        self.vec_var = None
        if cfg.mode == "vec_out":
            self.vec_var = self.D[-1]
        elif cfg.mode == "vec_red":
            self.vec_var = self.R[-1]
        # register tiles: R rows of the second-to-last output index at once. In vec_out each row
        # holds 4·U output lanes; in vec_red each row is one output, a dot product over the
        # reduction, and rows share whatever the dot product reads that does not depend on the row
        self.row_var = self.D[-2] if (cfg.mode in ("vec_out", "vec_red") and cfg.R > 1 and self.n >= 2) else None
        self.row_level = self.lvl_of[self.row_var] if self.row_var is not None else None
        self.vec_level = self.lvl_of[self.vec_var] if self.vec_var is not None else None
        self.uses_vectors = cfg.mode in ("vec_out", "vec_red")
        self.lists: dict[int, Seq] = {}
        self.memo: dict[int, dict] = {}
        self.memo_post: dict = {}
        self.in_epilogue = False
        self.cur_level = 0
        self.out: Seq | None = None
        self.vector = False
        self.row_groups = 1
        self.vec_groups = 1
        self.counters: dict = {}
        self.accesses: dict = {}
        self.scalars: dict = {}
        self.label_n = 0
        self.acc_val: dict = {}
        self.let_val: dict = {}
        self.rand_k: dict = {}
        self.stream = None
        self.loop_gen: dict[int, int] = {}
        self._counter_started: set = set()
        self._post: dict = {}
        self._level: dict = {}
        self._varies: dict = {}
        self._varies_row: dict = {}
        self._vars: dict = {}
        self.par = cfg.parallel and self.n >= 1
        self.par_start = self.par_end = None

    # ------------------------------------------------------------------ emission helpers
    def I(self, fmt, *regs, comment=""):
        self.out.add(Ins(fmt, regs, comment=comment))

    def vx(self, name=""):
        return VReg("x", name)

    def vv(self, name=""):
        return VReg("v", name)

    def label(self, hint="L"):
        self.label_n += 1
        return f"L{self.k.name}_{hint}{self.label_n}"

    def mov_imm(self, r, value: int, w=False):
        """Load an integer constant into a GPR (a VReg, or a scratch name like 'x16')."""
        bits = 32 if w else 64
        value &= (1 << bits) - 1
        reg = [r] if isinstance(r, VReg) else []
        name = ("{0:w}" if w else "{0}") if isinstance(r, VReg) else (("w" if w else "x") + r[1:])
        if value == 0:
            self.out.add(Ins(f"mov {name}, #0", reg))
            return
        first = True
        for i in range(bits // 16):
            c = (value >> (16 * i)) & 0xFFFF
            if c == 0:
                continue
            if first:
                self.out.add(Ins(f"movz {name}, #{c}" + (f", lsl #{16 * i}" if i else ""), reg))
                first = False
            else:
                self.out.add(Ins(f"movk {name}, #{c}, lsl #{16 * i}", reg))

    def add_imm(self, dst: VReg, src: VReg, value: int):
        if value == 0:
            if dst is not src:
                self.I("mov {0}, {1}", dst, src)
            return
        if 0 < value < 4096:
            self.I(f"add {{0}}, {{1}}, #{value}", dst, src)
        elif -4096 < value < 0:
            self.I(f"sub {{0}}, {{1}}, #{-value}", dst, src)
        elif value > 0 and value % 4096 == 0 and value < (4096 << 12):
            self.I(f"add {{0}}, {{1}}, #{value >> 12}, lsl #12", dst, src)
        elif value < 0 and (-value) % 4096 == 0 and -value < (4096 << 12):
            self.I(f"sub {{0}}, {{1}}, #{(-value) >> 12}, lsl #12", dst, src)
        else:
            self.mov_imm("x16", value)
            self.I("add {0}, {1}, x16", dst, src)

    def addr(self, base: VReg, off: int, scale: int, limit: int):
        """Operand text for [base, #off]; returns (text, base) or ('[x16]', None) after
        materializing the address in x16 when the immediate does not fit."""
        if off == 0:
            return "[{R}]", base
        if 0 < off <= limit and off % scale == 0:
            return f"[{{R}}, #{off}]", base
        self.mov_imm("x16", off)
        self.I("add x16, {0}, x16", base)
        return "[x16]", None

    def mem(self, op: str, vreg: VReg, base: VReg, off: int, kind: str):
        """ldr/str of an s (4-byte) or q (16-byte) register at base + off."""
        scale, limit = (4, 16380) if kind == "s" else (16, 65520)
        text, b = self.addr(base, off, scale, limit)
        if b is None:
            self.I(f"{op} {{0:{kind}}}, {text}", vreg)
        else:
            self.I(f"{op} {{0:{kind}}}, " + text.replace("{R}", "{1}"), vreg, b)

    def at(self, level: int):
        return _At(self, level)

    # ------------------------------------------------------------------ analysis
    def expr_vars(self, e) -> set:
        k = id(e)
        r = self._vars.get(k)
        if r is None:
            r = set()
            for x in ir.iter_expr(e):
                if isinstance(x, Load):
                    r.update(x.offset.vars())
                elif isinstance(x, Index):
                    r.update(x.affine.vars())
                elif isinstance(x, Rand):
                    r.update(self.D)
            self._vars[k] = r
        return r

    def in_place(self, e) -> bool:
        """After the reduction (in the epilogue), a value that varies with the innermost loops is
        computed where it is used. Hoisted to the top of the loop body, it would come before the
        reduction loop and hold a register all through it (the old weights of a fused update
        took 16 of the 32 vector registers that a 4×16 tile needs)."""
        return self.in_epilogue and self.cfg.hoist and self.level(e) >= self.cur_level

    def is_post(self, e) -> bool:
        k = id(e)
        r = self._post.get(k)
        if r is None:
            r = any(isinstance(x, (Acc, LetRef)) for x in ir.iter_expr(e))
            self._post[k] = r
        return r

    def level(self, e) -> int:
        k = id(e)
        r = self._level.get(k)
        if r is None:
            r = max((self.lvl_of[v] for v in self.expr_vars(e) if v in self.lvl_of), default=0)
            self._level[k] = r
        return r

    def _varies_in(self, e, var, cache, acc_varies: bool) -> bool:
        if var is None:
            return False
        k = id(e)
        r = cache.get(k)
        if r is None:
            r = var in self.expr_vars(e)
            if not r:
                for x in ir.iter_expr(e):
                    if isinstance(x, Acc) and acc_varies:
                        r = True
                    elif isinstance(x, LetRef) and self._varies_in(x.let.value, var, cache, acc_varies):
                        r = True
            cache[k] = r
        return r

    def varies(self, e) -> bool:
        """Different per vector lane (depends on the vectorized index)."""
        return self._varies_in(e, self.vec_var, self._varies, self.cfg.mode == "vec_out")

    def varies_row(self, e) -> bool:
        """Different per row group (depends on the tiled row index)."""
        return self._varies_in(e, self.row_var, self._varies_row, True)

    def gkey(self, e, g):
        r, u = g
        return (r if self.varies_row(e) else None, u if self.varies(e) else None)

    # ------------------------------------------------------------------ accesses
    def access(self, buf, offset: Affine) -> Access:
        key = (buf.root.id, offset.key())
        a = self.accesses.get(key)
        if a is None:
            gv = set()
            for g in offset.gathers():
                gv.update(g.load.offset.vars())
            base = max((self.lvl_of[v] for v in gv), default=0)
            per_row = self.row_var is not None and self.row_var in gv
            a = Access(buf, offset, base, per_row)
            self.accesses[key] = a
        return a

    def collect_accesses(self):
        def reg(buf, off):
            self.access(buf, off)
            for g in off.gathers():
                reg(g.load.buf, g.load.offset)
        for e in self.k.exprs():
            for ld in ir.loads_in(e):
                reg(ld.buf, ld.offset)
            for x in ir.iter_expr(e):
                if isinstance(x, Index):           # a gathered value used as a number
                    for g in x.affine.gathers():
                        reg(g.load.buf, g.load.offset)
        for s in self.k.stores:
            reg(s.buf, s.offset)

    def ptr_for(self, a: Access, g, lanes: bool):
        """(pointer, byte offset) of access a for register group g = (row, col)."""
        r, u = g
        if a.per_row:
            base, off = a.ptrs[r], 0
        else:
            base = a.ptr
            off = 4 * a.offset.coef(self.row_var) * r if self.row_var is not None else 0
        if lanes and self.vec_var is not None:
            off += 4 * a.offset.coef(self.vec_var) * 4 * u
        return base, off

    def scalar_reg(self, buf) -> VReg:
        """Runtime i32 scalar (sign-extended), loaded once at kernel entry."""
        r = self.scalars.get(buf.root.id)
        if r is None:
            r = self.vx(buf.name)
            with self.at(0):
                sym = self.sym(buf)
                self.I(f"adrp x16, {sym}@PAGE")
                if ir.indirect(buf):
                    self.I(f"ldr x16, [x16, {sym}@PAGEOFF]")
                    self.I("ldrsw {0}, [x16]", r)
                else:
                    self.I(f"ldrsw {{0}}, [x16, {sym}@PAGEOFF]", r)
            self.scalars[buf.root.id] = r
        return r

    def ensure_pointer(self, a: Access):
        """Initialize a's pointer(s) once per emission of its base-level loop."""
        tag = (a.base_level, self.loop_gen.get(a.base_level, 0))
        if a.inited == tag:
            return
        a.inited = tag
        if a.per_row:
            a.ptrs = {r: self.init_pointer(a, r) for r in range(self.row_groups)}
        else:
            a.ptr = self.init_pointer(a, 0)

    def init_pointer(self, a: Access, r: int) -> VReg:
        """Compute a pointer from scratch (at the access's base level) for row group r."""
        p = self.vx(a.buf.name)
        sym = self.sym(a.buf)
        self.I(f"adrp {{0}}, {sym}@PAGE", p)
        if ir.indirect(a.buf):
            self.I(f"ldr {{0}}, [{{0}}, {sym}@PAGEOFF]", p)       # the caller's array
        else:
            self.I(f"add {{0}}, {{0}}, {sym}@PAGEOFF", p)
        extra = a.offset.const
        for t, c in a.offset.terms.items():
            if isinstance(t, ScalarRef):
                self.madd_const(p, self.scalar_reg(t.buf), 4 * c)
            elif isinstance(t, ir.Var):
                lv = self.lvl_of.get(t)
                if lv is not None and lv <= a.base_level:
                    self.madd_const(p, self.counters[t], 4 * c)
                    if t is self.row_var:
                        extra += c * r
            elif isinstance(t, Gather):
                self.madd_const(p, self.gather_value(t, r), 4 * c)
        if self.par and a.base_level == 0 and a.offset.coef(self.D[0]):
            self.madd_const(p, self.par_start, 4 * a.offset.coef(self.D[0]))   # chunk start
        self.add_imm(p, p, 4 * extra)
        return p

    def madd_const(self, p: VReg, x: VReg, c: int):
        if c == 0:
            return
        if c > 0 and (c & (c - 1)) == 0:
            self.I(f"add {{0}}, {{0}}, {{1}}, lsl #{c.bit_length() - 1}", p, x)
            return
        self.mov_imm("x17", c)
        self.I("madd {0}, {1}, x17, {0}", p, x)

    def msub_const(self, p, x, c):
        if c == 0:
            return
        self.mov_imm("x17", c)
        self.I("msub {0}, {1}, x17, {0}", p, x)

    def gather_value(self, g: Gather, r: int) -> VReg:
        """Load an integer index from memory and bounds-check it (scalar GPR)."""
        ga = self.access(g.load.buf, g.load.offset)
        self.ensure_pointer(ga)
        base, off = self.ptr_for(ga, (r, 0), False)
        v = self.vx("gidx")
        text, b = self.addr(base, off, 4, 16380)
        if b is None:
            self.I(f"ldrsw {{0}}, {text}", v)
        else:
            self.I("ldrsw {0}, " + text.replace("{R}", "{1}"), v, b)
        ok = self.label("ok")
        lo, hi = g.check_range()                 # lo <= v < hi, as one unsigned compare of v - lo
        c = v
        if lo:
            c = self.vx("gchk")
            self.mov_imm("x16", -lo)
            self.I("add {0}, {1}, x16", c, v)
        if hi - lo < 4096 and hi - lo >= 0:
            self.I(f"cmp {{0}}, #{hi - lo}", c)
        else:
            self.mov_imm("x16", hi - lo)
            self.I("cmp {0}, x16", c)
        self.I(f"b.lo {ok}")
        msg = self.fail_msg(f"index out of bounds ({g.where or 'gather'}; dimension size {g.bound})")
        self.I("mov x1, {0}", v)
        self.I(f"adrp x0, {msg}@PAGE")
        self.I(f"add x0, x0, {msg}@PAGEOFF")
        self.I("bl _anvil_rt_oob")
        self.out.add(Label(ok))
        return v

    # ------------------------------------------------------------------ constants
    def const(self, value, dtype=F32) -> VReg:
        bits = f32_bits(float(value)) if dtype == F32 else (int(value) & 0xFFFFFFFF)
        lvl = 0 if self.cfg.hoist else self.cur_level
        table = self.memo.setdefault(lvl, {})
        key = ("const", bits, dtype)
        r = table.get(key)
        if r is not None:
            return r
        r = self.vv(f"c{value:g}" if dtype == F32 else f"c{value}")
        with self.at(lvl):
            if bits == 0:
                self.I("movi {0}.4s, #0", r)
            elif dtype == F32 and _fmov_encodable(float(value)):
                self.I(f"fmov {{0}}.4s, #{float(value)!r}", r)
            else:
                self.mov_imm("x16", bits, w=True)
                self.I("dup {0}.4s, w16", r)
        table[key] = r
        return r

    # ------------------------------------------------------------------ evaluation
    def ev(self, e, g=(0, 0)) -> VReg:
        if isinstance(e, Const):
            return self.const(e.value, e.dtype)
        gk = self.gkey(e, g)
        if self.is_post(e) or self.in_place(e):
            key = (e.key(), gk)
            r = self.memo_post.get(key)
            if r is None:
                r = self.gen(e, g)
                self.memo_post[key] = r
            return r
        lvl = min(self.level(e), self.cur_level) if self.cfg.hoist else self.cur_level
        table = self.memo.setdefault(lvl, {})
        key = (e.key(), gk)
        r = table.get(key)
        if r is not None:
            return r
        with self.at(lvl):
            saved = self.vector
            if self.vec_level is None or lvl < self.vec_level:
                self.vector = self.uses_vectors        # hoisted values are broadcast across lanes
            r = self.gen(e, g)
            self.vector = saved
        table[key] = r
        return r

    def ev_s0(self, e: Load, g) -> VReg:
        """A lane-uniform load as a scalar in lane 0 (for by-element multiply)."""
        rk = g[0] if self.varies_row(e) else None
        lvl = min(self.level(e), self.cur_level) if self.cfg.hoist else self.cur_level
        table = self.memo.setdefault(lvl, {})
        key = ("s0", e.key(), rk)
        r = table.get(key)
        if r is None:
            with self.at(lvl):
                a = self.access(e.buf, e.offset)
                base, off = self.ptr_for(a, g, False)
                r = self.vv(e.buf.name)
                self.mem("ldr", r, base, off, "s")
            table[key] = r
        return r

    def gen(self, e, g) -> VReg:
        if isinstance(e, Load):
            return self.gen_load(e, g)
        if isinstance(e, Index):
            return self.gen_index(e.affine, g)
        if isinstance(e, Rand):
            return self.gen_rand(e, g)
        if isinstance(e, Acc):
            return self.acc_val[g if self.cfg.mode == "vec_out" else (g[0], 0)]
        if isinstance(e, LetRef):
            return self.let_val[(e.let, self.gkey(e.let.value, g))]
        if isinstance(e, Unary):
            if e.op == "detach":
                return self.ev(e.a, g)
            return self.gen_unary(e.op, self.ev(e.a, g), e.a.dtype)
        if isinstance(e, Binary):
            return self.gen_binary(e, g)
        if isinstance(e, Select):
            m, inv = self.ev_mask(e.c, g)
            a = self.ev(e.a, g)
            b = self.ev(e.b, g)
            if inv:
                a, b = b, a
            d = self.vv("sel")
            self.I("mov {0}.16b, {1}.16b", d, m)
            self.I("bsl {0}.16b, {1}.16b, {2}.16b", d, a, b)
            return d
        raise NotImplementedError(type(e).__name__)

    # ------------------------------------------------------------------ loads / index values
    def gen_load(self, e: Load, g) -> VReg:
        a = self.access(e.buf, e.offset)
        d = self.vv(e.buf.name)
        varies = self.varies(e)
        if not self.vector:
            base, off = self.ptr_for(a, g, False)
            self.mem("ldr", d, base, off, "s")
            return d
        if not varies:
            base, off = self.ptr_for(a, g, False)
            if off:
                self.mov_imm("x16", off)
                self.I("add x16, {0}, x16", base)
                self.I("ld1r {{0}.4s}, [x16]", d)
            else:
                self.I("ld1r {{0}.4s}, [{1}]", d, base)
            return d
        c = e.offset.coef(self.vec_var)
        base, off = self.ptr_for(a, g, True)
        if c == 1:
            self.mem("ldr", d, base, off, "q")
            return d
        for lane in range(4):              # strided: gather 4 lanes
            lo = off + 4 * c * lane
            if lane == 0:
                self.mem("ldr", d, base, lo, "s")
            else:
                t = self.vv("lane")
                self.mem("ldr", t, base, lo, "s")
                self.I(f"mov {{0}}.s[{lane}], {{1}}.s[0]", d, t)
        return d

    def affine_gpr(self, aff: Affine, extra: int = 0, row: int = 0) -> VReg:
        r = self.vx("idx")
        self.mov_imm(r, aff.const + extra)
        for t, c in aff.terms.items():
            if isinstance(t, ir.Var):
                if c > 0:
                    self.madd_const(r, self.counters[t], c)
                else:
                    self.msub_const(r, self.counters[t], -c)
            elif isinstance(t, ScalarRef):
                s = self.scalar_reg(t.buf)
                if c > 0:
                    self.madd_const(r, s, c)
                else:
                    self.msub_const(r, s, -c)
            elif isinstance(t, Gather):
                gv = self.gather_value(t, row)
                if c > 0:
                    self.madd_const(r, gv, c)
                else:
                    self.msub_const(r, gv, -c)
            else:
                raise NotImplementedError(f"index term {t!r}")
        return r

    def gen_index(self, aff: Affine, g) -> VReg:
        r, u = g
        extra = aff.coef(self.row_var) * r if self.row_var is not None else 0
        c = aff.coef(self.vec_var) if self.vec_var is not None else 0
        lanes = self.vector and c != 0
        if lanes:
            extra += 4 * u * c
        x = self.affine_gpr(aff, extra, r)
        d = self.vv("ix")
        self.I("dup {0}.4s, {1:w}", d, x)
        if lanes:
            self.I("add {0}.4s, {0}.4s, {1}.4s", d, self.iota(c))
        return d

    def iota(self, c: int) -> VReg:
        key = ("iota", c)
        table = self.memo.setdefault(0, {})
        r = table.get(key)
        if r is None:
            r = self.vv(f"iota{c}")
            with self.at(0):
                lanes = [(c * l) & 0xFFFFFFFF for l in range(4)]
                self.mov_imm("x16", lanes[0] | (lanes[1] << 32))
                self.I("fmov {0:d}, x16", r)
                self.mov_imm("x16", lanes[2] | (lanes[3] << 32))
                self.I("mov {0}.d[1], x16", r)
            table[key] = r
        return r

    # ------------------------------------------------------------------ random numbers
    def rand_key(self, salt: int) -> VReg:
        r = self.rand_k.get(salt)
        if r is not None:
            return r
        with self.at(0):
            if self.stream is None:
                self.stream = self.vx("stream")
                t = self.vx("t")
                self.I("adrp x16, _anvil_stream@PAGE")
                self.I("add x16, x16, _anvil_stream@PAGEOFF")
                self.I("ldr {0:w}, [x16]", self.stream)
                self.I("add {0:w}, {1:w}, #1", t, self.stream)
                self.I("str {0:w}, [x16]", t)
            k = self.vx("rk")
            self.mov_imm("x16", STREAM_MUL, w=True)
            self.I("mul {0:w}, {1:w}, w16", k, self.stream)
            self.mov_imm("x16", (salt * SALT_MUL + MIX) & 0xFFFFFFFF, w=True)
            self.I("add {0:w}, {0:w}, w16", k)
            self.I("adrp x16, _anvil_seed@PAGE")
            self.I("ldr w17, [x16, _anvil_seed@PAGEOFF]")
            self.I("add {0:w}, {0:w}, w17", k)
            r = self.vv(f"rk{salt}")
            self.I("dup {0}.4s, {1:w}", r, k)
        self.rand_k[salt] = r
        return r

    def lowbias(self, x: VReg) -> VReg:
        c1 = self.const(0x7FEB352D, I32)
        c2 = self.const(0x846CA68B, I32)
        t = self.vv()
        y = self.vv()
        self.I("ushr {0}.4s, {1}.4s, #16", t, x)
        self.I("eor {0}.16b, {1}.16b, {2}.16b", y, x, t)
        self.I("mul {0}.4s, {0}.4s, {1}.4s", y, c1)
        self.I("ushr {0}.4s, {1}.4s, #15", t, y)
        self.I("eor {0}.16b, {0}.16b, {1}.16b", y, t)
        self.I("mul {0}.4s, {0}.4s, {1}.4s", y, c2)
        self.I("ushr {0}.4s, {1}.4s, #16", t, y)
        self.I("eor {0}.16b, {0}.16b, {1}.16b", y, t)
        return y

    def gen_rand(self, e: Rand, g) -> VReg:
        strides = ir.row_major_strides([v.extent for v in self.D])
        idx = self.gen_index(Affine(0, {v: s for v, s in zip(self.D, strides)}), g)
        h = self.vv("h")
        self.I("mul {0}.4s, {1}.4s, {2}.4s", h, idx, self.const(GOLD, I32))
        self.I("add {0}.4s, {0}.4s, {1}.4s", h, self.rand_key(e.salt))
        h = self.lowbias(self.lowbias(h))
        self.I("ushr {0}.4s, {0}.4s, #8", h)
        if e.open_low:
            self.I("add {0}.4s, {0}.4s, {1}.4s", h, self.const(1, I32))
        d = self.vv("u")
        self.I("ucvtf {0}.4s, {1}.4s", d, h)
        self.I("fmul {0}.4s, {0}.4s, {1}.4s", d, self.const(2.0 ** -24))
        return d

    # ------------------------------------------------------------------ unary
    def gen_unary(self, op, a: VReg, adt) -> VReg:
        d = self.vv(op)
        isint = adt == I32
        if op == "neg":
            self.I("neg {0}.4s, {1}.4s" if isint else "fneg {0}.4s, {1}.4s", d, a)
        elif op == "abs":
            self.I("abs {0}.4s, {1}.4s" if isint else "fabs {0}.4s, {1}.4s", d, a)
        elif op == "sqrt":
            self.I("fsqrt {0}.4s, {1}.4s", d, a)
        elif op == "rsqrt":
            t = self.vv()
            self.I("fsqrt {0}.4s, {1}.4s", t, a)
            self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, self.const(1.0), t)
        elif op == "recip":
            self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, self.const(1.0), a)
        elif op == "floor":
            self.I("frintm {0}.4s, {1}.4s", d, a)
        elif op == "ceil":
            self.I("frintp {0}.4s, {1}.4s", d, a)
        elif op == "round":
            self.I("frintn {0}.4s, {1}.4s", d, a)
        elif op == "f32":
            self.I("scvtf {0}.4s, {1}.4s", d, a)
        elif op == "i32":
            self.I("fcvtzs {0}.4s, {1}.4s", d, a)
        elif op == "sign":
            p, q = self.vv(), self.vv()
            if isint:
                self.I("cmgt {0}.4s, {1}.4s, #0", p, a)
                self.I("cmlt {0}.4s, {1}.4s, #0", q, a)
                self.I("sub {0}.4s, {1}.4s, {2}.4s", d, q, p)
            else:
                self.I("fcmgt {0}.4s, {1}.4s, #0.0", p, a)
                self.I("fcmlt {0}.4s, {1}.4s, #0.0", q, a)
                self.I("sub {0}.4s, {1}.4s, {2}.4s", d, q, p)
                self.I("scvtf {0}.4s, {0}.4s", d)
        elif op == "not":
            m = self.vv()
            self.I("cmeq {0}.4s, {1}.4s, #0" if isint else "fcmeq {0}.4s, {1}.4s, #0.0", m, a)
            self.I("and {0}.16b, {1}.16b, {2}.16b", d, m, self.const(1.0))
        elif op == "exp":
            return self.vexp(a)
        elif op == "log":
            return self.vlog(a)
        elif op == "tanh":
            return self.vtanh(a)
        elif op == "sigmoid":
            n = self.vv()
            self.I("fneg {0}.4s, {1}.4s", n, a)
            ex = self.vexp(n)
            self.I("fadd {0}.4s, {1}.4s, {2}.4s", ex, ex, self.const(1.0))
            self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, self.const(1.0), ex)
        elif op == "sin":
            return self.vsincos(a, cos=False)
        elif op == "cos":
            return self.vsincos(a, cos=True)
        else:
            raise NotImplementedError(op)
        return d

    # ------------------------------------------------------------------ binary
    def ev_mask(self, c, g):
        """Mask (all-ones lanes where c is true). Returns (vreg, inverted)."""
        if isinstance(c, Binary) and c.op in ir.COMPARISONS:
            if self.is_post(c) or not self.cfg.hoist:
                return self.compare(c, g)
            lvl = min(self.level(c), self.cur_level)
            table = self.memo.setdefault(lvl, {})
            key = ("mask", c.key(), self.gkey(c, g))
            if key in table:
                return table[key]
            with self.at(lvl):
                saved = self.vector
                if self.vec_level is None or lvl < self.vec_level:
                    self.vector = self.uses_vectors
                r = self.compare(c, g)
                self.vector = saved
            table[key] = r
            return r
        v = self.ev(c, g)
        m = self.vv("m")
        self.I("cmeq {0}.4s, {1}.4s, #0" if c.dtype == I32 else "fcmeq {0}.4s, {1}.4s, #0.0", m, v)
        return m, True

    def compare(self, c: Binary, g):
        a = self.ev(c.a, g)
        b = self.ev(c.b, g)
        pre = "cm" if c.a.dtype == I32 else "fcm"
        m = self.vv("m")
        op = c.op
        if op in ("gt", "ge", "eq"):
            self.I(f"{pre}{op} {{0}}.4s, {{1}}.4s, {{2}}.4s", m, a, b)
        elif op == "lt":
            self.I(f"{pre}gt {{0}}.4s, {{1}}.4s, {{2}}.4s", m, b, a)
        elif op == "le":
            self.I(f"{pre}ge {{0}}.4s, {{1}}.4s, {{2}}.4s", m, b, a)
        elif op == "ne":
            self.I(f"{pre}eq {{0}}.4s, {{1}}.4s, {{2}}.4s", m, a, b)
            return m, True
        return m, False

    def gen_binary(self, e: Binary, g) -> VReg:
        op = e.op
        if op in ir.COMPARISONS:
            m, inv = self.ev_mask(e, g)
            d = self.vv("cmp")
            one = self.const(1.0)
            if inv:
                self.I("bic {0}.16b, {1}.16b, {2}.16b", d, one, m)
            else:
                self.I("and {0}.16b, {1}.16b, {2}.16b", d, m, one)
            return d
        if op in ("and", "or"):
            ma, ia = self.ev_mask(e.a, g)
            mb, ib = self.ev_mask(e.b, g)
            ta, tb = self.vv(), self.vv()
            self.I("mvn {0}.16b, {1}.16b" if ia else "mov {0}.16b, {1}.16b", ta, ma)
            self.I("mvn {0}.16b, {1}.16b" if ib else "mov {0}.16b, {1}.16b", tb, mb)
            t = self.vv()
            self.I("and {0}.16b, {1}.16b, {2}.16b" if op == "and" else "orr {0}.16b, {1}.16b, {2}.16b", t, ta, tb)
            d = self.vv(op)
            self.I("and {0}.16b, {1}.16b, {2}.16b", d, t, self.const(1.0))
            return d
        if op == "pow" and isinstance(e.b, Const):
            return self.gen_pow_const(self.ev(e.a, g), e.b.value)
        a = self.ev(e.a, g)
        b = self.ev(e.b, g)
        if op == "pow":
            la = self.vlog(a)
            self.I("fmul {0}.4s, {0}.4s, {1}.4s", la, b)
            return self.vexp(la)
        d = self.vv(op)
        if e.a.dtype == I32:
            table = {"add": "add", "sub": "sub", "mul": "mul", "max": "smax", "min": "smin"}
            if op in table:
                self.I(f"{table[op]} {{0}}.4s, {{1}}.4s, {{2}}.4s", d, a, b)
                return d
            if op in ("idiv", "mod"):
                return self.int_divmod(op, a, b)
            raise NotImplementedError(f"i32 {op}")
        table = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv", "max": "fmax", "min": "fmin"}
        if op in table:
            self.I(f"{table[op]} {{0}}.4s, {{1}}.4s, {{2}}.4s", d, a, b)
            return d
        if op == "idiv":
            self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, a, b)
            self.I("frintm {0}.4s, {0}.4s", d)
            return d
        if op == "mod":
            q = self.vv()
            self.I("fdiv {0}.4s, {1}.4s, {2}.4s", q, a, b)
            self.I("frintm {0}.4s, {0}.4s", q)
            self.I("mov {0}.16b, {1}.16b", d, a)
            self.I("fmls {0}.4s, {1}.4s, {2}.4s", d, q, b)
            return d
        raise NotImplementedError(op)

    def gen_pow_const(self, a: VReg, p: float) -> VReg:
        if p == int(p) and 1 <= abs(p) <= 8:
            n = int(abs(p))
            result = None
            sq = a
            while n:
                if n & 1:
                    if result is None:
                        result = sq
                    else:
                        t = self.vv()
                        self.I("fmul {0}.4s, {1}.4s, {2}.4s", t, result, sq)
                        result = t
                n >>= 1
                if n:
                    t = self.vv()
                    self.I("fmul {0}.4s, {1}.4s, {1}.4s", t, sq)
                    sq = t
            if p < 0:
                d = self.vv()
                self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, self.const(1.0), result)
                return d
            return result
        if p in (0.5, -0.5, 1.5):
            t = self.vv()
            self.I("fsqrt {0}.4s, {1}.4s", t, a)
            if p == 0.5:
                return t
            d = self.vv()
            if p == -0.5:
                self.I("fdiv {0}.4s, {1}.4s, {2}.4s", d, self.const(1.0), t)
            else:
                self.I("fmul {0}.4s, {1}.4s, {2}.4s", d, t, a)
            return d
        if p == 0:
            return self.const(1.0)
        la = self.vlog(a)
        self.I("fmul {0}.4s, {0}.4s, {1}.4s", la, self.const(float(p)))
        return self.vexp(la)

    def int_divmod(self, op, a: VReg, b: VReg) -> VReg:
        """Floor division / modulo (Python semantics), lane by lane through GPRs."""
        d = self.vv(op)
        self.I("movi {0}.4s, #0", d)
        for l in range(4 if self.vector else 1):
            self.I(f"mov w16, {{0}}.s[{l}]", a)
            self.I(f"mov w17, {{0}}.s[{l}]", b)
            q, r, t = self.vx("q"), self.vx("r"), self.vx("t")
            self.I("sdiv {0:w}, w16, w17", q)
            self.I("msub {0:w}, {1:w}, w17, w16", r, q)
            self.I("eor {0:w}, {1:w}, w17", t, r)
            self.I("cmp {0:w}, #0", r)
            self.I("ccmp {0:w}, #0, #0, ne", t)     # lt <=> r != 0 and sign(r) != sign(b)
            if op == "idiv":
                self.I("cinc {0:w}, {0:w}, ge", q)
                self.I("sub {0:w}, {0:w}, #1", q)
                self.I(f"mov {{0}}.s[{l}], {{1:w}}", d, q)
            else:
                self.I("add w16, {0:w}, w17", r)
                self.I("csel {0:w}, w16, {0:w}, lt", r)
                self.I(f"mov {{0}}.s[{l}], {{1:w}}", d, r)
        return d

    # ------------------------------------------------------------------ loops
    def counter(self, v) -> VReg:
        c = self.counters.get(v)
        if c is None:
            c = self.vx(v.name)
            self.counters[v] = c
        return c

    def emit_loop(self, v, level, start: int, stop: int, step: int, body_fn, seq: Seq):
        """for v in range(start, stop, step), emitted into seq. The trip count is static,
        except for the outermost loop of a parallel kernel, which runs while v + step <= end
        over the chunk [start, end) passed in by the runtime."""
        dynamic = self.par and level == 1
        trips = (stop - start + step - 1) // step if stop > start else 0
        if trips == 0 and not dynamic:
            return
        c = self.counter(v)
        saved_out = self.out
        self.out = seq
        if v not in self._counter_started:
            if dynamic:
                self.I("mov {0}, {1}", c, self.par_start)
            elif start == 0:
                self.mov_imm(c, 0)
        self._counter_started.add(v)
        top = self.label("loop")
        check = top + "_check"
        lim = None
        seq.add(LoopMark("begin", top))
        if dynamic:
            lim = self.par_end
            if step > 1:
                lim = self.vx("lim")
                self.I(f"sub {{0}}, {{1}}, #{step - 1}", lim, self.par_end)
            self.I(f"b {check}")
        seq.add(Label(top))
        pre, body, latch = Seq(), Seq(), Seq()
        seq.add(pre)
        seq.add(body)
        seq.add(latch)
        saved_level = self.cur_level
        self.cur_level = level
        self.lists[level] = pre
        self.memo[level] = {}
        for L in list(self.memo):
            if L > level:
                del self.memo[L]
        # pointers recomputed every iteration of this loop
        self.out = pre
        self.loop_gen[level] = self.loop_gen.get(level, 0) + 1
        for a in list(self.accesses.values()):
            if a.base_level == level:
                self.ensure_pointer(a)
        self.out = body
        body_fn(body)
        # latch: advance pointers and the counter
        self.out = latch
        for a in self.accesses.values():
            if a.base_level < level:
                cf = a.offset.coef(v)
                if cf:
                    for p in a.all_ptrs():
                        self.add_imm(p, p, 4 * cf * step)
        self.add_imm(c, c, step)
        if dynamic:
            self.out.add(Label(check))
            self.I("cmp {0}, {1}", c, lim)
        else:
            end = start + trips * step
            if end < 4096:
                self.I(f"cmp {{0}}, #{end}", c)
            else:
                self.mov_imm("x16", end)
                self.I("cmp {0}, x16", c)
        self.I(f"b.lt {top}")
        seq.add(LoopMark("end", top))
        del self.lists[level]
        self.memo.pop(level, None)
        self.cur_level = saved_level
        self.out = saved_out

    def rewind(self, v, level, total: int, seq: Seq):
        """After a loop over v covering `total` elements, restore pointers that an
        enclosing loop keeps advancing."""
        saved = self.out
        self.out = seq
        for a in self.accesses.values():
            if a.base_level < level - 1:
                cf = a.offset.coef(v)
                if cf:
                    for p in a.all_ptrs():
                        self.add_imm(p, p, -4 * cf * total)
        self._counter_started.discard(v)
        self.out = saved

    def variants(self, extent: int):
        """(start, stop, step, vector, groups) pieces covering [0, extent)."""
        U = self.cfg.U
        out = []
        pos = 0
        main = extent // (4 * U) * (4 * U)
        if main:
            out.append((0, main, 4 * U, True, U))
            pos = main
        mid = pos + (extent - pos) // 4 * 4
        if mid > pos:
            out.append((pos, mid, 4, True, 1))
            pos = mid
        if extent > pos:
            out.append((pos, extent, 1, False, 1))
        return out

    # ------------------------------------------------------------------ kernel structure
    def lower(self) -> list:
        self.collect_accesses()
        entry = Seq()
        self.lists[0] = entry
        self.memo[0] = {}
        self.out = entry
        if self.par:
            self.I("mov x16, x0")
            self.I("mov x17, x1")
            self.par_start = self.vx("start")
            self.par_end = self.vx("end")
            self.I("mov {0}, x16", self.par_start)
            self.I("mov {0}, x17", self.par_end)
        self.vector = self.uses_vectors
        self.loop_gen = {0: 1}
        for a in list(self.accesses.values()):
            if a.base_level == 0:
                self.ensure_pointer(a)
        body = Seq()
        self.out = body
        self.emit_domain(1, body)
        return entry.flatten() + body.flatten()

    def emit_domain(self, level, seq: Seq):
        if level > self.n:
            self.emit_core(seq)
            return
        v = self.D[level - 1]

        def nxt(body):
            self.emit_domain(level + 1, body)
        dynamic = self.par and level == 1
        if v is self.row_var:
            R = self.cfg.R
            main = v.extent // R * R
            pieces = ([(0, main, R, R)] if main else []) + ([(main, v.extent, 1, 1)] if v.extent > main else [])
            if dynamic:
                pieces = [(0, 0, R, R), (0, 0, 1, 1)]
            for (s, e, step, rg) in pieces:
                self.row_groups = rg
                self.emit_loop(v, level, s, e, step, nxt, seq)
            self.row_groups = 1
            self.rewind(v, level, v.extent, seq)
            return
        if self.cfg.mode == "vec_out" and v is self.vec_var:
            pieces = self.variants(v.extent)
            if dynamic:
                U = self.cfg.U
                pieces = [(0, 0, 4 * U, True, U)] + ([(0, 0, 4, True, 1)] if U > 1 else []) + [(0, 0, 1, False, 1)]
            for (s, e, step, vec, groups) in pieces:
                self.vector = vec
                self.vec_groups = groups
                self.emit_loop(v, level, s, e, step, nxt, seq)
            self.vector = self.uses_vectors
            self.vec_groups = 1
            self.rewind(v, level, v.extent, seq)
            return
        self.emit_loop(v, level, 0, v.extent, 1, nxt, seq)
        self.rewind(v, level, v.extent, seq)

    def groups(self):
        if self.cfg.mode == "vec_out":
            return [(r, u) for r in range(self.row_groups) for u in range(self.vec_groups)]
        return [(r, 0) for r in range(self.row_groups)]

    def emit_core(self, seq: Seq):
        """Reduction + statements at the innermost domain level."""
        saved = self.out
        self.out = seq
        if self.cfg.mode == "vec_red":
            self.vector = True
        elif self.cfg.mode == "scalar":
            self.vector = False
        self.memo_post = {}
        self.let_val = {}
        if self.k.red is not None:
            self.emit_reduction(seq)
        if self.cfg.mode == "vec_red":
            self.vector = False
        self.out = seq
        self.in_epilogue = self.k.red is not None
        for g in self.groups():
            for st in self.k.stmts:
                if isinstance(st, Let):
                    gk = self.gkey(st.value, g)
                    if (st, gk) not in self.let_val:
                        self.let_val[(st, gk)] = self.ev(st.value, g)
                else:
                    self.emit_store(st, g)
        self.in_epilogue = False
        self.out = saved

    def red_identity(self, op, dtype):
        if op == "sum":
            return self.const(0.0 if dtype == F32 else 0, dtype)
        if op == "prod":
            return self.const(1.0 if dtype == F32 else 1, dtype)
        if op in ("max", "argmax"):
            return self.const(-math.inf) if dtype == F32 else self.const(-2 ** 31, I32)
        if op in ("min", "argmin"):
            return self.const(math.inf) if dtype == F32 else self.const(2 ** 31 - 1, I32)
        raise NotImplementedError(op)

    def emit_reduction(self, seq: Seq):
        red = self.k.red
        op = red.op
        bdt = red.body.dtype
        if op in ("argmax", "argmin"):
            accv = self.vv("acc")
            self.I("mov {0}.16b, {1}.16b", accv, self.red_identity(op, F32))
            idx = self.vx("argidx")
            self.mov_imm(idx, 0)
            self.arg = (accv, idx)
            self.emit_red_loops(1, seq, [(0, 0)])
            d = self.vv("arg")
            self.I("dup {0}.4s, {1:w}", d, idx)
            self.acc_val = {(0, 0): d}
            return
        if self.cfg.mode == "vec_red":
            rows = range(self.row_groups)
            self.vaccs = {}
            for r in rows:
                for u in range(self.cfg.U):
                    a = self.vv(f"vacc{r}_{u}")
                    self.I("mov {0}.16b, {1}.16b", a, self.red_identity(op, bdt))
                    self.vaccs[(r, u)] = a
            self.sacc = {}
            for r in rows:
                self.sacc[r] = self.vv(f"sacc{r}")
                self.I("mov {0}.16b, {1}.16b", self.sacc[r], self.red_identity(op, bdt))
            self.emit_red_loops(1, seq, None)
            self.acc_val = {}
            for r in rows:
                acc = self.vaccs[(r, 0)]
                for u in range(1, self.cfg.U):
                    self.combine(op, bdt, acc, self.vaccs[(r, u)])
                h = self.vv("hsum")
                if op == "sum":
                    if bdt == F32:
                        self.I("faddp {0}.4s, {1}.4s, {1}.4s", h, acc)
                        self.I("faddp {0:s}, {0}.2s", h)
                    else:
                        self.I("addv {0:s}, {1}.4s", h, acc)
                elif op == "max":
                    self.I("fmaxv {0:s}, {1}.4s" if bdt == F32 else "smaxv {0:s}, {1}.4s", h, acc)
                elif op == "min":
                    self.I("fminv {0:s}, {1}.4s" if bdt == F32 else "sminv {0:s}, {1}.4s", h, acc)
                else:
                    raise NotImplementedError(op)
                self.combine(op, bdt, h, self.sacc[r])
                self.acc_val[(r, 0)] = h
            return
        groups = self.groups()
        self.accs = {}
        for g in groups:
            a = self.vv(f"acc{g[0]}_{g[1]}")
            if op == "sum":
                self.I("movi {0}.4s, #0", a)      # (no shared zero register live across the tile)
            else:
                self.I("mov {0}.16b, {1}.16b", a, self.red_identity(op, bdt))
            self.accs[g] = a
        self.emit_red_loops(1, seq, groups)
        self.acc_val = dict(self.accs)

    def combine(self, op, dtype, acc: VReg, val: VReg):
        if dtype == F32:
            ins = {"sum": "fadd", "max": "fmax", "min": "fmin", "prod": "fmul"}[op]
        else:
            ins = {"sum": "add", "max": "smax", "min": "smin", "prod": "mul"}[op]
        self.I(f"{ins} {{0}}.4s, {{0}}.4s, {{1}}.4s", acc, val)

    def emit_red_loops(self, j, seq: Seq, groups):
        """Loop over reduction var j (1-based); the innermost body accumulates.
        In vec_red mode `groups` is None and decided per variant."""
        if j > self.m:
            self.emit_accumulate(groups)
            return
        v = self.R[j - 1]
        level = self.n + j
        if self.cfg.mode == "vec_red" and v is self.vec_var:
            for (s, e, step, vec, vg) in self.variants(v.extent):
                self.vector = vec
                gs = [(r, u) for r in range(self.row_groups) for u in range(vg)] if vec else "tail"
                self.emit_loop(v, level, s, e, step,
                               lambda body, gs=gs: self.emit_red_loops(j + 1, body, gs), seq)
            self.vector = True
            self.rewind(v, level, v.extent, seq)
            return
        self.emit_loop(v, level, 0, v.extent, 1, lambda body: self.emit_red_loops(j + 1, body, groups), seq)
        self.rewind(v, level, v.extent, seq)

    def emit_accumulate(self, groups):
        red = self.k.red
        op = red.op
        body = red.body
        if op in ("argmax", "argmin"):
            accv, idx = self.arg
            b = self.ev(body, (0, 0))
            if body.dtype == I32:
                t = self.vv()
                self.I("scvtf {0}.4s, {1}.4s", t, b)
                b = t
            self.I("fcmp {0:s}, {1:s}", b, accv)
            cond = "gt" if op == "argmax" else "mi"
            self.I(f"fcsel {{0:s}}, {{1:s}}, {{0:s}}, {cond}", accv, b)
            self.I(f"csel {{0}}, {{1}}, {{0}}, {cond}", idx, self.counter(self.R[0]))
            return
        if self.cfg.mode == "vec_red":
            if groups == "tail":
                pairs = [(self.sacc[r], (r, 0)) for r in range(self.row_groups)]
            else:
                pairs = [(self.vaccs[g], g) for g in groups]
        else:
            pairs = [(self.accs[g], g) for g in groups]
        fma = op == "sum" and isinstance(body, Binary) and body.op == "mul" and body.dtype == F32
        if fma and self.vector:
            va, vb = self.varies(body.a), self.varies(body.b)
            if va != vb:
                uni, var = (body.a, body.b) if not va else (body.b, body.a)
                for acc, g in pairs:
                    s = self.ev_s0(uni, g) if isinstance(uni, Load) else self.ev(uni, g)
                    w = self.ev(var, g)
                    self.I("fmla {0}.4s, {1}.4s, {2}.s[0]", acc, w, s)
                return
        for acc, g in pairs:
            if fma:
                self.I("fmla {0}.4s, {1}.4s, {2}.4s", acc, self.ev(body.a, g), self.ev(body.b, g))
            else:
                self.combine(op, body.dtype, acc, self.ev(body, g))

    def emit_store(self, st: Store, g):
        a = self.access(st.buf, st.offset)
        v = self.ev(st.value, g)
        isint = st.buf.dtype == I32
        if not self.vector:
            base, off = self.ptr_for(a, g, False)
            if st.accumulate:
                t = self.vv()
                self.mem("ldr", t, base, off, "s")
                self.I("add {0}.4s, {0}.4s, {1}.4s" if isint else "fadd {0}.4s, {0}.4s, {1}.4s", t, v)
                v = t
            self.mem("str", v, base, off, "s")
            return
        if st.offset.coef(self.vec_var) != 1:
            raise RuntimeError("vector store must be contiguous")
        base, off = self.ptr_for(a, g, True)
        if st.accumulate:
            t = self.vv()
            self.mem("ldr", t, base, off, "q")
            self.I("add {0}.4s, {0}.4s, {1}.4s" if isint else "fadd {0}.4s, {0}.4s, {1}.4s", t, v)
            v = t
        self.mem("str", v, base, off, "q")


class _At:
    def __init__(self, low: Lowering, level: int):
        self.low = low
        self.level = level

    def __enter__(self):
        self.saved = self.low.out
        self.saved_level = self.low.cur_level
        self.low.out = self.low.lists[self.level]
        self.low.cur_level = self.level
        return self

    def __exit__(self, *exc):
        self.low.out = self.saved
        self.low.cur_level = self.saved_level
        return False


# ----------------------------------------------------------------------------- schedule selection

def feasible_vec_out(k: Kernel) -> bool:
    if not k.domain:
        return False
    v = k.domain[-1]
    if k.red is not None and k.red.op in ("argmax", "argmin", "prod"):
        return False
    for s in k.stores:
        if s.offset.coef(v) != 1:
            return False
        for g in s.offset.gathers():
            if v in g.load.offset.vars():
                return False
    for e in k.exprs():
        for x in ir.iter_expr(e):
            gathers = x.offset.gathers() if isinstance(x, Load) else x.affine.gathers() if isinstance(x, Index) else []
            if any(v in g.load.offset.vars() for g in gathers):
                return False                  # a different gathered value in each lane
            if isinstance(x, Binary) and x.op in ("idiv", "mod") and x.dtype == I32 and depends_on(x, v):
                return False      # integer division has no vector instruction (invariant ones are hoisted)
    return True


def depends_on(e, v) -> bool:
    """Whether e can take different values as loop variable v changes."""
    for x in ir.iter_expr(e):
        if isinstance(x, Load) and v in x.offset.vars():
            return True
        if isinstance(x, Index) and v in x.affine.vars():
            return True
        if isinstance(x, (Rand, Acc)):
            return True
        if isinstance(x, LetRef) and depends_on(x.let.value, v):
            return True
    return False


def strided_loads(k: Kernel, v) -> int:
    return sum(1 for e in k.exprs() for ld in ir.loads_in(e) if ld.offset.coef(v) not in (0, 1))


def feasible_vec_red(k: Kernel) -> bool:
    if k.red is None or k.red.op not in ("sum", "max", "min"):
        return False
    v = k.red.vars[-1]
    for x in ir.iter_expr(k.red.body):
        if isinstance(x, Load):
            if x.offset.coef(v) not in (0, 1):
                return False
            for g in x.offset.gathers():
                if v in g.load.offset.vars():
                    return False
        if isinstance(x, Index) and any(v in g.load.offset.vars() for g in x.affine.gathers()):
            return False
        if isinstance(x, Rand):
            return False
        if isinstance(x, Binary) and x.op in ("idiv", "mod") and x.dtype == I32 and depends_on(x, v):
            return False
    return True


def choose_configs(k: Kernel) -> list[Config]:
    """Schedules to try, best first (later ones need fewer registers)."""
    out = []
    size = sum(1 for e in k.exprs() for _ in ir.iter_expr(e))
    vo = feasible_vec_out(k)
    vr = feasible_vec_red(k)
    n_last = k.domain[-1].extent if k.domain else 0
    r_last = k.red.vars[-1].extent if k.red else 0
    rows = k.domain[-2].extent if len(k.domain) >= 2 else 0
    awkward_out = n_last < 16 and n_last % 4 != 0          # e.g. 5 wide: vectors waste lanes
    prefer_red = vr and r_last >= 8 and (not vo or strided_loads(k, k.domain[-1]) > 0 or n_last < 4
                                         or (awkward_out and r_last >= 2 * n_last))
    big_u = 4 if size <= 40 else 2
    if prefer_red:
        red_nodes = sum(1 for _ in ir.iter_expr(k.red.body)) if k.red is not None else 0
        if rows >= 4 and red_nodes <= 8:
            # several outputs' dot products at once: loads that do not depend on the row are shared
            out += [Config("vec_red", 4, 4), Config("vec_red", 2, 4)]
        out += [Config("vec_red", big_u), Config("vec_red", 1)]
    red_size = sum(1 for _ in ir.iter_expr(k.red.body)) if k.red is not None else 0
    if vo and n_last >= 4:
        # (an epilogue, such as a fused optimizer update, runs once per output and does not count)
        if k.red is not None and rows >= 4 and red_size <= 24:
            u = 4 if n_last >= 16 else (2 if n_last >= 8 else 1)
            # without hoisting, constants are made where they are used: a fused epilogue (an
            # optimizer update) then still fits next to the 16 accumulators of a 4×16 tile
            out += [Config("vec_out", u, 4), Config("vec_out", u, 4, hoist=False), Config("vec_out", min(u, 2), 4)]
        out += [Config("vec_out", big_u), Config("vec_out", 2), Config("vec_out", 1)]
    if vr and not prefer_red and r_last >= 4:
        out += [Config("vec_red", 1)]
    out += [Config("scalar", 1), Config("scalar", 1, hoist=False)]
    seen, uniq = set(), []
    for c in out:
        key = (c.mode, c.U, c.R, c.hoist)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq
