"""Inline NEON implementations of transcendental functions (4 × f32 lanes).

No libm calls: every function is a short branch-free instruction sequence that the
register allocator schedules together with the surrounding kernel.

    exp   Cody–Waite range reduction to |r| ≤ ln2/2, degree-7 polynomial, 2^n by exponent bits
    log   exponent/mantissa split, m ∈ [√½, √2), atanh series in s = (m-1)/(m+1)
    tanh  via expm1(2|x|) (accurate near 0), sign copied back
    sin/cos  reduction by π/2 (3-part constant), Taylor polynomials, quadrant select
"""
from __future__ import annotations

import math

from ..ir import I32


class VecMath:
    def poly(self, x, coefs: list[float]):
        """Horner: c0 + x*(c1 + x*(c2 + ...)) with fused multiply-adds."""
        p = self.vv("p")
        self.I("mov {0}.16b, {1}.16b", p, self.const(coefs[-1]))
        for c in reversed(coefs[:-1]):
            q = self.vv("p")
            self.I("mov {0}.16b, {1}.16b", q, self.const(c))
            self.I("fmla {0}.4s, {1}.4s, {2}.4s", q, p, x)
            p = q
        return p

    def exp_core(self, t, minus_one: bool = False):
        """t already clamped. Returns (p, scale) with exp(t) = p * scale, or with
        minus_one: (q, scale) with exp(t) - 1 = scale*q + (scale - 1)."""
        n = self.vv("n")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", n, t, self.const(1.4426950408889634))
        self.I("frintn {0}.4s, {0}.4s", n)
        r = self.vv("r")
        self.I("mov {0}.16b, {1}.16b", r, t)
        self.I("fmls {0}.4s, {1}.4s, {2}.4s", r, n, self.const(0.693359375))
        self.I("fmls {0}.4s, {1}.4s, {2}.4s", r, n, self.const(-2.12194440e-4))
        if minus_one:
            inner = self.poly(r, [1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040])
            p = self.vv("q")
            self.I("fmul {0}.4s, {1}.4s, {2}.4s", p, inner, r)
        else:
            p = self.poly(r, [1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040])
        ni = self.vv("ni")
        self.I("fcvtzs {0}.4s, {1}.4s", ni, n)
        self.I("add {0}.4s, {0}.4s, {1}.4s", ni, self.const(127, I32))
        self.I("shl {0}.4s, {0}.4s, #23", ni)
        return p, ni

    def vexp(self, x):
        t = self.vv("t")
        self.I("fmin {0}.4s, {1}.4s, {2}.4s", t, x, self.const(88.0))
        self.I("fmax {0}.4s, {0}.4s, {1}.4s", t, self.const(-87.33))
        p, scale = self.exp_core(t)
        d = self.vv("exp")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", d, p, scale)
        return d

    def vlog(self, x):
        e = self.vv("e")
        self.I("ushr {0}.4s, {1}.4s, #23", e, x)
        self.I("sub {0}.4s, {0}.4s, {1}.4s", e, self.const(127, I32))
        mnt = self.vv("mant")
        self.I("and {0}.16b, {1}.16b, {2}.16b", mnt, x, self.const(0x007FFFFF, I32))
        self.I("orr {0}.16b, {0}.16b, {1}.16b", mnt, self.const(0x3F800000, I32))
        big = self.vv("big")
        self.I("fcmgt {0}.4s, {1}.4s, {2}.4s", big, mnt, self.const(1.41421356))
        half = self.vv("half")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", half, mnt, self.const(0.5))
        self.I("bit {0}.16b, {1}.16b, {2}.16b", mnt, half, big)
        self.I("sub {0}.4s, {0}.4s, {1}.4s", e, big)          # mask is -1: e + 1 where big
        f = self.vv("f")
        self.I("fsub {0}.4s, {1}.4s, {2}.4s", f, mnt, self.const(1.0))
        den = self.vv("den")
        self.I("fadd {0}.4s, {1}.4s, {2}.4s", den, mnt, self.const(1.0))
        s = self.vv("s")
        self.I("fdiv {0}.4s, {1}.4s, {2}.4s", s, f, den)
        z = self.vv("z")
        self.I("fmul {0}.4s, {1}.4s, {1}.4s", z, s)
        p = self.poly(z, [2.0, 2 / 3, 2 / 5, 2 / 7, 2 / 9, 2 / 11])
        lm = self.vv("lm")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", lm, p, s)
        ef = self.vv("ef")
        self.I("scvtf {0}.4s, {1}.4s", ef, e)
        d = self.vv("log")
        self.I("mov {0}.16b, {1}.16b", d, lm)
        self.I("fmla {0}.4s, {1}.4s, {2}.4s", d, ef, self.const(0.6931471805599453))
        # special cases: x == 0 -> -inf, x < 0 -> nan, x == inf -> inf, nan -> nan
        m = self.vv("m")
        self.I("fcmeq {0}.4s, {1}.4s, #0.0", m, x)
        self.I("bit {0}.16b, {1}.16b, {2}.16b", d, self.const(-math.inf), m)
        self.I("fcmlt {0}.4s, {1}.4s, #0.0", m, x)
        self.I("bit {0}.16b, {1}.16b, {2}.16b", d, self.const(math.nan), m)
        self.I("fcmeq {0}.4s, {1}.4s, {2}.4s", m, x, self.const(math.inf))
        self.I("bit {0}.16b, {1}.16b, {2}.16b", d, self.const(math.inf), m)
        self.I("fcmeq {0}.4s, {1}.4s, {1}.4s", m, x)
        self.I("bif {0}.16b, {1}.16b, {2}.16b", d, x, m)
        return d

    def vtanh(self, x):
        a = self.vv("a")
        self.I("fabs {0}.4s, {1}.4s", a, x)
        self.I("fmin {0}.4s, {0}.4s, {1}.4s", a, self.const(9.0))
        u = self.vv("u")
        self.I("fadd {0}.4s, {1}.4s, {1}.4s", u, a)
        q, scale = self.exp_core(u, minus_one=True)
        em1 = self.vv("em1")
        self.I("fsub {0}.4s, {1}.4s, {2}.4s", em1, scale, self.const(1.0))
        self.I("fmla {0}.4s, {1}.4s, {2}.4s", em1, q, scale)
        den = self.vv("den")
        self.I("fadd {0}.4s, {1}.4s, {2}.4s", den, em1, self.const(2.0))
        t = self.vv("tanh")
        self.I("fdiv {0}.4s, {1}.4s, {2}.4s", t, em1, den)
        self.I("bit {0}.16b, {1}.16b, {2}.16b", t, x, self.const(0x80000000, I32))   # copy sign
        m = self.vv("m")
        self.I("fcmeq {0}.4s, {1}.4s, {1}.4s", m, x)
        self.I("bif {0}.16b, {1}.16b, {2}.16b", t, x, m)
        return t

    def vsincos(self, x, cos: bool):
        k = self.vv("k")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", k, x, self.const(0.6366197723675814))
        self.I("frintn {0}.4s, {0}.4s", k)
        r = self.vv("r")
        self.I("mov {0}.16b, {1}.16b", r, x)
        self.I("fmls {0}.4s, {1}.4s, {2}.4s", r, k, self.const(1.5703125))
        self.I("fmls {0}.4s, {1}.4s, {2}.4s", r, k, self.const(4.837512969970703125e-4))
        self.I("fmls {0}.4s, {1}.4s, {2}.4s", r, k, self.const(7.54978995489188216e-8))
        r2 = self.vv("r2")
        self.I("fmul {0}.4s, {1}.4s, {1}.4s", r2, r)
        sp = self.poly(r2, [1.0, -1 / 6, 1 / 120, -1 / 5040, 1 / 362880])
        s = self.vv("sin_r")
        self.I("fmul {0}.4s, {1}.4s, {2}.4s", s, sp, r)
        c = self.poly(r2, [1.0, -1 / 2, 1 / 24, -1 / 720, 1 / 40320, -1 / 3628800])
        q = self.vv("q")
        self.I("fcvtzs {0}.4s, {1}.4s", q, k)
        if cos:
            self.I("add {0}.4s, {0}.4s, {1}.4s", q, self.const(1, I32))
        sw = self.vv("swap")
        self.I("cmtst {0}.4s, {1}.4s, {2}.4s", sw, q, self.const(1, I32))
        d = self.vv("sincos")
        self.I("mov {0}.16b, {1}.16b", d, sw)
        self.I("bsl {0}.16b, {1}.16b, {2}.16b", d, c, s)
        ng = self.vv("neg")
        self.I("cmtst {0}.4s, {1}.4s, {2}.4s", ng, q, self.const(2, I32))
        self.I("and {0}.16b, {0}.16b, {1}.16b", ng, self.const(0x80000000, I32))
        self.I("eor {0}.16b, {0}.16b, {1}.16b", d, ng)
        return d
