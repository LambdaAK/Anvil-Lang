"""Test helpers: compile text, run on the interpreter, finite-difference gradient checks."""
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from anvil import ir  # noqa: E402
from anvil.driver import compile_text  # noqa: E402
from anvil.interp import Interpreter  # noqa: E402


def run_text(text, optimize=True, float_dtype=np.float32, path="<test>.anvil"):
    c = compile_text(text, path=path, optimize=optimize)
    out = io.StringIO()
    it = Interpreter(c.program, out=out, float_dtype=float_dtype)
    it.run()
    return out.getvalue(), it, c


def buffer_of(c, name):
    b = c.elab.globals.vars[name]
    return b.val.buf


def gradcheck(text, params, grads, loss="loss", n_checks=6, h=1e-6, rtol=2e-4, atol=1e-7, seed=0):
    """`params` are names of tensors in the program; `grads` the names their gradients are bound to."""
    c = compile_text(text, path="<gradcheck>.anvil", optimize=False)
    prog = c.program
    it = Interpreter(prog, out=io.StringIO(), float_dtype=np.float64)
    it.run()
    pbufs = [buffer_of(c, p) for p in params]
    gbufs = [buffer_of(c, g) for g in grads]
    lbuf = buffer_of(c, loss)
    analytic = [it.arr(g)[:g.numel].copy() for g in gbufs]
    # forward statements = everything before the first gradient kernel
    stmts = prog.main.stmts
    first_grad = len(stmts)
    for i, st in enumerate(stmts):
        if isinstance(st, ir.KernelStmt) and any(s.buf.root.kind == "grad" for s in st.kernel.stores):
            first_grad = i
            break
    fwd = stmts[:first_grad]

    def f(pbuf, idx, delta):
        it2 = Interpreter(prog, out=io.StringIO(), float_dtype=np.float64)
        for st in fwd:
            it2.exec_stmt(st)
            writes = set()
            if isinstance(st, ir.KernelStmt):
                writes = {s.buf.root for s in st.kernel.stores}
            if pbuf.root in writes or (pbuf.kind in ("const",) and st is fwd[0]):
                it2.arr(pbuf)[idx] += delta
        if pbuf.kind == "const" and not fwd:
            pass
        return float(it2.arr(lbuf)[0])

    rng = np.random.default_rng(seed)
    for pbuf, g in zip(pbufs, analytic):
        n = pbuf.numel
        picks = rng.choice(n, size=min(n_checks, n), replace=False)
        for idx in picks:
            num = (f(pbuf, idx, h) - f(pbuf, idx, -h)) / (2 * h)
            ana = g[idx]
            if not np.isclose(ana, num, rtol=rtol, atol=atol):
                raise AssertionError(f"gradient mismatch for {pbuf.name}[{idx}]: autodiff {ana:.8g} vs "
                                     f"finite difference {num:.8g}")
    return analytic


# ----------------------------------------------------------------------------- native backend
import re
import subprocess
import tempfile

_NUM = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+(?:e[+-]?\d+)?|-?inf|nan")


def run_native(text, path="<test>.anvil", optimize=True, seed=0, env=None, stdin=None, consts=None):
    from anvil.backend.aarch64 import generate
    from anvil.backend.toolchain import assemble_and_link
    c = compile_text(text, path=path, optimize=optimize, consts=consts)
    asm, _ = generate(c.program, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, "prog.s")
        exe = os.path.join(d, "prog")
        with open(s, "w") as f:
            f.write(asm)
        assemble_and_link(s, exe)
        full_env = dict(os.environ, **(env or {}))
        r = subprocess.run([exe], capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                           env=full_env, input=stdin if stdin is not None else "")
    return r.stdout, r.stderr, r.returncode, c


def outputs_match(a: str, b: str, rtol=2e-3, atol=2e-3):
    """Same text, numbers equal up to tolerance."""
    ta = _NUM.split(a)
    tb = _NUM.split(b)
    if ta != tb:
        return False, f"text differs:\n--- native\n{a}\n--- interp\n{b}"
    na = _NUM.findall(a)
    nb = _NUM.findall(b)
    for x, y in zip(na, nb):
        fx, fy = float(x), float(y)
        if fx != fy and not (abs(fx - fy) <= atol + rtol * abs(fy)):
            if not (x == y):
                return False, f"{x} vs {y}\n--- native\n{a}\n--- interp\n{b}"
    return True, ""


def check_native(text, stdin=None, **kw):
    out, err, code, c = run_native(text, stdin=stdin, **kw)
    if code != 0:
        raise AssertionError(f"native program failed (exit {code}):\n{err}\n{out}")
    ref = io.StringIO()
    Interpreter(c.program, out=ref, inp=io.StringIO(stdin or "")).run()
    ok, msg = outputs_match(out, ref.getvalue())
    assert ok, msg
    return out


# ----------------------------------------------------------------------------- CUDA backend (emulated on the CPU)
def run_cuda(text, path="<test>.anvil", optimize=True, seed=0, stdin=None, consts=None):
    """Compile to CUDA, build the .cu as C++ (-DANVIL_EMULATE), and run it."""
    from anvil.backend.cuda import build, generate
    c = compile_text(text, path=path, optimize=optimize, consts=consts)
    src = generate(c.program, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "prog")
        build(src, exe, emulate=True)
        r = subprocess.run([exe], capture_output=True, encoding="utf-8", errors="replace", timeout=300,
                           input=stdin if stdin is not None else "")
    return r.stdout, r.stderr, r.returncode, c


def check_cuda(text, stdin=None, **kw):
    out, err, code, c = run_cuda(text, stdin=stdin, **kw)
    if code != 0:
        raise AssertionError(f"CUDA program failed (exit {code}):\n{err}\n{out}")
    ref = io.StringIO()
    Interpreter(c.program, out=ref, inp=io.StringIO(stdin or "")).run()
    ok, msg = outputs_match(out, ref.getvalue())
    assert ok, msg.replace("--- native", "--- cuda")
    return out


# ----------------------------------------------------------------------------- Metal backend (the Apple GPU)
def run_metal(text, path="<test>.anvil", optimize=True, seed=0, stdin=None, consts=None):
    from anvil.backend.metal import build, generate
    c = compile_text(text, path=path, optimize=optimize, consts=consts)
    src = generate(c.program, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "prog")
        build(src, exe)
        r = subprocess.run([exe], capture_output=True, encoding="utf-8", errors="replace", timeout=300,
                           input=stdin if stdin is not None else "")
    return r.stdout, r.stderr, r.returncode, c


def check_metal(text, stdin=None, **kw):
    out, err, code, c = run_metal(text, stdin=stdin, **kw)
    if code != 0:
        raise AssertionError(f"Metal program failed (exit {code}):\n{err}\n{out}")
    ref = io.StringIO()
    Interpreter(c.program, out=ref, inp=io.StringIO(stdin or "")).run()
    ok, msg = outputs_match(out, ref.getvalue())
    assert ok, msg.replace("--- native", "--- metal")
    return out
