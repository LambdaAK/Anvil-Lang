"""Anvil against NumPy (with Apple Accelerate) on the operations a training loop spends its time in.

    python3 benchmarks/run.py            # everything (about a minute)
    python3 benchmarks/run.py softmax    # benchmarks whose name contains "softmax"

Each benchmark is the same computation twice: an Anvil program, compiled to native code, that
times REPS repetitions with clock(), and NumPy code timed with perf_counter. Both finish with a
reduction to one number, so neither side can skip work (Anvil fuses that reduction into the
computation; NumPy makes a separate pass). Times are the best of three runs. The results go to
stdout as a Markdown table, and to benchmarks/results.md.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANVIL = os.path.join(ROOT, "bin", "anvil")

TIMER = """
start = clock()
for rep in range(REPS):
{body}
print("{{(clock() - start) / REPS * 1000:.4f}}")
"""


def anvil_program(setup: str, body: str, reps: int) -> str:
    body = "\n".join("    " + line for line in body.strip().splitlines())
    return f"const REPS = {reps}\nchk = 0.0\n{setup.strip()}\n" + TIMER.format(body=body) + "print(chk)\n"


def time_anvil(src: str, runs: int = 3) -> float:
    """Milliseconds per repetition: the best of `runs` runs of the compiled program."""
    with tempfile.NamedTemporaryFile("w", suffix=".anvil", dir=HERE, delete=False) as f:
        f.write(src)
        path = f.name
    try:
        best = float("inf")
        for _ in range(runs):
            r = subprocess.run([ANVIL, "run", path], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"anvil failed:\n{r.stderr}\n{src}")
            best = min(best, float(r.stdout.split()[-2]))
        return best
    finally:
        os.unlink(path)


def time_numpy(fn, reps: int, runs: int = 3) -> float:
    fn()                                                    # warm up
    best = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t) / reps * 1000)
    return best


rng = np.random.default_rng(0)
load_at_start = 0.0


def randn(*shape):
    return rng.standard_normal(shape, dtype=np.float32)


def np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def np_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x * x * x)))


def bench_matmul(n=1024):
    A, B = randn(n, n), randn(n, n)
    setup = f"A: [{n}, {n}] ~ normal(0, 1)\nB: [{n}, {n}] ~ normal(0, 1)"
    return dict(
        name=f"matmul {n}×{n}×{n}", what="C = A @ B, then mean(C)",
        anvil=anvil_program(setup, "chk = chk + mean(A @ B)", 10),
        numpy=(lambda: (A @ B).mean(), 10), flops=2 * n ** 3)


def bench_linear(n=256, d=1024):
    x, W, b = randn(n, d), randn(d, d) * 0.03, randn(d)
    setup = f"x: [{n}, {d}] ~ normal(0, 1)\nW: [{d}, {d}] ~ normal(0, 0.03)\nb: [{d}] ~ normal(0, 1)"
    return dict(
        name=f"linear + gelu {n}×{d}→{d}", what="gelu(x @ W + b), then sum",
        anvil=anvil_program(setup, "chk = chk + sum(gelu(x @ W + b))", 20),
        numpy=(lambda: np_gelu(x @ W + b).sum(), 20), flops=2 * n * d * d)


def bench_elementwise(n=1 << 22):
    x = randn(n)
    setup = f"x: [{n}] ~ normal(0, 1)"
    return dict(
        name="elementwise, 4M floats", what="gelu(1.01 x + 0.1) · sigmoid(x), then sum",
        anvil=anvil_program(setup, "chk = chk + sum(gelu(x * 1.01 + 0.1) * sigmoid(x))", 20),
        numpy=(lambda: (np_gelu(x * np.float32(1.01) + np.float32(0.1)) * (1 / (1 + np.exp(-x)))).sum(), 20),
        bytes=4 * n)


def bench_softmax(n=4096, d=1024):
    x = randn(n, d)
    setup = f"x: [{n}, {d}] ~ normal(0, 1)\nw: [{d}] ~ normal(0, 1)"
    w = randn(d)
    return dict(
        name=f"softmax {n}×{d}", what="softmax over rows, then a weighted sum",
        anvil=anvil_program(setup, "chk = chk + sum(softmax(x) * w)", 20),
        numpy=(lambda: (np_softmax(x) * w).sum(), 20), bytes=4 * n * d)


def bench_layernorm(n=4096, d=1024):
    x, w = randn(n, d), randn(d)

    def np_ln():
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (((x - mu) / np.sqrt(var + np.float32(1e-5))) * w).sum()
    setup = f"x: [{n}, {d}] ~ normal(0, 1)\nw: [{d}] ~ normal(0, 1)"
    return dict(
        name=f"layer norm {n}×{d}", what="layer_norm over rows, then a weighted sum",
        anvil=anvil_program(setup, "chk = chk + sum(layer_norm(x) * w)", 20),
        numpy=(np_ln, 20), bytes=4 * n * d)


def bench_attention(b=16, t=256, d=64):
    q, k, v = randn(b, t, d), randn(b, t, d), randn(b, t, d)
    mask = np.where(np.arange(t)[None, :] <= np.arange(t)[:, None], 0, -1e9).astype(np.float32)

    def np_attn():
        s = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(d)) + mask
        return (np_softmax(s) @ v).sum()
    setup = f"""q: [{b}, {t}, {d}] ~ normal(0, 1)
k: [{b}, {t}, {d}] ~ normal(0, 1)
v: [{b}, {t}, {d}] ~ normal(0, 1)"""
    body = f"""s[n, i, j] = sum q[n, i, c] * k[n, j, c] / {d ** 0.5} + (0.0 if j <= i else -1e9)
p = softmax(s)
o[n, i, c] = sum p[n, i, j] * v[n, j, c]
chk = chk + sum(o)"""
    return dict(
        name=f"causal attention {b}×{t}×{d}", what="softmax(q kᵀ/√d + mask) v, then sum",
        anvil=anvil_program(setup, body, 20), numpy=(np_attn, 20), flops=4 * b * t * t * d)


BENCHES = [bench_matmul, bench_linear, bench_elementwise, bench_softmax, bench_layernorm, bench_attention]


FROM_PYTHON = [
    ("softmax 4096×1024", "fn f(x: [n, d]) = softmax(x)", lambda: [randn(4096, 1024)], lambda x: np_softmax(x)),
    ("layer norm 4096×1024", "fn f(x: [n, d], w: [d]) = layer_norm(x) * w",
     lambda: [randn(4096, 1024), randn(1024)],
     lambda x, w: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + np.float32(1e-5)) * w),
    ("gelu(x @ W + b) 256×1024→1024", "fn f(x: [n, d], W: [d, e], b: [e]) = gelu(x @ W + b)",
     lambda: [randn(256, 1024), randn(1024, 1024) * 0.03, randn(1024)], lambda x, W, b: np_gelu(x @ W + b)),
    ("causal attention 16×256×64", """
fn f(q: [b, t, d], k: [b, t, d], v: [b, t, d]):
    s[n, i, j] = sum q[n, i, c] * k[n, j, c] / sqrt(d) + (0.0 if j <= i else -1e9)
    return softmax(s) @ v
""", lambda: [randn(16, 256, 64) for _ in range(3)],
     lambda q, k, v: np_softmax(q @ k.transpose(0, 2, 1) / np.float32(8) + np.where(
         np.arange(256)[None, :] <= np.arange(256)[:, None], 0, -1e9).astype(np.float32)) @ v),
]


def from_python():
    """anvil.function called on NumPy arrays (arguments and results are not copied)."""
    sys.path.insert(0, ROOT)
    import anvil
    rows = []
    for name, src, make, ref in FROM_PYTHON:
        args = make()
        f = anvil.function(src)
        got, want = f(*args), ref(*args)
        assert np.allclose(got, want, rtol=1e-3, atol=1e-4), name
        t_anvil = time_numpy(lambda: f(*args), 20)
        t_np = time_numpy(lambda: ref(*args), 20)
        rows.append((name, t_anvil, t_np))
        print(f"{'python: ' + name:32s} anvil {t_anvil:8.3f} ms   numpy {t_np:8.3f} ms   {t_np / t_anvil:5.2f}×", flush=True)
    return rows


def mnist_epochs():
    """The MNIST example against benchmarks/mnist_numpy.py: seconds per training epoch (5 epochs; the
    test-set evaluations are not counted)."""
    data = os.path.join(ROOT, "examples", "data", "train-images-idx3-ubyte.gz")
    if not os.path.exists(data):
        return None

    def best_of_3(cmd, parse):
        best = float("inf")
        for _ in range(3):
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(ROOT, "examples")).stdout
            best = min(best, parse(out))
        return best
    # experiments/mnist_sweep.anvil with its defaults is examples/mnist.anvil, timed to the millisecond
    anvil = best_of_3([ANVIL, "run", os.path.join(ROOT, "experiments", "mnist_sweep.anvil"), "--set", "EPOCHS=5"],
                    lambda out: float(out.split("result")[1].split()[1]))
    npy = best_of_3([sys.executable, os.path.join(HERE, "mnist_numpy.py")],
                    lambda out: float(out.split("training:")[1].split()[0]))
    return anvil, npy


def compile_times():
    """Seconds from source to executable with an empty cache (front end, optimizer, assembler)."""
    out = []
    for name in ["mnist", "transformer", "checkers"]:
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as d:
            t = time.perf_counter()
            subprocess.run([ANVIL, "build", os.path.join(ROOT, "examples", f"{name}.anvil"), "-o", os.path.join(d, "x")],
                           capture_output=True, env=dict(os.environ, ANVIL_CACHE=cache), cwd=os.path.join(ROOT, "examples"))
            out.append((name, time.perf_counter() - t))
    return out


def main():
    global load_at_start
    load_at_start = os.getloadavg()[0]
    pick = sys.argv[1:]
    rows = []
    for make in BENCHES:
        b = make()
        if pick and not any(p in b["name"] for p in pick):
            continue
        t_anvil = time_anvil(b["anvil"])
        t_np = time_numpy(*b["numpy"])
        if "flops" in b:
            rate = lambda ms: f"{b['flops'] / ms / 1e6:.0f} GFLOP/s"
        else:
            rate = lambda ms: f"{b['bytes'] / ms / 1e6:.1f} GB/s"
        rows.append((b["name"], b["what"], t_anvil, t_np, rate(t_anvil), rate(t_np)))
        print(f"{b['name']:32s} anvil {t_anvil:8.3f} ms   numpy {t_np:8.3f} ms   {t_np / t_anvil:5.2f}×", flush=True)
    load = os.getloadavg()[0]
    lines = [f"Measured on {platform.machine()} {cpu_name()}, NumPy {np.__version__} "
             f"(BLAS: {blas_name()}), {time.strftime('%Y-%m-%d')}. Best of three runs. Load average at the "
             f"start: {load_at_start:.1f}, at the end: {load:.1f} (other programs' work makes both sides slower and "
             f"noisier).", "",
             "| benchmark | computation | Anvil | NumPy | Anvil speed-up |", "|---|---|---|---|---|"]
    for name, what, te, tn, re_, rn in rows:
        lines.append(f"| {name} | {what} | {te:.2f} ms ({re_}) | {tn:.2f} ms ({rn}) | **{tn / te:.2f}×** |")
    if not pick:
        m = mnist_epochs()
        if m is not None:
            print(f"{'MNIST MLP epoch':32s} anvil {m[0]:8.3f} s    numpy {m[1]:8.3f} s    {m[1] / m[0]:5.2f}×", flush=True)
            lines.append(f"| MNIST MLP, one epoch | 784-128-10, batch 64, SGD (forward, backward, update) | "
                         f"{m[0]:.3f} s | {m[1]:.3f} s | **{m[1] / m[0]:.2f}×** |")
        lines += ["", "Called from Python: `anvil.function(source)(arrays)` against the same NumPy code "
                  "(time per call, results checked against NumPy).", "",
                  "| function | anvil.function | NumPy | Anvil speed-up |", "|---|---|---|---|"]
        for name, te, tn in from_python():
            lines.append(f"| {name} | {te:.2f} ms | {tn:.2f} ms | **{tn / te:.2f}×** |")
        lines += ["", "| compile (empty cache) | seconds |", "|---|---|"]
        for name, t in compile_times():
            print(f"compile {name:24s} {t:.2f} s", flush=True)
            lines.append(f"| examples/{name}.anvil | {t:.2f} |")
        with open(os.path.join(HERE, "results.md"), "w") as f:
            f.write("# Benchmark results\n\n" + "\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def blas_name():
    try:
        cfg = np.show_config(mode="dicts")
        return cfg["Build Dependencies"]["blas"]["name"]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
