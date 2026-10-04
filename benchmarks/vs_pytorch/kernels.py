"""Single operations called from Python: anvil.function against PyTorch (eager, torch.compile, MPS).

    benchmarks/.venv-torch/bin/python benchmarks/vs_pytorch/kernels.py

Each operation is the same function four ways: an Anvil function compiled with anvil.function (called
on NumPy arrays, which it reads and writes in place: no copies), PyTorch eager on the CPU, the same
under torch.compile, and PyTorch on the GPU (MPS; each call waits for the GPU). Inputs are made once,
outside the timing. Times are the median of 15 samples, each the average of enough calls to take
~20 ms. The first call of each is timed too: for Anvil and torch.compile it includes compiling.
Results: kernels.json and kernels.md next to this file.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import anvil  # noqa: E402

rng = np.random.default_rng(0)


def randn(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def causal_mask(t):
    return torch.triu(torch.full((t, t), float("-inf")), 1)


KERNELS = [
    dict(name="matmul 1024×1024×1024", flops=2 * 1024 ** 3,
         anvil="fn f(a: [n, k], b: [k, m]) = a @ b",
         torch=lambda a, b: a @ b, args=lambda: [randn(1024, 1024), randn(1024, 1024)]),
    dict(name="linear + GELU 256×1024→1024", flops=2 * 256 * 1024 * 1024,
         anvil="fn f(x: [n, d], W: [d, e], b: [e]) = gelu(x @ W + b)",
         torch=lambda x, W, b: F.gelu(x @ W + b, approximate="tanh"),
         args=lambda: [randn(256, 1024), randn(1024, 1024, scale=0.03), randn(1024)]),
    dict(name="elementwise chain, 4M elements", bytes=3 * 4 * 4 * 2 ** 20,
         anvil="fn f(x: [n], y: [n]) = relu(x * 2 + y) - 0.5 * abs(x) + sqrt(abs(y) + 1)",
         torch=lambda x, y: torch.relu(x * 2 + y) - 0.5 * x.abs() + torch.sqrt(y.abs() + 1),
         args=lambda: [randn(4 * 2 ** 20), randn(4 * 2 ** 20)]),
    dict(name="softmax 4096×1024", bytes=2 * 4 * 4096 * 1024,
         anvil="fn f(x: [n, d]) = softmax(x)", torch=lambda x: torch.softmax(x, -1), args=lambda: [randn(4096, 1024)]),
    dict(name="layer norm 4096×1024", bytes=2 * 4 * 4096 * 1024,
         anvil="fn f(x: [n, d], w: [d]) = layer_norm(x) * w",
         torch=lambda x, w: F.layer_norm(x, (x.shape[-1],)) * w, args=lambda: [randn(4096, 1024), randn(1024)]),
    dict(name="causal attention 16×256×64", flops=4 * 16 * 256 * 256 * 64,
         anvil="""
fn f(q: [b, t, d], k: [b, t, d], v: [b, t, d]):
    s[n, i, j] = sum q[n, i, c] * k[n, j, c] / sqrt(d) + (0.0 if j <= i else -1e9)
    return softmax(s) @ v
""", torch=lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
         args=lambda: [randn(16, 256, 64) for _ in range(3)]),
    dict(name="conv 5×5 1→32 + ReLU + pool, 256×28×28", flops=2 * 256 * 32 * 24 * 24 * 25,
         anvil="""
fn f(x: [n, 1, 28, 28], W: [32, 1, 5, 5], b: [32]):
    patches[n, i, j, c, u, v] = x[n, c, i + u, j + v] where u < 5, v < 5
    out[n, o, i, j] = sum patches[n, i, j, c, u, v] * W[o, c, u, v] + b[o]
    return max_pool2d(relu(out))
""", torch=lambda x, W, b: F.max_pool2d(F.relu(F.conv2d(x, W, b)), 2),
         args=lambda: [randn(256, 1, 28, 28), randn(32, 1, 5, 5, scale=0.2), randn(32)]),
    dict(name="MLP 784-128-10, batch 1 (latency)", flops=2 * (784 * 128 + 128 * 10),
         anvil="fn f(x: [n, 784], W1: [784, 128], b1: [128], W2: [128, 10], b2: [10]) = relu(x @ W1 + b1) @ W2 + b2",
         torch=lambda x, W1, b1, W2, b2: torch.relu(x @ W1 + b1) @ W2 + b2,
         args=lambda: [randn(1, 784), randn(784, 128, scale=0.05), randn(128), randn(128, 10, scale=0.1), randn(10)]),
    dict(name="MLP 784-128-10, batch 1024", flops=2 * 1024 * (784 * 128 + 128 * 10),
         anvil="fn f(x: [n, 784], W1: [784, 128], b1: [128], W2: [128, 10], b2: [10]) = relu(x @ W1 + b1) @ W2 + b2",
         torch=lambda x, W1, b1, W2, b2: torch.relu(x @ W1 + b1) @ W2 + b2,
         args=lambda: [randn(1024, 784), randn(784, 128, scale=0.05), randn(128), randn(128, 10, scale=0.1), randn(10)]),
]


def timed(fn, sync=lambda: None, samples=15, target=0.02):
    """(first call, median seconds per call)."""
    t = time.perf_counter()
    fn()
    sync()
    first = time.perf_counter() - t
    for _ in range(3):
        fn()
    sync()
    t = time.perf_counter()
    fn()
    sync()
    one = max(time.perf_counter() - t, 1e-7)
    reps = max(1, min(10_000, int(target / one)))
    times = []
    for _ in range(samples):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        times.append((time.perf_counter() - t) / reps)
    return first, statistics.median(times)


def main():
    torch.manual_seed(0)
    mps = torch.backends.mps.is_available()
    rows = []
    for k in KERNELS:
        args = k["args"]()
        cpu = [torch.from_numpy(a) for a in args]
        f = anvil.function(k["anvil"])
        want = k["torch"](*cpu).numpy()
        got = f(*args)
        assert np.allclose(got, want, rtol=2e-3, atol=2e-3), (k["name"], np.abs(got - want).max())
        row = {"name": k["name"]}
        g = anvil.function(k["anvil"])                                  # a fresh one: its first call compiles
        row["anvil_first"], row["anvil"] = timed(lambda: g(*args))
        row["torch_first"], row["torch"] = timed(lambda: k["torch"](*cpu))
        torch._dynamo.reset()
        c = torch.compile(k["torch"])
        with torch.no_grad():
            row["compile_first"], row["compile"] = timed(lambda: c(*cpu))
        if mps:
            gpu = [x.to("mps") for x in cpu]
            row["mps_first"], row["mps"] = timed(lambda: k["torch"](*gpu), torch.mps.synchronize)
        for key in ("flops", "bytes"):
            if key in k:
                row[key] = k[key]
        rows.append(row)
        print(f"{k['name']:42s} anvil {row['anvil'] * 1e3:9.4f} ms   torch {row['torch'] * 1e3:9.4f}   "
              f"compile {row['compile'] * 1e3:9.4f}   mps {row.get('mps', float('nan')) * 1e3:9.4f}", flush=True)

    with open(os.path.join(HERE, "kernels.json"), "w") as fh:
        json.dump({"torch": torch.__version__, "threads": torch.get_num_threads(), "rows": rows}, fh, indent=1)
    lines = ["| operation | Anvil (anvil.function) | PyTorch eager | torch.compile | PyTorch MPS | Anvil vs eager | Anvil vs compile |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        def ms(v):
            return f"{v * 1e3:.4f} ms" if v < 1e-3 else f"{v * 1e3:.3f} ms"
        lines.append(f"| {r['name']} | {ms(r['anvil'])} | {ms(r['torch'])} | {ms(r['compile'])} | "
                     f"{ms(r['mps']) if 'mps' in r else '—'} | {r['torch'] / r['anvil']:.2f}× | {r['compile'] / r['anvil']:.2f}× |")
    lines += ["", "First call (compiling included for Anvil and torch.compile):", "",
              "| operation | Anvil | PyTorch eager | torch.compile | PyTorch MPS |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['anvil_first']:.2f} s | {r['torch_first'] * 1e3:.1f} ms | "
                     f"{r['compile_first']:.2f} s | {r.get('mps_first', float('nan')) * 1e3:.1f} ms |")
    with open(os.path.join(HERE, "kernels.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
