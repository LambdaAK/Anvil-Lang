# Benchmark results

Measured on arm64 Apple M3 Max, NumPy 2.5.3 (BLAS: accelerate), 2026-10-07. Best of three runs. Load average at the start: 17.1, at the end: 15.6 (other programs' work makes both sides slower and noisier).

| benchmark | computation | Anvil | NumPy | Anvil speed-up |
|---|---|---|---|---|
| matmul 1024×1024×1024 | C = A @ B, then mean(C) | 0.94 ms (2276 GFLOP/s) | 1.21 ms (1775 GFLOP/s) | **1.28×** |
| linear + gelu 256×1024→1024 | gelu(x @ W + b), then sum | 0.52 ms (1033 GFLOP/s) | 0.96 ms (557 GFLOP/s) | **1.85×** |
| elementwise, 4M floats | gelu(1.01 x + 0.1) · sigmoid(x), then sum | 5.61 ms (3.0 GB/s) | 17.41 ms (1.0 GB/s) | **3.10×** |
| softmax 4096×1024 | softmax over rows, then a weighted sum | 1.00 ms (16.7 GB/s) | 10.49 ms (1.6 GB/s) | **10.45×** |
| layer norm 4096×1024 | layer_norm over rows, then a weighted sum | 0.58 ms (28.9 GB/s) | 6.98 ms (2.4 GB/s) | **12.04×** |
| causal attention 16×256×64 | softmax(q kᵀ/√d + mask) v, then sum | 1.27 ms (212 GFLOP/s) | 4.39 ms (61 GFLOP/s) | **3.46×** |
| MNIST MLP, one epoch | 784-128-10, batch 64, SGD (forward, backward, update) | 0.066 s | 0.121 s | **1.84×** |

Called from Python: `anvil.function(source)(arrays)` against the same NumPy code (time per call, results checked against NumPy).

| function | anvil.function | NumPy | Anvil speed-up |
|---|---|---|---|
| softmax 4096×1024 | 0.76 ms | 8.32 ms | **10.95×** |
| layer norm 4096×1024 | 0.25 ms | 4.74 ms | **18.82×** |
| gelu(x @ W + b) 256×1024→1024 | 0.40 ms | 1.23 ms | **3.10×** |
| causal attention 16×256×64 | 1.30 ms | 4.84 ms | **3.73×** |

| compile (empty cache) | seconds |
|---|---|
| examples/mnist.anvil | 0.24 |
| examples/transformer.anvil | 0.76 |
| examples/checkers.anvil | 0.90 |
