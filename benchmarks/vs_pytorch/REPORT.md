# Anvil against PyTorch on an Apple M3 Max

I wrote nine small machine-learning projects twice, once in Anvil and once in PyTorch, and timed
both on the same machine. Each pair uses the same models, data, hyperparameters and number of
steps. The results agree within the noise between random seeds.

**Summary**

- **On the CPU, Anvil trains 6.5× faster than PyTorch at default settings** (geometric mean over the
  nine projects; 2.3× to 22× per project). Comparing single threads, so that one core does the
  work in both, Anvil is 2.3× faster. Giving PyTorch its best thread count for every project, Anvil is
  still 3.6× faster, and with both at their best, 2.9× (see [Threads](#threads)).
- **`torch.compile` didn't help** on any of these projects: in steady state Anvil is 7.9× faster than
  compiled PyTorch.
- **On the GPU, Anvil's Metal backend is 2.2× faster than PyTorch's MPS backend** (geometric mean).
  PyTorch wins on convolutions, though: on the larger CNN, MPS is 5.4× faster than Anvil on Metal and
  3.5× faster than Anvil on the CPU. That is the one project where PyTorch is the fastest option overall.
- **Small models gain the most.** A training step of softmax regression takes 26 µs in Anvil and
  158–374 µs in PyTorch. The matrix products in that step take 22 µs; the rest of PyTorch's time
  is per-operation overhead. For the 1.9M-parameter
  MLP, most of the time is spent in matrix products running on the same Apple AMX hardware, and
  Anvil's lead drops to 1.3× single-threaded and 2.2× when PyTorch uses its best thread count.
- **Anvil uses less memory and starts faster.** A process uses 5–360 MB against PyTorch's
  320–970 MB, except on the EMNIST CNN, where the two are about equal. An Anvil program is running
  within milliseconds; PyTorch needs about 1.5 s just for `import torch`. Anvil's compile step
  costs 0.2–0.6 s, and 3.3 s for the char-RNN, whose 32 unrolled time steps make 1,376 kernels.
- **Writing the comparison found four Anvil bugs**, all now fixed: a wrong gradient, a Metal build
  failure, idle CPU use, and a clock that counted sleep. See [What the comparison found in
  Anvil](#what-the-comparison-found-in-anvil).

Contents: [Setup](#setup) · [The projects](#the-projects) · [How it was measured](#how-it-was-measured) ·
[Training speed](#training-speed) · [Threads](#threads) · [The GPU](#the-gpu) ·
[Start-up, compile time, memory](#start-up-compile-time-memory) · [The results agree](#the-results-agree) ·
[Single operations from Python](#single-operations-from-python) · [Why Anvil is faster](#why-anvil-is-faster) ·
[Where Anvil is behind](#where-anvil-is-behind) · [What the comparison found in Anvil](#what-the-comparison-found-in-anvil) ·
[Reproducing](#reproducing)

## Setup

| | |
|---|---|
| machine | MacBook Pro, Apple M3 Max (12 performance + 4 efficiency cores, 40-core GPU), 48 GB |
| system | macOS 15.0.1, Apple clang 16; on battery, normal power mode (not Low Power) |
| PyTorch | 2.14.1 from PyPI, Python 3.14.7. BLAS: Accelerate; threads: OpenMP, 12 by default; no oneDNN (MKLDNN) in the macOS build |
| Anvil | this repository: native ARM64 code (NEON, with matrix products through Accelerate's `cblas_sgemm` on the AMX unit), 8 threads by default; the Metal backend for the GPU |
| background | other programs (a VPN, a Logitech updater, Chrome, a Python experiment) used about 2 of the 16 cores throughout; load average 8–14 |

## The projects

Nine small training programs, each written twice: [`anvil/*.anvil`](anvil) and [`torch/*.py`](torch).
Both versions use the same model, the same initialization (He-normal weights, zero biases), the
same optimizer and hyperparameters, the same batch size and number of steps, and the same data.
Every dataset is held in memory as one tensor, and each epoch reshuffles it.

| project | model | parameters | data | steps × batch | what it stresses |
|---|---|---|---|---|---|
| `logreg` | softmax regression, SGD | 7,850 | MNIST | 3,000 × 100 | the fixed cost of a training step |
| `mlp` | 784-128-10 perceptron, SGD | 101,770 | MNIST | 4,685 × 64 | small matrix products |
| `wide_mlp` | 784-1024-1024-10 perceptron, Adam | 1,863,690 | MNIST | 936 × 128 | large matrix products |
| `cnn` | LeNet: conv 8, conv 16, linear; Adam | 5,994 | MNIST | 2,400 × 50 | small convolutions |
| `emnist_cnn` | the `bin/draw --text` letter network: conv 32, conv 64, linear 256, dropout; Adam | 330,430 | 120,000 EMNIST characters | 937 × 128 | large convolutions |
| `vae` | variational autoencoder, 784-256-16-256-784; Adam | 415,024 | MNIST | 1,800 × 100 | mixed layers, sampling |
| `charrnn` | GRU language model, 32-step backpropagation through time; Adam | 103,424 | this README (51 KB) | 1,000 × 32×32 | many small operations in sequence |
| `gpt` | transformer: 2 blocks, 4 heads, width 64, context 64; Adam | 136,064 | this README | 1,000 × 16×64 | attention, layer norm, GELU |
| `spirals` | 2-64-64-2 tanh network on 2,000 points; Adam | 4,482 | synthetic | 4,000 × 100 | almost nothing but the cost of a step |

The PyTorch versions are written the way a PyTorch user would write them for speed:
- the data is a tensor on the device, and batches are gathered with `randperm`, without a `DataLoader`
- nothing is read back to Python inside the loop: losses accumulate as tensors
- gradients are reset with `zero_grad(set_to_none=True)`
- `charrnn` uses `nn.GRU` and `gpt` uses `F.scaled_dot_product_attention`, PyTorch's fused implementations

Two differences in the math remain:
- `nn.GRU` applies the reset gate after the hidden state's linear map, while Anvil's GRU applies it
  before. The cost is the same.
- PyTorch's `binary_cross_entropy` clamps the log at −100, while Anvil adds 1e-7 inside it.

The two versions start from different random numbers, so their results differ slightly. They are
within the noise of each other (see Results).

## How it was measured

Each project ran in seven configurations:

| configuration | what it is |
|---|---|
| `anvil` | Anvil compiled to ARM64 (NEON, matrix products through Accelerate on the AMX unit), default threads (8, the performance cores up to 8) |
| `anvil-1t` | the same, `ANVIL_THREADS=1` |
| `anvil-metal` | Anvil compiled for the GPU (Metal kernels, matrix products through Metal Performance Shaders) |
| `torch` | PyTorch eager on the CPU, default threads (12: OpenMP over the performance cores; BLAS is Accelerate) |
| `torch-1t` | the same, `torch.set_num_threads(1)` |
| `torch-compile` | PyTorch with `torch.compile(model)` (Inductor, CPU) |
| `torch-mps` | PyTorch eager on the GPU (MPS) |

- **Timing.** Every program prints the training time of each epoch, or of each chunk of 100 steps
  for the step-based projects. Only training is timed: loading data and evaluating the test set are
  left out. GPU timings include waiting for the GPU to finish (`torch.mps.synchronize()`; in Anvil,
  `clock()` waits for the GPU).
- **Steady state vs first epoch.** "Steady state" is the median epoch after the first. The first
  epoch is reported separately, because it carries the warm-up, and for `torch.compile` the
  compilation itself. `emnist_cnn` has only one epoch, so for it steady state *is* the first epoch.
- **Process and memory.** Around each run the runner records the process's wall time and peak
  memory (`/usr/bin/time -l`). Anvil's compile time (`anvil build`) is measured separately.
- **Repeats.** Each configuration ran 3 times, round-robin across configurations, and the tables
  show the median.
- **Background load.** The machine was in normal use, with other programs taking about 2 of its 16
  cores (load average 8–15). The load was recorded with every run.

## Training speed

Seconds per epoch in steady state: the median epoch after the first, then the median over 3 runs.
For `charrnn` and `gpt` an "epoch" is 100 steps, and for `spirals` it is 20 epochs of 20 steps.
Lower is better; the fastest on each row is in bold.

| project | Anvil | Anvil, 1 thread | Anvil Metal | PyTorch | PyTorch, 1 thread | torch.compile | PyTorch MPS |
|---|---|---|---|---|---|---|---|
| logreg | **0.016** | 0.031 | 0.030 | 0.224 | 0.095 | 0.281 | 0.218 |
| mlp | **0.067** | 0.085 | 0.073 | 0.501 | 0.180 | 0.881 | 0.483 |
| wide_mlp | 1.106 | 2.196 | **0.253** | 2.542 | 2.785 | 2.661 | 0.577 |
| cnn | **0.616** | 2.143 | 3.542 | 13.510 | 4.802 | 14.465 | 1.602 |
| emnist_cnn | 8.058 | 15.731 | 12.426 | 32.533 | 33.126 | 35.789 | **2.300** |
| vae | 0.306 | 0.520 | **0.175** | 1.771 | 1.152 | 2.051 | 1.039 |
| charrnn | **0.341** | 0.467 | 0.990 | 1.073 | 0.646 | 1.081 | 0.968 |
| gpt | 0.268 | 0.538 | **0.158** | 0.835 | 1.059 | 0.876 | 0.456 |
| spirals | **0.016** | **0.016** | 0.054 | 0.251 | 0.107 | 0.472 | 0.422 |

The same numbers as Anvil's speed-up (PyTorch's time divided by Anvil's; above 1 means Anvil is faster):

| project | CPU, defaults | CPU, 1 thread each | Anvil vs torch.compile | Metal vs MPS |
|---|---|---|---|---|
| logreg | 14.2× | 3.1× | 17.8× | 7.2× |
| mlp | 7.5× | 2.1× | 13.2× | 6.7× |
| wide_mlp | 2.3× | 1.3× | 2.4× | 2.3× |
| cnn | 21.9× | 2.2× | 23.5× | **0.45×** |
| emnist_cnn | 4.0× | 2.1× | 4.4× | **0.19×** |
| vae | 5.8× | 2.2× | 6.7× | 5.9× |
| charrnn | 3.1× | 1.4× | 3.2× | **0.98×** |
| gpt | 3.1× | 2.0× | 3.3× | 2.9× |
| spirals | 16.1× | 6.9× | 30.3× | 7.8× |
| **geometric mean** | **6.5×** | **2.3×** | **7.9×** | **2.2×** |

The time per training step makes the fixed costs visible:

| µs per step | Anvil | Anvil, 1 thread | PyTorch | PyTorch, 1 thread | torch.compile | PyTorch MPS |
|---|---|---|---|---|---|---|
| logreg (784×10, batch 100) | 26 | 51 | 374 | 158 | 469 | 364 |
| spirals (2-64-64-2, batch 100) | 39 | 39 | 627 | 267 | 1,181 | 1,055 |
| mlp (784-128-10, batch 64) | 71 | 91 | 535 | 193 | 941 | 516 |

Runs were steady. Over each set of 3 runs, the slowest was within 4% of the fastest (median) for
every configuration. The worst cases were 6% for Anvil and 14% for PyTorch.

## Threads

Six of the projects at 1, 2, 4, 8 and 12 threads, 2 runs each. The numbers are seconds per epoch
in steady state, as in [Training speed](#training-speed); the best for each framework is in bold.

| project | Anvil 1 | Anvil 2 | Anvil 4 | Anvil 8 | Anvil 12 | PyTorch 1 | PyTorch 2 | PyTorch 4 | PyTorch 8 | PyTorch 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | 0.088 | 0.074 | **0.066** | 0.069 | 0.089 | **0.175** | 0.357 | 0.345 | 0.427 | 0.490 |
| wide_mlp | 2.257 | 1.379 | **1.079** | 1.157 | 1.627 | 2.820 | **2.476** | 2.523 | 2.673 | 2.655 |
| cnn | 2.187 | 1.274 | 0.797 | **0.613** | 0.977 | **4.802** | 6.446 | 7.635 | 11.095 | 13.468 |
| emnist_cnn | 15.796 | 11.078 | 9.082 | **8.019** | 8.180 | 33.090 | 26.571 | **24.366** | 29.276 | 32.567 |
| charrnn | 0.466 | 0.403 | **0.335** | 0.374 | 0.423 | **0.614** | 0.748 | 0.786 | 0.918 | 1.022 |
| gpt | 0.535 | 0.358 | **0.272** | 0.291 | 0.302 | 1.045 | 0.781 | **0.653** | 0.767 | 0.813 |

- **Anvil speeds up to 4–8 threads**, by 1.3× (mlp) to 3.6× (cnn) over one thread. At 12 it slows
  down by up to 59% against 8: other programs occupied about 2 cores, and 12 threads then spill onto
  busy or efficiency cores. That is why Anvil's default is 8.
- **PyTorch speeds up only on the larger models**: 1.6× on the GPT, 1.4× on the EMNIST CNN (both
  at 4 threads), and 1.14× on the wide MLP at 2. On mlp, cnn and charrnn every extra thread makes
  it *slower*, and at its default of 12 threads the CNN takes 2.8× as long as on one. Each of
  these operations is too small to pay for waking and synchronizing 12 OpenMP threads, and the
  next operation can't start until the slowest thread finishes.
- **The fairest comparison.** PyTorch's default of 12 threads is its worst or second-worst setting
  in 5 of these 6 projects. With each framework at its own best thread count, Anvil is **2.9×**
  faster (geometric mean), from 1.8× on charrnn to 7.8× on the CNN. Anvil at its default against
  PyTorch at its best, over all nine projects, is **3.6×**. That uses PyTorch's single-thread
  times for logreg, vae and spirals, where 1 thread was its best setting.

## The GPU

Anvil's Metal backend generates its own kernels, routes matrix products to Metal Performance
Shaders, and keeps loop counters and the optimizer's step count on the CPU, so a training step
never waits for the GPU. PyTorch's MPS backend dispatches each operation from Python.

- **Steps with modest arithmetic.** Anvil's Metal backend wins by 6–8× on logreg, mlp, vae and
  spirals. Here PyTorch MPS pays about 0.4–1 ms per step in dispatch, which is slower than Anvil's
  *CPU* (and slower than PyTorch's own CPU).
- **The wide MLP and the transformer** have real arithmetic in them. Anvil on Metal is the fastest
  configuration of all: 2.3× MPS on the wide MLP and 2.9× on the GPT.
- **Convolutions.** MPS calls Apple's tuned convolution kernels; Anvil builds a convolution from an
  im2col copy and a stack of small MPS matrix products. On the EMNIST CNN, MPS takes 2.30 s per
  epoch against Anvil's 12.4 s on Metal and 8.1 s on the CPU. On the small CNN, Anvil's CPU (0.62 s)
  still beats both GPUs.
- **The char-RNN** is 32 small steps in sequence with batch 32. Neither GPU helps: both take
  about 1 s per 100 steps, three times Anvil's CPU.

## Start-up, compile time, memory

| | Anvil | PyTorch |
|---|---|---|
| start a process (`import torch` / nothing) | ~0 | ~1.5 s |
| whole run of `spirals` (process wall time) | 0.16 s (plus 0.25 s to compile) | 4.0 s eager, 7.7 s with torch.compile |
| whole run of `logreg` | 0.18 s (plus 0.17 s to compile) | 2.7 s eager |
| compile | `anvil build`: 0.17–0.58 s, except `charrnn` 3.3 s; Metal 0.9–1.8 s, except `charrnn` 35 s | `torch.compile`: +0.8–0.9 s in the first epoch, from a warm on-disk cache; the first compile on this machine took 7.7 s |
| warm-up | none: the first epoch takes as long as the rest | +0.01–0.02 s eager, +0.1–0.2 s on MPS |

Peak memory of the process, in MB:

| project | Anvil | Anvil Metal | PyTorch | torch.compile | PyTorch MPS |
|---|---|---|---|---|---|
| logreg | 214 | 329 | 595 | 704 | 699 |
| mlp | 226 | 405 | 605 | 712 | 706 |
| wide_mlp | 363 | 535 | 866 | 970 | 720 |
| cnn | 307 | 2,373 | 712 | 765 | 720 |
| emnist_cnn | 1,318 | 5,054 | 1,198 | 1,155 | 946 |
| vae | 214 | 401 | 630 | 690 | 710 |
| charrnn | 17 | 2,003 | 375 | 438 | 451 |
| gpt | 58 | 366 | 392 | 486 | 447 |
| spirals | 5 | 106 | 318 | 432 | 440 |

- About 300 MB of every PyTorch process is the library itself.
- On the CPU, Anvil's memory is the data plus one statically planned arena. The MNIST projects hold
  60,000 images as floats (188 MB), and the text projects need 17 MB and 58 MB in total.
- Anvil's Metal backend is the exception. It gives every tensor its own buffer, with none of the
  CPU's arena sharing, which costs 2–5 GB on the CNNs and the char-RNN.

## The results agree

Test accuracy, or the final loss for `vae` and bits per byte for `charrnn` and `gpt` (lower is
better for those three). Seed 0, median of 3 runs:

| project | Anvil | PyTorch | PyTorch MPS |
|---|---|---|---|
| logreg | 92.13% | 92.09% | 92.04% |
| mlp | 96.84% | 96.95% | 96.83% |
| wide_mlp | 97.39% | 97.90% | 97.04% |
| cnn | 98.46% | 97.92% | 98.33% |
| emnist_cnn | 83.09% | 82.57% | 82.82% |
| vae | 116.69 | 116.88 | 116.79 |
| charrnn | 1.728 | 1.868 | 1.872 |
| gpt | 2.575 | 2.483 | 2.540 |
| spirals | 99.50% | 99.65% | 99.60% |

The two largest gaps go in opposite directions (wide_mlp and gpt favor PyTorch, charrnn Anvil), so
I re-ran those three projects with seeds 1–3:

| project | Anvil, seeds 1–3 | PyTorch, seeds 1–3 |
|---|---|---|
| gpt (bits per byte) | 2.423, 2.473, 2.465 | 2.504, 2.491, 2.508 |
| wide_mlp (accuracy) | 97.26%, 96.67%, 97.47% | 97.06%, 97.44%, 97.30% |
| charrnn (bits per byte) | 1.725, 1.757, 1.757 | 1.826, 1.874, 1.845 |

- **gpt and wide_mlp:** the gaps are noise. On other seeds Anvil's GPT is slightly ahead.
- **charrnn:** Anvil is consistently about 0.1 bits per byte better. The likely reason is the
  difference in the GRU itself: PyTorch's `nn.GRU` applies the reset gate after the hidden state's
  linear map, Anvil's GRU before.
- **Between backends:** Anvil's CPU, 1-thread and Metal runs give the same numbers to four digits.
  The random numbers are the same on every backend, and so is the order of summation.

## Single operations from Python

[`kernels.py`](kernels.py) times single operations called from Python. On the Anvil side,
`anvil.function` is called on NumPy arrays and reads and writes them in place. On the PyTorch side,
the same function runs eager, under `torch.compile`, and on MPS (waiting for the GPU each call).
Times are medians of 15 samples.

| operation | Anvil | PyTorch eager | torch.compile | PyTorch MPS | Anvil vs eager | Anvil vs compile |
|---|---|---|---|---|---|---|
| matmul 1024×1024×1024 | 1.011 ms | 1.000 ms | 1.007 ms | 0.222 ms | 0.99× | 1.00× |
| linear + GELU 256×1024→1024 | 0.537 ms | 0.551 ms | 0.690 ms | 0.091 ms | 1.03× | 1.29× |
| elementwise chain, 4M elements | 0.608 ms | 5.108 ms | 0.811 ms | 0.896 ms | **8.40×** | 1.33× |
| softmax 4096×1024 | 1.069 ms | 1.251 ms | 1.285 ms | 0.102 ms | 1.17× | 1.20× |
| layer norm 4096×1024 | 0.723 ms | 1.291 ms | 1.146 ms | 0.383 ms | 1.79× | 1.59× |
| causal attention 16×256×64 | 1.230 ms | 0.434 ms | 0.480 ms | 0.052 ms | **0.35×** | 0.39× |
| conv 5×5 1→32 + ReLU + pool, 256×28×28 | 1.660 ms | 8.334 ms | 6.947 ms | 0.431 ms | **5.02×** | 4.19× |
| MLP 784-128-10, batch 1 | 0.0148 ms | 0.0057 ms | 0.0162 ms | 0.0422 ms | **0.38×** | 1.09× |
| MLP 784-128-10, batch 1024 | 0.272 ms | 0.287 ms | 0.343 ms | 0.072 ms | 1.05× | 1.26× |
| **geometric mean** | | | | | **1.33×** | **1.25×** |

- **One operation at a time, the two CPUs are close.** Most of Anvil's advantage in training comes
  from what happens *between* operations, and a single call has no between.
- **Matrix products are a tie.** Both call Accelerate's `sgemm` on the AMX unit, at 2.1 TFLOP/s for
  the 1024³ product.
- **Anvil wins where it fuses or PyTorch's CPU path is weak.**
  - The elementwise chain runs at 83 GB/s in Anvil against 10 GB/s in PyTorch eager. torch.compile
    fuses it too and closes most of the gap.
  - Layer norm is 1.8× faster.
  - The convolution is 5× faster, again because PyTorch's macOS build has no oneDNN.
- **PyTorch wins two operations.**
  - **Causal attention**, 2.8× faster: `scaled_dot_product_attention` is a fused kernel, while Anvil
    writes out the 256×256 score matrix of every head.
  - **Batch-1 latency**: 5.7 µs against 14.8 µs, where Anvil's cost of a call from Python through
    `ctypes` dominates. Inside an Anvil program there's no such cost.
- **The GPU wins anything large.** MPS runs the 1024³ product at 9.7 TFLOP/s, 4.6× the CPU. But a
  batch-1 call takes 42 µs on MPS, longer than either CPU.
- **First call.** `anvil.function` compiles a function in about 0.2 s. `torch.compile` took 0.03–0.6 s
  here because its on-disk cache was warm from the training runs; a cold compile takes seconds.

## Why Anvil is faster

- **No interpreter in the loop.** An Anvil program compiles, training loop included, into one native
  executable with every shape known in advance. There is no Python between operations, no
  dispatcher choosing a kernel for each call, no autograd graph built during the forward pass, and
  no allocation: every tensor has a fixed place in memory planned at compile time. PyTorch pays all
  of that on every operation. Profiled on one thread, a logreg step is 18 PyTorch operations (53
  counting the ones they call) and takes 159 µs, of which the matrix products themselves take
  22 µs. The rest, about 8 µs per operation, is overhead. Anvil's whole step, arithmetic included,
  takes 51 µs on one thread. That overhead is why logreg, mlp and spirals show the largest ratios.
- **Fusion, including the optimizer.** Anvil differentiates the program before optimizing it, so
  elementwise operations, reductions and the optimizer's update are fused across the forward and
  backward passes. An SGD or Adam update happens inside the kernel that computes the weight
  gradient, rather than as a separate pass over every parameter. The elementwise chain in [Single
  operations from Python](#single-operations-from-python) shows the effect on its own: 8.4× PyTorch
  eager.
- **Convolutions without oneDNN.** PyTorch's macOS build has no oneDNN, so its CPU convolutions
  fall back to generic implementations. They hardly use extra threads: EMNIST takes 33 s per epoch
  with 1 thread and 32.5 s with 12. Anvil lowers a convolution to an im2col copy and a matrix product
  on the AMX unit, and its forward and backward kernels are threaded like any others.
- **Threads.** Anvil's worker threads take chunks of a kernel as they become free. OpenMP, which
  PyTorch uses, splits each operation evenly and then waits for the last thread. With other
  programs using some of the cores, that last thread is often late, and small operations cost more
  to coordinate than to compute. [Threads](#threads) shows the effect.

## Where Anvil is behind

- **GPU convolutions**, by 5× against MPS on the EMNIST CNN (above). The fix would be to call MPS's
  convolution, or MPSGraph, the way products already go to MPS.
- **GPU memory**: no arena sharing on Metal yet.
- **Compile time for unrolled programs.** `static for` over 32 time steps makes 1,376 kernels: 3.3 s
  to compile for the CPU, and 35 s for Metal, where clang compiles a very large Objective-C++ file.
  A run-time loop with autodiff through it would avoid the unrolling.
- **Fused attention.** PyTorch's `scaled_dot_product_attention` is 2.8× Anvil's attention written in
  index notation at 16×256×64, because Anvil materializes the score matrix. It barely matters at the
  GPT's 64-token context, but it would at longer ones.
- **Breadth.** These nine projects are the kind Anvil is built for: models defined in the program,
  data in memory, fixed shapes. PyTorch handles dynamic shapes, a huge library of operations,
  pretrained models, distributed training and CUDA. Anvil's CUDA backend has never run on a real
  NVIDIA GPU.

## What the comparison found in Anvil

Writing the same programs twice and comparing results found four Anvil bugs, all fixed and tested:

1. **A wrong gradient in reparameterized sampling.** In `z ~ normal(mu, sigma)`, the gradient with
   respect to `sigma` (which should be the sample's noise) came from a kernel that drew *new*
   random numbers. The VAE's variance therefore learned from the wrong noise. Its loss was 179.8
   after one epoch where PyTorch reached 159.7; the two now agree (116.7 and 116.9 after three
   epochs). The noise of a sample with tensor parameters now gets its own buffer. The test is
   `tests/test_gradcheck.py::test_reparameterized_samples`, for both `normal` and `uniform`.
2. **Metal kernels with more than 29 tensors failed to build.** In `charrnn`, the sum of the 32
   unrolled losses fuses into one kernel that reads 32 tensors, and Metal binds at most 31 buffers.
   Buffers past the limit now go through an argument buffer of GPU addresses. The test is
   `test_metal.py::test_kernels_with_more_tensors_than_metal_binds`.
3. **Idle threads used CPU.** An idle Anvil program, such as the `bin/draw` server between requests,
   used 22% of a core, because its worker threads woke every 100 µs to look for work. They now
   sleep longer each time, up to 4 ms, and use 0.7%. A new parallel kernel never waits for them,
   because the thread that starts it works on it too.
4. **`clock()` counted time the Mac slept.** Part of a re-run happened while the Mac was asleep
   with its lid closed, and two Anvil runs reported epochs longer than the whole process had run.
   `clock()` now uses `CLOCK_UPTIME_RAW`, the clock Python's `perf_counter` uses, which stops
   during sleep. The affected measurements were re-run with the Mac kept awake.

## Reproducing

```bash
python3 -m venv benchmarks/.venv-torch && benchmarks/.venv-torch/bin/pip install torch numpy
python3 benchmarks/vs_pytorch/prepare.py                 # the EMNIST slice and the text corpus
python3 benchmarks/vs_pytorch/run.py                     # the 9 projects × 7 configurations × 3 runs (~25 min)
benchmarks/.venv-torch/bin/python benchmarks/vs_pytorch/kernels.py      # single operations (~3 min)
python3 benchmarks/vs_pytorch/summary.py                 # the statistics quoted here
```

Every run is in `results.json`, the tables in `results.md`, `results_threads.md` and `kernels.md`.
Keep the Mac awake (`caffeinate -i`) and its lid open while the benchmarks run.
