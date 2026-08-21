# Making Anvil better than what people use today

PyTorch is the default because it is flexible, has everything, and runs on GPUs. It is weak at
things Anvil is built around: it finds shape errors only when the code runs, it hides what a
training step costs, it is not deterministic by default, it is hard to deploy without Python, and
small models spend their time in per-operation overhead. These are ideas for leaning into those
strengths and closing the gaps, grouped by what they improve. ✅ = done, ▶ = in this round.

## 1. Find bugs before (or the moment) they happen

| | idea | why it beats the usual tools |
|---|---|---|
| ✅ | shapes are types, checked at compile time | PyTorch finds a shape error when that line runs, maybe an hour into training |
| ✅ | shapes in the editor: errors as you type, shape on hover and after definitions | no tensor library can show shapes before running |
| ✅ | deterministic: bit-identical results for any thread count and every run | PyTorch needs flags and still isn't, on GPU |
| ✅ | `anvil run --check`: the first NaN or infinity is reported with the tensor, the element, the line and the math that made it | PyTorch's anomaly mode is slow and points at autograd internals |
| ✅ | warnings for training bugs the compiler can see: a parameter the loss does not depend on (it never learns), a parameter `minimize` never updates | silent in PyTorch |
| | `anvil gradcheck file.anvil`: check the program's own gradients against finite differences | |
| | warn about a loss that cannot go down (no parameters), labels outside the number of classes, … | |

## 2. Know what a program costs before running it

| | idea | |
|---|---|---|
| ✅ | memory known at compile time (`anvil build`), temporaries share one arena | |
| ✅ | `anvil cost file.anvil`: parameters, FLOPs and memory traffic per training step, peak memory, broken down by source line | PyTorch users find out with a profiler, after the fact |
| | estimated time per step from the cost model, before compiling | |

## 3. Usability

| | idea | |
|---|---|---|
| ✅ | call Anvil from Python on NumPy arrays (`anvil.function`) | |
| ✅ | `use "file.anvil"` to share models between programs | |
| ✅ | NumPy files: `npy("w.npy")` loads weights or data saved from NumPy/PyTorch (shapes from the file header), `save_npy` writes them | bring models in and out of the PyTorch world |
| ✅ | `anvil export`: a trained model as a C library + header, no runtime dependencies, callable from C, C++, Swift, Rust, Go | deploying PyTorch models means TorchScript/ONNX/ExecuTorch |
| ✅ | standard library: dropout, learning-rate schedules (warmup + cosine, exponential), gradient clipping (`clip_norm`), label smoothing; batch norm still to do | |
| ✅ | Jupyter: a `%%anvil` cell magic; Anvil functions defined in a cell are callable from Python cells | notebooks are where people work |
| | `anvil watch file.anvil`: re-run on every save | |
| | `anvil fmt`: one canonical layout | |
| ✅ | compile-time `if` in model bodies (`if DEPTH == 2: l2 = Linear(…)`) | |
| | metrics logging to CSV and live terminal plots (sparklines) | |
| | a language reference and a tutorial | |

## 4. Speed

| | idea | |
|---|---|---|
| ✅ | whole-program fusion, register-tiled NEON kernels, threads, static memory | 1.3× NumPy on MNIST, 11–21× on softmax/layer norm |
| ✅ | matrix products through Apple's AMX coprocessor (Accelerate `sgemm`), including stacks and sums of products; epilogues split off | matmul 1024³ 4.6× faster, now ahead of NumPy; wide MLP 2×, transformer 1.3×, CNN 1.4× |
| ✅ | a Metal backend: train on the Apple GPU (shared memory, waits only when the host reads, loop counters on the host, products on Metal Performance Shaders) | 3.7–5.2× the CPU on wide MLPs; every example matches the interpreter |
| | half precision (f16 NEON arithmetic: 2× throughput, half the memory) | |
| | `anvil tune`: time the candidate schedules of each kernel and keep the fastest | |
| | SME on M4 | |
| | compile faster: cache the elaborated prelude | |
| ✅ | compiled functions bake what they read from the program (weights loaded once, at compile time) | the drawing app's reader: 2.4 → 0.07 ms a call |

## 5. Language power

| | idea | |
|---|---|---|
| | differentiate through run-time loops (a tape of loop-carried values) | |
| | a dynamic batch dimension (the last, smaller batch) | |
| | guarded indices: `x[t - 1] if t > 0 else 0` without a bounds error | |
| | `vmap`, forward-mode `jvp`, Hessian-vector products | |
| | complex numbers and an FFT | |

## 6. Platforms

| | idea | |
|---|---|---|
| ✅ | AArch64 macOS (native), CUDA (one `.cu` file) | |
| | a portable C backend (the CUDA emulation already is one): any machine with a C compiler | |
| | WebAssembly: run models in the browser (the drawing app with no server) | |
| | Linux AArch64 and x86-64 AVX2 | |
