# Anvil: The Epic Plan

## The vision

Today Anvil means "write the math, get the machine code." After this plan, Anvil is **the compiler that shows its work**. You write naive three-line attention. Anvil prints the algebraic derivation of FlashAttention, checks every step, and runs the result in O(T) memory on NEON, Metal and a real NVIDIA GPU. Swapped axes, NaN-prone math, units and data leaks become type errors that show up before step 0. Functions and gradients become values that compile away to straight-line SIMD. One source file trains on an M3 Max, a Raspberry Pi, a Jetson and an H100, and prints joules everywhere. In exact mode the weights come out bit-identical. The compiler stays small enough for a class to read in a semester, and every chapter ends in a demo you can run on the projector.

---

## The arcs, ranked

### 1. The Algebraic Compiler: "Three general laws in, four famous kernels out, each step checked." (judges: 7.5)

Every Anvil kernel is already a sum-product comprehension over affine indices (`ir.Reduction(vars, op, body)`), and monoid and semiring algebra works directly on that shape. The prelude would hold only general facts:
- sum and max are commutative monoids
- `exp(a+b) = exp(a)*exp(b)`
- multiplication distributes over sum

There is no softmax-specific rule. When the compiler finds a reduction whose body depends on another reduction's result, it searches for a rescaling correction and builds the online (m, l, o) accumulator itself. The same unmodified pass then produces FlashAttention, Welford layer_norm, one-pass logsumexp and streaming linear + cross-entropy, and it was never told about any of them. This fixes the README's worst weakness (attention materializes the T x T score matrix) by generalizing what the language already is.

**Flagship demo.** `anvil ir --explain examples/attention.anvil` prints a derivation ladder: softmax fused into the matmul, then the banana split, then the online normalizer, then tiling. Each rung is its own runnable .anvil program, linked to source spans. Each rung is run automatically against the one before it, and the output shows the max numerical difference, peak memory and `anvil cost` FLOPs. Next, causal attention at 16k context with 16 heads: 16 GiB of f32 score matrices disappear and memory becomes O(T). The table reports memory and time against both PyTorch SDPA and MLX's fused SDPA. Encore: write an HMM forward algorithm once, decode it with `with semiring viterbi:`, and get the posteriors from `grad` (forward-backward for free).

**What we'd build**
- **Semiring desugaring on today's IR (1-2 weeks).** Tropical and log semirings lower to existing max/sum/exp/log kernels whose gradients autodiff.py already has. This gives Viterbi, all-pairs shortest paths as a min-plus `sum` on Metal, and HMM posteriors via `grad`.
- **`monoid` and `semiring` declarations as type-class instances.** These follow the `optimizer`-block precedent, and `with semiring` is just sugar for picking an instance. QuickCheck-style law checks fail at compile time with a counterexample. Each law carries a tag saying whether it holds exactly or only over the reals. "Holds over the reals, fails over f32, here is the counterexample" is a lecture on its own.
- **One new IR construct, not a general e-graph.** A tile statement: a GEMM tile, then an online-monoid (m, l, acc) update, then a GEMM tile. It gets an interpreter implementation, a CPU path (blocked Accelerate sgemm plus a NEON rescale) and a Metal simdgroup template. FlashAttention, streaming cross-entropy and Welford all reuse it.
- **Rematerialization as a law in the same rule set.** "Save lse, recompute P" becomes a choice the cost model makes, which yields the FlashAttention-2 backward pass. That lets the 16k demo train, not just run inference.
- **Parallel associative scan for linear recurrences.** It is written as `static for` over log2(T) levels and trains a minGRU or Mamba over 1,024 bytes. It also addresses the "no autodiff through run-time loops" weakness.
- **The low-rank demo through a cheaper route.** `sum((U @ V.T)**2)` goes from 10 GB to a 32x32 computation via a sum-product normal form plus a contraction-order search. An egg-style e-graph comes later, as a pre-pass that proposes rewrites to the existing optimizer rather than replacing it.

**Effort.** Semiring and monoid core: 2-3 weeks. Tile statement plus FlashAttention forward on NEON and Metal: 8-12 weeks. Backward pass via remat after that. Deferred: Datalog and SQL joins, Herbie-style rewriting, correctly rounded libm, Penrose diagrams, TACO sparse formats, and a Mirage-style superoptimizer.

**Judges' caveat:** If the online-softmax monoid is declared rather than built by the compiler, HN will say you hard-coded the trick (Mirage, Neptune and FlashLight already claim "the compiler finds FlashAttention"). Beating SDPA on Metal is also not guaranteed, so lead with memory.

### 2. Shapes Are Theorems: "Your tensors are checked before a single step runs." (7.0)

Today Anvil checks shapes the way C++ checks templates: with concrete numbers, at the call site. examples/transformer.anvil has T = D = 64, so a swapped `P[d, t]` compiles and trains silently worse. A judge confirmed it: the file passes at D = 64 and fails with a two-span error at `--set D=32`. This arc catches bugs that PyTorch and JAX cannot catch statically: named axes, an interval checker for numerically dangerous math, and units of measure carried through `grad`.

**Flagship demo (about 3 weeks out).**
- The swapped `P[d, t]` becomes a two-span error naming axes T and D. Next to it, a run shows the bits-per-byte it costs today.
- Textbook `log(sigmoid(z))` is rejected before step 0 with: "log can receive 0: sigmoid(z) underflows for z < -88; use log_sigmoid".
- `anvil cost --symbolic D,T,L` reads Kaplan's 6*N*tokens law plus the 12*L*T^2*D attention term off the class's own 60-line GPT.
- Closer: hover over an Adam update and the inferred units show it is dimensionless. That is why Adam is scale-invariant and SGD is not.

**What we'd build**
- Named dims (`dim T = 64`) carried through index notation, including the D -> (HEADS, DH) reshape split, and hooked into the existing two-span check in `solve_ranges`.
- Interval analysis on the concrete kernel IR, with a built-in log-sum-exp lemma so the prelude's own `log_softmax` passes. Results come in three tiers: proved, definitely bad, unknown.
- Symbolic `anvil cost` by Newton interpolation over `--set` grids. Compiles take about 0.5 s, so this costs days.
- Units of measure through `grad`. This is the headline novelty: the judges could not find it in any ML language.
- `anvil check --generic`, built as a separate shape-level checker over the AST rather than by making the elaborator abstract:
  - data dims are symbolic, while hyperparameters (k, stride, pad) stay concrete so every obligation stays linear and the Omega test can solve it;
  - `...` functions are checked at ranks 1-4;
  - a fuzz harness re-elaborates every example at random sizes.
- A dead-gradient tracer that names the exact edge that killed a gradient, plus examples/types/ planted-bug files (extending examples/errors/) as a 10-week lab sequence.

**Effort.** First wins in 1-3 weeks. Generic checker in 2-3 months. Handed to other arcs: precision inference (NVIDIA arc) and uniqueness types (Functional arc). Parked: typed holes and type providers.

**Judges' caveat:** Most of the theory has owners (Futhark, Dex, Remora, F#, DEBAR), so credit them openly. The interval prover will mostly answer "unknown" downstream of trained weights, so precision "proofs" only cover activations.

### 3. Functional Anvil: "Transformations are values, and they compile away to straight-line NEON." (6.75)

The elaborator is a staging interpreter that inlines every call, and `FnVal` already captures its scope. Closures and higher-order functions therefore already work; only lambda syntax is missing. A judge deleted one guard (autodiff.py:147-149) in a scratch copy. Reverse-over-reverse autodiff then matched finite differences up to third order, and all 58 existing tests stayed green. Grad-of-grad, HVPs and PINNs are days to two weeks away. The real multi-month core is autodiff through run-time control flow. Today a gradient through a run-time `for` is silently wrong ([1,1,1,1] in the judge's test).

**Flagship demo.**
- `newton = fn(f) => fn(x) => x - f(x) / grad(f)(x)` on a run-time input. `anvil asm` shows about 50 straight-line ops and no calls.
- `vmap(train)(lrs, keys)` trains a 64-model MNIST grid as one compiled batched-GEMM program, with an ASCII accuracy heatmap filling in. It is timed against the same 12 lines of JAX on the M3 Max.
- A physics-informed net solves the heat equation with no data, using the loss `dt(u) - alpha * dx(dx(u))`. `show` draws it beside the analytic solution.

**What we'd build**
- Lambda syntax, `grad(f)` and `value_and_grad` as transformers, the guard deleted, and a second-order gradcheck over the whole prelude. Tests check that higher-order code and hand-inlined code emit identical kernel IR, which doubles as an autograder.
- vmap as one IR rewrite: add a domain variable, and add `b*stride` to the offsets of the buffers that vary. That takes 2-4 weeks for kernels and For blocks.
- Forward mode as a standalone jvp pass per kernel built on `simplify.deriv`. We would not rebuild the working reverse mode, because blas.py's GEMM matcher is tuned to its output.
- Autodiff through structured control flow as the stated core:
  - [T, ...] tapes in the static arena, indexed by the loop counter;
  - then a compile-time Revolve schedule;
  - then `custom_vjp`, `fixed_point` and `odeint`.

  This also fixes the char-RNN compile blowup (2.2 s at SEQ=32, 15.6 s at SEQ=128).
- Records and pytrees, so optimizers are written once. `anvil ir --stage=parse,defun,vmap,linearize,fuse,schedule` lets students see the program after each transformation. An "Anvil Core" subset of about 1,500 lines serves as homework.
- `specialize(net.forward)`: a trained net partially evaluated into one call-free NEON function, benchmarked at batch 1 against ONNX Runtime.

**Effort.** Weeks for lambdas, higher-order AD and vmap. One to two months for control-flow AD. Optional later: effects and handlers, a PPL (HMC/SVI), stream fusion.

**Judges' caveat:** On novelty this is JAX plus Dex rebuilt for Apple silicon, so lead with measured speed and the story that it goes all the way to machine code. Also, 64 distinct models means batched GEMMs, not one giant GEMM.

### 4. Same Bits, Proved Right: "Train on an H100, resume on a MacBook, same SHA-256, and a proof says why." (6.75)

Anvil is already bit-identical across thread counts (tests/test_threads.py). It has one counter-based RNG shared by every backend, its own polynomial transcendentals, and a deterministic CUDA split reduction. This arc writes docs/NUMERICS.md as the formal IEEE float semantics of the kernel IR: the canonical reduction tree, explicit FMA placement, and the shared vecmath polynomials. It then proves that each emitted NEON, MSL and CUDA kernel computes exactly that. The two halves become one result: the hashes match because a proof says they must. One correction to today's claim: determinism currently holds per machine only, because Accelerate's AMX accumulation order depends on the chip. Exact mode fixes that.

**Flagship demo.**
- The M3 CPU, Metal and a rented H100 each train MNIST for 1,000 steps and print the same 64-hex-digit weight hash.
- A checkpoint saved on the H100 resumes on the Mac and matches the uninterrupted loss curve to the last bit.
- Flip `--fast` (Accelerate, MPS, TF32) and the hashes diverge. `anvil diff-bits` names the first kernel where they split.
- A planted off-by-one in loop collapsing passes the whole test suite. The certifier rejects it and names the kernel and the failing index.
- An honest exact-vs-fast throughput table sits next to it.

**What we'd build**
- `anvil run --hash`, `anvil gradcheck`, and hash CI across thread counts, Metal and CUDA. The first milestone is renting an H100 for a day and getting the CUDA tests green.
- A canonical reduction defined as k-blocked sequential FMA chains (the existing vec_out GEMM almost does this already). Plus an IR-level FMA placement pass, `--fmad=false` for nvcc, exact-FMA emulation in the interpreter, and one FMA-explicit port of vecmath.py.
- A deterministic sort-plus-segmented scatter-add, and the `diff-bits` bisector.
- Per-pass translation validation that reports minimal counterexamples, adjoint dot-product tests for every backward kernel, an Ansmith type-directed fuzzer with a delta-debugging shrinker, and omega bounds certificates.
- A minimal Lean 4 semantics of the kernel IR over floats first, then per-program certificates. The autodiff correctness proof over the reals stays the moonshot.
- `anvil lab`: a laptop-only zoo of about 20 planted compiler bugs, each paired with the tool that catches it.

**Effort.** Weeks for hashes, gradcheck and validation. Four to eight weeks for CPU/Metal hash equality on the MLP. About 9-15 months for the whole arc. Moved out: WebGPU (it needs its own backend first), the Rewind debugger, flight recorders, TracIn.

**Judges' caveat:** Exact mode bypasses Accelerate and MPS, which carry today's headline speedups. Fast should stay the default, with exact as a flag.

### 5. Anvil Goes to NVIDIA: "The compiler predicts what the GPU will do, and the profiler agrees." (6.5)

The CUDA backend (cuda.py, 663 lines) passes every test in emulation and has never run on an NVIDIA GPU. Week one fixes that on a free Colab T4. After that, one Tile IR becomes the single GPU lowering for both targets: `simdgroup_matrix` on Metal and `mma.sync` on NVIDIA. Its first proof point closes Anvil's worst Metal gap, convolutions that run 5x slower than MPS. Anvil's edge on GPUs is that it sees the whole program statically. It can predict kernels, bytes and time before running, fuse the optimizer into GEMM epilogues, and launch the entire run once.

**Flagship demo.**
- From the MacBook, `anvil run --cuda --remote` trains the char-GPT.
- Before launch, `anvil cost --cuda` prints the kernel count, the HBM bytes moved, and a roofline time bound for the whole run.
- The entire run, early stopping included, is one CUDA Graph launch while the CPU sits idle.
- Nsight's measured bytes and time land within a few percent of the prediction.
- The baseline is torch.compile max-autotune with CUDA graphs and fused AdamW.
- Closing slide: `anvil ptx` shows `out[..., j] = e[..., j] / sum e[..., k]` next to the five `shfl.sync` instructions its sum became.

**What we'd build**
- First light: a Colab/remote runner, every example diffed against the interpreter on a T4 and an H100, nightly real-GPU CI, and a REPORT_CUDA.md.
- A single cudaMalloc arena from arena.py (an afternoon of work), plus cuBLASLt epilogues and host scalars ported from the Metal backend's pattern.
- One CUDA Graph with conditional WHILE/IF nodes. Its prerequisites are explicit tasks:
  - move the Adam step and bias-correction kernel ahead of the gradient kernels, so the full update fuses into dW (today only l3 fuses fully);
  - a device-side Feistel shuffle shared with the interpreter;
  - `%globaltimer` in place of clock();
  - a print ring buffer;
  - `break` rewritten as a flag variable;
  - splitting the graph at host-only calls.
- A shared Tile IR with CuTe-style layout algebra, presented as a type system in which composition and divisibility are checked. Milestone: the EMNIST CNN's implicit-GEMM convolution beats MPS on the Mac and runs correctly on a T4.
- A compile-time Nsight: coalescing, sectors per request and bank conflicts computed from affine offsets and shown as VS Code warnings, plus a SIMT visualizer.
- A lab ladder that runs on a free T4. Each step's speedup is predicted before it runs:
  - naive, one thread per output (today's backend);
  - coalesced loads;
  - shared-memory tiling;
  - `shfl.sync` warp reduction;
  - `mma.sync` tensor cores on sm_75.

**Effort.** First light in 1-2 weeks. Graphs and arena in about a month. Tile IR in 2-3 months. Optional second arc: wgmma/TMA/fp8 on H100, a persistent megakernel, Instant-NGP. CUTLASS stays a benchmark only. No LLVM backend. Instead of a `tile fn` escape hatch, add schedule annotations on index notation.

**Judges' caveat:** Most of the GPU techniques are prior art (tinygrad, CuTe DSL, FlashAttention-3, Mirage MPK), and MNIST-scale wins disappear against torch.compile reduce-overhead. The headline is the prediction matching the measurement on a real workload.

### 6. The Classroom Is the Computer: "Scan a QR code; your phone compiles, type-checks and trains Anvil." (6.5)

The compiler needs only the standard library plus NumPy, which Pyodide ships, so the front end can run in a browser tab today. Every example compiles with a recursion depth under 150. Add a WebAssembly backend that writes .wasm bytes itself, Anvil's first backend that doesn't shell out to `cc`, and the language lives at a URL. The spine of the arc is course infrastructure: Compiler Theater, Build Your Own Anvil, and `anvil grade`.

**Flagship demo.**
- Slide one is a QR code. Phones compile examples/mnist.anvil and train on a 10k-sample subset in the tab.
- Change 128 to 120 and the shape error appears inline.
- A student handwrites `y[i] = sum W[i,j] * x[j]` on a tablet. bin/draw's EMNIST beam-search reader decodes it, with the parser and elaborator replacing the dictionary as the filter on the beam, so readings that don't type-check lose. The model then trains.
- Climax: the room races its own wasm GEMM tiling passes on a leaderboard.

**What we'd build**
- A Pyodide playground with ide.py diagnostics and shape inlay hints. blas.match is reused in the interpreter so recognized products run as np.matmul; the interpreter takes about 35 ms per MNIST step natively, which is too slow otherwise.
- Compiler Theater: one trace hook in the optimizer's single `changed` setter (optimize.py ~422-427) animates each source line into kernels and then arena slots.
- Wasm in two stages. First, a scalar emitter modeled on cuda.py's KernelGen and tested against the interpreter in Node 20. Second, SIMD tiles that share the NEON GEMM scheduler through one 4-lane abstraction. coi-serviceworker supplies the COOP/COEP headers GitHub Pages can't set.
- Build Your Own Anvil plus `anvil grade` (contraction equivalence, FLOP budgets, gradchecks) with Gradescope output. A types/FP track:
  - shape inference as Hindley-Milner-style unification over dimension variables;
  - forward-mode `grad` via dual numbers;
  - `vmap` on a small typed core.
- WGSL/WebGPU and the Living Book after that.

**Effort.** Weeks for the playground and Theater. Months for wasm and the course. Stretch: a DiLoCo phone swarm. It needs a relay server, and it can show a live hash that all phones agree on, not one printed on a slide in advance. Dropped: Arcade, Glass Box GPT, Arena.

**Judges' caveat:** In-browser training and build-your-own-framework courses have well-known predecessors (ConvNetJS, TF Playground, minitorch, needle). And 80 phones pulling Pyodide plus MNIST over lecture-hall Wi-Fi is a real live-demo risk.

### 7. Private by Construction: "The compiler proves what your model can leak." (6.25)

The IR is first-order and structured: named buffers, plus If/While/Break/Print/RTCall nodes that carry source spans. An information-flow pass with pc labels is therefore a few hundred lines. Leaks become compile errors with source paths, including the termination channel in `if loss < 0.05: break`. Step counts and batch shapes are static, so the DP accountant can run at compile time. There is also a real hole to close today: every loader calls `os.path.expanduser`, so a `use`d library can read ~/... and bake it into an `anvil export`.

**Flagship demo.**
- The compiler refuses to train and lists three leaks with source paths: a print inside the loop, the early-stopping `break`, and a `save` of a non-DP model.
- A friendly-looking `use "fancy_layers.anvil"` tries to read ~/.aws/credentials and fails at its own line: "this file was granted no read access".
- Finale, rehearsed offline on FEMNIST first: `private[per student] images` gives user-level DP at epsilon = 2 on a model pretrained on public EMNIST. LiRA's ROC collapses to the diagonal. A one-run audit with hundreds of canaries lands close to the proved bound.
- Participants opt in and are pseudonymous.

**What we'd build**
- Hardening (days): capability-scoped reads in one shared loader helper plus `use`, an `--allow-read` flag, checkpoints that carry names, shapes and hashes, and libFuzzer harnesses for the assembly loaders.
- Inferred information-flow labels with `declassify`, and train/val/test as labels. This pass is shared with arc 8.
- A per-example gradient transform at `Backprop.emit_grad`, checked against batch-1 gradcheck for the MLP, the im2col CNN and Embedding.
- DP-SGD with the privacy unit as a typed axis, an RDP/PRV accountant run at compile time, and epsilon shown as an inlay hint.
- LiRA plus one-run auditing (Steinke, Nasr, Jagielski 2023), with ensembles as a tensor axis.
- Constant-time inference as an opt-in mode. zkML via sum-check is the one crypto moonshot, because index notation is already the proof.

**Effort.** Days for hardening, weeks for labels, months for DP-SGD and the arena. Dropped: FHE, two-party computation, confidential H100, federated iPads.

**Judges' caveat:** The novelty underneath is thin (Fuzz/Duet sensitivity types, Opacus accountants, Viaduct). The demo's statistics must be per-student and validated in advance, or privacy experts will catch the gap.

### 8. The Compiler Reads Your Data: "The test set is a linear resource." (6.25)

Anvil already reads csv/idx/npy headers at compile time and rejects a bad CSV field with its line number. This arc turns data hygiene into compiler output. The test split gets an affine type: it can be consumed once, by `report()`. Any repeated look has to go through a budgeted declassifier (Thresholdout, Ladder) whose remaining budget lives in the type. `cross_validate` becomes a construct the compiler understands, so fitting preprocessing outside the fold is a type error. That check is sound, because Anvil sees the whole program.

**Flagship demo.**
- The ESL feature-selection trap (selecting features on all the data and then cross-validating, which reports about 3% error on pure noise in sklearn) is rejected at the selection line.
- UCI Adult: `test = csv("adult.test")` refuses to compile because its labels read `<=50K.` with a trailing period. Hovering shows `workclass: Workclass?`.
- `anvil data audit` shows the 25 most suspicious MNIST test digits as ASCII art in about three seconds. The class votes on each one, then sees which of the roughly 15 confirmed label errors made the top 25.

**What we'd build**
- A typed csv provider:
  - RecordVal values;
  - enum columns stored as i32 codes and printed through PickVal;
  - Option columns as a value plus a mask;
  - `onehot(enum) @ W` rewritten to a gather.

  It is built on the shared type-system arc, not a second type system.
- Split labels and `folds(n, k)` carried on values by the elaborator, the same way shape metadata is, plus an eight-program Leakage Zoo as homework.
- `anvil data audit`: cross-validated confident learning from one compilation, and near-duplicate search via a tiled AMX GEMM plus a new top-k reduction. It also runs on Metal and CUDA, giving a measured CPU vs GPU comparison.
- Content-hash receipts, with the claim stated as "same machine and same build".
- A narrow storage dtype (u8/f16) used only for loads, so only the Load codegen changes. EMNIST goes from 2.19 GB to 0.55 GB.

**Effort.** About 6-8 weeks for the core. Moved: semiring joins to arc 1. Dropped: SIMD parsers, the tiktoken race, in-graph JPEG, DuckDB, FineWeb, Datalog.

**Judges' caveat:** The individual pieces have owners (FSharp.Data, cleanlab, notebook leakage analyzers). Only the sound, whole-program leakage typing is new, so lead with it and report audit recall rather than precision.

### 9. The Speedrun: "GPT-2 on your laptop overnight, from a pure-Python compiler." (6.0)

The headline is a race nobody has claimed: GPT-2 124M trained from scratch to 3.28 FineWeb validation loss on one M3 Max. A judge already compiled a GPT-2 124M-sized Anvil program in 2.5 s and trained it on Metal at about 0.78 s per 2048-token step. That is about 2.3 TFLOP/s, roughly 15% of peak. The same judge wrote Muon as a 14-line `optimizer` block that trained MNIST to 97.8% with no compiler changes. Reaching "about a day" needs roughly 3x more throughput. Everything the race forces us to build (bf16 storage, fused attention, rematerialization, streaming) stays in the language afterwards.

**Flagship demo.**
- Before launch, `anvil cost` publishes a prediction of MFU and finish time.
- examples/speedrun/gpt2.anvil, readable in one sitting, trains overnight next to same-architecture MLX and PyTorch-MPS ports on the same machine.
- A sparkline dashboard shows samples turning from noise into English.
- Classroom version: a TinyStories or Shakespeare GPT reaches a fixed loss in 10 minutes on a student's MacBook, with a class leaderboard.

**What we'd build**
- Muon and the modded-nanogpt architecture tricks (ReLU^2, QK-norm, RoPE, logit softcap, value embeddings, U-net skips), each a few lines of index notation, with forward and backward matching llm.c numerically.
- A gradient-accumulation primitive. Today `minimize` steps immediately, and microbatches inside `static for` keep every activation alive.
- bf16 as a storage dtype threaded through the IR (`nbytes` is hard-coded to 4*numel).
- Fused attention on Metal, from arc 1, or a hand-matched kernel first.
- An mmap'd FineWeb .bin loader (the shards are pre-tokenized, so no BPE is needed) and Revolve-style chain remat via `under memory 8GB`.
- "Parallelism as Types": `f32[B @ dp, D @ tp]` on a virtual mesh over shared memory, collectives derived from the contracted indices, a red squiggle reading "partial sum over D reaches layer_norm unreduced", and predicted vs measured communication.

**Effort.** Months, and it depends on arcs 1 and 5. The 8xH100 record attempt, tiktoken parity and FP8 are dropped. H100 comes back as "same file, change the mesh line, predicted vs measured" once CUDA runs.

**Judges' caveat:** Sharding types are prior art from Mesh-TensorFlow, GSPMD and JAX, and "about a day" needs about 3x today's Metal throughput.

### 10. Differentiable Physics: "Draw a word and the smoke learns to spell it." (6.0)

Index notation, the static arena, fusion and `grad` make Anvil a scientific-computing language in disguise. Boundary conditions go into the type (`f32[N wrap, N clamp]`), and splitting each domain into interior and edge pieces keeps every access affine. The compiler runs von Neumann stability analysis and shows CFL bounds as inlay hints. `scan` with checkpointing planned at compile time makes long rollouts trainable. Today that is the bottleneck: a 64-step heat solver checks in 0.7 s, but 512 unrolled steps took 25 s to check and 30 s to build, and produced 436K lines of assembly and a 98 MB tape.

**Flagship demo.**
- Drag DT in the editor and the CFL hint turns red. Run it, and the simulation blows up exactly where the compiler said it would.
- Write "ANVIL" in the browser. Each optimization step replays 100 steps of Stable Fluids, and gradients flow back through checkpointed time. `anvil cost` prints the schedule: "11 snapshots x 3.1 MB, 2.6x recompute". A puff of smoke sharpens into the letters.
- In the lattice Boltzmann wind tunnel, a hand-drawn blob reshapes itself to cut drag.

**What we'd build**
- Boundary types with interior/edge splitting in `solve_ranges`, with heat, wave and Gray-Scott solvers each as one fused pass per step, shown as pages in bin/draw.
- Compile-time stability analysis for linear constant-coefficient schemes, with `simplify.deriv` extracting the stencil coefficients.
- `scan` with sqrt(N) checkpointing, then Revolve (shared with Functional Anvil), and one `custom_vjp` mechanism reused by CG, multigrid, LAPACK and MPC.
- The LBM wind tunnel first: it is pure affine stencils and needs only scan. Smoke comes second, once the gather backward pass is deterministic (fixed-point or sorted accumulation).
- Second-order AD as an explicit milestone, molecular dynamics with forces = -grad(energy), and a DiffTaichi-style soft walker trained through a compiled gym.

**Effort.** Weeks per physics demo, months for scan and boundary types. A separate later arc covers control, Lie groups, WCET `deadline` and a Crazyflie; it first needs a Cortex-M backend. Moved to an examples gallery: option Greeks, Gaussian processes, drums, the planet simulation.

**Judges' caveat:** The demos have prior art (DiffTaichi, PhiFlow, Warp, JAX-MD; "Learning to Fly in Seconds" is literally the title of the RLtools paper). What is new is the compiler checking stability and planning the checkpoints.

### 11. Anvil x C++: "Anvil's shapes are guarantees everywhere it runs." (5.75)

Compiled Anvil runs only on macOS on Apple silicon today, which shuts out most of a class. But cuda.py's output already compiles as plain C++ under `-DANVIL_EMULATE`. A judge trained MNIST through it and got the same losses as native, about 90x slower (37.3 s vs 0.4 s). Turning that path into a readable, fast C++20 backend brings Anvil to Linux, a Raspberry Pi and a Jetson, where CUDA finally meets real hardware. Shapes cross the language boundary as `std::mdspan` extents.

**Flagship demo.**
- One mnist.anvil produces labeled C++ on a Raspberry Pi 5, NEON on the Mac and CUDA on a Jetson, with matching loss curves.
- A typed export header makes clang reject a 28x27 image at compile time.
- Stretch encore, "one string, two compilers": clang shape-checks and differentiates an `ANVIL(R"(...)")` MLP subset at compile time. Its loss curve matches the Python compiler's bit for bit through the shared RNG. The static_assert message prints the Anvil excerpt with a ^~~~ under the bad 784; clang can't place its caret inside the string itself.

**What we'd build**
- `anvil export --cpp`: an mdspan-typed header with concepts (this already works on Apple clang 16), plus .pyi stubs for the Python side.
- A C++20 backend grown from the emulation path, closing the 90x gap with:
  - vec_out loop order;
  - OpenMP;
  - GEMMs routed to OpenBLAS;
  - lists of source spans, so every loop nest names its Anvil line (fusion keeps only one span today, optimize.py:886).
- A Linux AArch64 port. runtime.s has 118 `@PAGE` relocations and relies on Apple's stack-based variadic calling convention, so Linux uses the C runtime in anvil_host.h. Docker CI on arm64 and amd64.
- A reentrant C ABI with `--train` exports, Rust and Swift bindings, and a no-malloc certificate derived from the static arena.
- One embedding that shows something only Anvil can do: a PyTorch custom op whose fused backward pass Anvil derives.
- `anvil explain --target cpp,neon,...` for side-by-side labs.

**Effort.** Days for typed exports, weeks for the C++ backend and Linux, months for the ABI. Moved: x86/AVX-512/RVV/SVE to a separate machine-code arc, CUTLASS to the NVIDIA arc, and Godot/JUCE/ROS 2/DuckDB/Xcode to a later applications list.

**Judges' caveat:** Full constexpr Anvil would mean porting about 6,000 lines of frontend to constexpr C++, and Apple clang has no P2996 reflection. Keep it to a CTRE-style subset.

### 12. Every Transistor: "Same accuracy, N times fewer joules." (5.75)

Apple silicon is Anvil's home turf, but every run uses only one engine. Energy can be measured without sudo (`rusage_info_v6.ri_energy_nj`). Static shapes and a static arena let the compiler schedule a whole step across engines itself, using HEFT list scheduling, and report joules per source line. On unified memory, energy wins are far more likely than raw speed wins, and no framework reports energy per line.

**Flagship demo.**
- A reproducible joules-to-accuracy leaderboard (the MNIST CNN to 99%, a small GPT to a fixed loss) against PyTorch CPU, MPS and MLX, all from one unchanged source file.
- A live Gantt chart spreads one step over the P-cores, E-cores, AMX and GPU, on a transformer large enough for the overlap to pay. `anvil top` shows each engine's watts and the line it is running.

**What we'd build**
- `energy()`, `anvil run --energy`, `anvil cost --energy`, and Instruments signposts named by source line.
- All cores in use: the thread-pool cap lifted, and the E-cores on their own QoS pool prefetching the next batch into a double buffer.
- Metal on the planned arena (charrnn on Metal drops from 2,003 MB to about 20 MB), plus the cheap convolution fix: fold (n, i, j) into one larger product, or call MPSGraph's convolution.
- A HEFT scheduler that doesn't assume any particular hardware, working over an abstract list of engines. `anvil sched --machine m3max.toml` lets students edit engine specs and predict schedules on any laptop. CPU+CUDA is the second target.
- Neural Engine, in two steps:
  - First, a one-week spike: hand-write the MIL for the MNIST CNN training step with the weights held as MLState, and check where MLComputePlan places each operation.
  - Then the full step on the ANE: forward, backward, weight gradients and the update. That goes one step past maderix/ANE, which computes weight gradients on the CPU.

**Effort.** Days for energy, weeks for cores and arena, months for the scheduler. Full-step ANE training is the moonshot. Cut: Vision Pro, the iPad app, the thermal governor, the convolution algorithm zoo.

**Judges' caveat:** It is Apple-only, and all engines share one memory bus, so a five-lane speed demo might be only about 1.1x faster than `--metal`. Lead with joules.

### 13. Anvil Reads the World's Models: "Llama, read as 150 lines of math." (5.75)

Safetensors and GGUF headers become types at compile time; the npy loader already parses headers and converts f16. A wrong port then fails with the tensor's name. `anvil pull` turns a Hugging Face config.json into readable index notation. The current language can already do much of this:
- A judge wrote a 40-line GPT-2-sized KV-cache decoder: 117 tok/s on CPU and 128 tok/s on Metal in f32.
- A Qwen2.5-1.5B-shaped decoder compiles in 0.23 s and runs at 29.5 tok/s on Metal in f32, about 45% of memory bandwidth.

**Flagship demo.**
- qwen2.anvil on the projector, with grouped-query attention written literally as `k[b, j, h // G, e]`.
- `anvil cost` predicts tokens/s from weight bytes per token over bandwidth. The measured tokens/s appears next to it, alongside mlx_lm at the same precision.
- Then a race only Anvil can win: QLoRA fine-tuning of Qwen-0.5B on the laptop as one whole-program-compiled training step, timed against MLX-LM and PyTorch MPS. llama.cpp can't train.
- Lab version: `anvil diff` across gpt2 -> llama -> qwen2 shows RMSNorm, RoPE, SwiGLU and GQA as line diffs.

**What we'd build**
- Fix the Metal fast-math `tanh` NaN first (metal.py:68: gelu gives NaN on Metal and the correct answer on CPU), and add Metal-vs-interpreter checks to the zoo tests.
- GPT-2 124M generating text from OpenAI's safetensors, with weights stacked [L, ...] at load.
- Storage-only dtypes (f16/bf16/i8/u8) widened on load. Quasi-affine `//` and `%` indices so the GQA line compiles; it is rejected today.
- q8_0 and q4_0 as packed buffers, matched to hand-written NEON and MSL GEMV kernels the way blas.py matches GEMMs, plus bitwise ops in the language.
- `compile_step` for the 9 benchmark models, and `anvil.explain` printing ResNet as about 60 lines of index notation.

**Effort.** Weeks for GPT-2. Months for quantized chat and QLoRA. Deferred: the Dynamo backend, PJRT, the lazy device, Triton, MLIR, LiteRT, `anvil serve`, Whisper, SD-Turbo, Gaussian splatting.

**Judges' caveat:** Batch-1 decode is limited by memory bandwidth, so the best case against llama.cpp and MLX is a tie. Element types smaller than 4 bytes touch every backend, and block-quantized formats break the IR's affine-addressing rule.

### 14. One Source, Every Static Machine: "If it computes, Anvil targets it." (5.5)

Anvil's "everything static" design (static shapes, a static arena, a memory footprint known before running) is exactly what targets without an allocator need. The kernel IR is small and affine, so each new target is a walk over about 15 node types. One XOR or MNIST source becomes a spreadsheet, a circuit, a bare-metal image and a microcontroller build, and each one prints its static footprint.

**Flagship demo.** `anvil build --target={xlsx,pi,pico,fpga,cuda} mnist.anvil`.
- The spreadsheet trains while you hold F9; click a gradient cell to read its SUMPRODUCT formula.
- The Verilog runs in Verilator against a fixed-point oracle, then on a Tang Nano board.
- A bare-metal Pi image prints epochs over UART with no Linux.
- Each target reports its footprint in cells, bytes, LUTs or gates, and every loss trajectory matches the fixed-point interpreter.

**What we'd build**
- `anvil export --xlsx`, a weekend of work, validated in headless LibreOffice.
- A freestanding C++ backend from the CUDA emulation path (shared with arc 11). It unlocks the Pico 2 (a Cortex-M33 with no NEON) and bare metal.
- A fixed-point/int8 dtype with a range and overflow type check that runs before the circuit is built.
- A Verilog backend with Verilator tests, and a QEMU-tested bare-metal Pi image (boot stub, PL011 UART, PSCI, MMU bring-up).
- Classroom reproductions, credited openly: ∂Forth filling a bubble-sort hole (Bošnjak et al. 2017), and ten-image dataset distillation (Wang et al. 2018; needs second-order AD and scan).
- Stretch goals with no date:
  - anvilc, a data-parallel Anvil front end written in Anvil (the lexer as a parallel scan, the AST as a parent vector) that tunes its own kernels through a cost model it compiled itself;
  - TinyTapeout.

**Effort.** Weekends for xlsx and ∂Forth; weeks to months per hardware target. Dropped: the quantum backend.

**Judges' caveat:** Many items reproduce famous work. A full self-hosting fixed point needs strings, scan, sort and bitwise ops, which Anvil doesn't have.

---

## The master plan

Seven shared foundations do most of the work. Build each one once and several arcs use it:
- **Control-flow AD + scan** (Functional): needed by Physics, Speedrun remat, distillation, minGRU/Mamba.
- **The online-monoid tile statement** (Algebraic): needed by FlashAttention everywhere, streaming cross-entropy, Welford, the Speedrun.
- **One Tile IR for Metal and CUDA** (NVIDIA): closes the convolution gap and runs derived FlashAttention on GPUs.
- **Storage dtypes below 4 bytes**: needed for LLM weights, bf16 in the Speedrun, quantization, tensor cores, fixed-point hardware.
- **The information-flow pass**: shared by Privacy and Data.
- **The C++ backend**: brings Linux, Jetson, Pi, Pico and student laptops.
- **NUMERICS.md**: the basis for Same Bits, and the correctness oracle CUDA never had.

The seasons below cover the trimmed core of each arc, not the full idea lists.

1. **Season 0, "First Light" (weeks 1-3).** CUDA runs on a real T4. Silent wrong gradients become compile errors. Fix the Metal tanh bug. Add hash CI and gradcheck, capability-scoped loaders, named dims and symbolic cost. The project's credibility gaps close before anything new is built.
2. **Season 1, "The Algebra" (about months 1-3).**
   - Semiring desugaring and law-checked monoids.
   - Lambdas and higher-order AD.
   - The tile statement, with derived FlashAttention forward on NEON and Metal and its derivation ladder.
   - The interval checker and units through `grad`.
   This is the season that sets Anvil's identity.
3. **Season 2, "The Machine" (about months 3-6).**
   - The shared Tile IR: the EMNIST convolution beats MPS and runs on a T4.
   - Whole-run CUDA Graphs with `anvil cost --cuda` checked against Nsight.
   - Derived FlashAttention on real NVIDIA hardware.
   - Storage dtypes.
   - The C++ backend on Linux and Jetson.
   - scan with tapes.
4. **Season 3, "The Types" (about months 6-9).**
   - The generic forall checker.
   - The information-flow pass behind both the leakage typing and the privacy labels.
   - vmap, records and pytrees.
   - The remat-as-law FlashAttention backward.
   - Revolve and `custom_vjp`.
5. **Season 4, "The Show" (about months 9-12+).**
   - The laptop GPT-2 speedrun and the Qwen QLoRA showdown.
   - Smoke and the wind tunnel.
   - Anvil at a URL with Compiler Theater and Build Your Own Anvil.
   - Same Bits with float-semantics certificates.
   - The joules leaderboard.
6. **Moonshot shelf, picked off when inspired.** The Mirage-style superoptimizer, the self-hosting anvilc, the Lean autodiff proof, full-step Neural Engine training, TinyTapeout, zkML via sum-check, the persistent megakernel, 8xH100.

## Quick wins

1. **CUDA first light on a free Colab T4.** Compile every example with nvcc and diff it against the interpreter. "CUDA too" goes from an untested claim to a measured one.
2. **Stop silent wrong gradients.** Gradients through a run-time `for` are wrong today ([1,1,1,1] in a judge's test), and through a run-time `if` they come out as zero. Make both compile errors until scan lands.
3. **Fix the Metal `tanh` NaN.** Fast-math `tanh` at metal.py:68 overflows (gelu gives NaN on Metal), and Metal outputs aren't checked against the interpreter today. Fix it and add Metal-vs-interpreter checks to the example tests.
4. **Higher-order `grad`.** Add lambda syntax and `grad(f)`, delete the guard at autodiff.py:147-149, and add a second-order gradcheck over the prelude. That ships the Newton and PINN demos in days.
5. **`dim T = 64` named dims.** The transformer's silently accepted `P[d, t]` becomes a two-span error. About a week.
6. **`anvil cost --symbolic D,T,L`.** Newton interpolation over `--set` grids prints 12*L*D^2 parameters and 6*N*B*T FLOPs per step. The transformer needs an `L` constant first.
7. **Capability-scoped loaders plus checkpoint headers.** Stop `use`d files from reading ~/..., and store names and shapes so a transposed checkpoint no longer loads silently. Days.
8. **`anvil run --hash` and `anvil gradcheck`.** SHA-256 of the parameters, with CI across thread counts and the CUDA-emulate path. This is the seed of Same Bits.
9. **`energy()` and `anvil run --energy`.** No sudo needed. Add a joules/epoch column to benchmarks/vs_pytorch/REPORT.md.
10. **Muon in the prelude, plus full Adam fusion.** A judge has already written Muon as a working 14-line block. Separately, move the Adam step-counter and bias-correction kernel ahead of the gradient kernels so the whole update fuses into dW for every layer; today only l3 fuses fully.

Next in line: `anvil export --xlsx` (a weekend), the mdspan-typed `anvil export --cpp` header (days), and real GPT-2 124M text generation from OpenAI's safetensors (1-2 weeks).

## Where I would start

**Day zero:** spend one afternoon running the CUDA tests on a free Colab T4. Every GPU claim later in the plan depends on it, and it costs nothing.

**The first real move: the semiring and monoid core of the Algebraic Compiler, shipped as "the HMM lecture."** Add `with semiring viterbi:` and `monoid` declarations whose laws are checked at compile time. Write the forward algorithm once, decode it with Viterbi, and take posteriors from `grad`.

Why this first:
- **It is the top-ranked arc,** and this piece needs no new IR. It lowers to max/sum/exp/log kernels whose gradients already exist, so it is 2-3 weeks to a finished classroom demo.
- **It covers the functional-programming and type-system interests in one feature:** type-class instances with laws, and a compile-time counterexample whenever a float "monoid" isn't actually associative.
- **It forces the right design decision early:** how tuple accumulators and law-driven rewrites work. That leads straight to the one new IR construct that FlashAttention, Welford and streaming cross-entropy all reuse on NEON, Metal and real CUDA.
- **It makes the NVIDIA arc build on the algebra,** instead of running as a separate effort.

Do this first, and "the compiler derived FlashAttention, and here is the checked proof it printed" becomes something you can actually show.

The full catalog of ideas behind this plan is in [BRAINSTORM.md](BRAINSTORM.md).
