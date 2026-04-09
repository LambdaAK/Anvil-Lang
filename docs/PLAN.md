# Anvil — design & build plan

> **Anvil** is a small programming language for machine learning. You write the math
> (tensors, index notation, a loss to minimize) and the compiler emits a single, readable,
> self-contained **AArch64 assembly** file with the forward pass, the backward pass, and the
> optimizer fused into vectorized NEON loops. No Python runtime, no framework, no allocator,
> no dependencies beyond libc.

The name is an acronym: **A**utodiff, **N**ative, **V**ectorized, **I**ndex **L**anguage. Index
notation (Einstein's summation convention) is the core idea of the language:
`y[i] = sum W[i, j] * x[j]` is valid Anvil, and so is `x @ W + b`.

---

## 1. Goals and principles

| Goal | What it means in practice |
|---|---|
| **Easy** | Python-flavoured syntax. Layers, losses and optimizers are all in the box. A complete training loop takes about 5 lines. |
| **Aesthetic** | Code should read like the math in a paper: index notation, `Σ`/`sum`, `~ normal(0, σ)`, `minimize loss with adam()`, `x \|> layer \|> relu`. |
| **Intuitive** | Few concepts with predictable semantics. Shapes are part of types and are checked *before* anything runs. Errors point at the exact source span and say what to do. |
| **Optimized for ML** | Built-in reverse-mode autodiff, static memory planning, kernel fusion, SIMD vectorization, and fast inline transcendental functions. |
| **Compiles to assembly** | The output is a single `.s` file you can read, diff, assemble with `cc`, and run. |

Design principles:

1. **Shapes are types.** Every tensor has a static shape. Shape errors are compile errors.
2. **Index notation is the core IR.** Every operation, from `+` to `@` to `softmax`, lowers to
   one form, the *tensor comprehension*: `out[I] = reduce_J body(I, J)`. Autodiff, fusion and code
   generation are each written once, against that form.
3. **Differentiation is a compiler pass, not a runtime tape.** `grad` and `minimize`
   are expanded at compile time into ordinary kernels that get fused and vectorized
   like everything else.
4. **Everything static.** Shapes are known at compile time, so every buffer has a fixed address and size.
   The binary prints its memory footprint up front, and nothing is allocated while it runs.
5. **The standard library is written in Anvil.** `relu`, `softmax`, `cross_entropy`, `Linear`,
   `adam` and the rest live in `prelude.anvil`. You can read them, copy them and change them.
6. **Safe by default.** Index bounds are proven at compile time where possible. Otherwise they are
   checked once per kernel (for runtime offsets) or per element (for gathers).

---

## 2. A taste of Anvil

```python
# mnist.anvil — a two-layer perceptron on MNIST

images = idx("data/train-images-idx3-ubyte.gz") / 255      # shape read from the file at compile time
labels: i32[60000] = idx("data/train-labels-idx1-ubyte.gz")
X = images.reshape(60000, 784)

model MLP:
    l1 = Linear(784, 128)
    l2 = Linear(128, 10)
    fn forward(x: [n, 784]) -> [n, 10] = x |> l1 |> relu |> l2

net = MLP()

for epoch in range(5):
    for x, y in batches(X, labels, size=64, shuffle=true):
        loss = cross_entropy(net(x), y)
        minimize loss with adam(lr=1e-3)
    print("epoch {epoch}: loss {loss:.4f}")
```

Index notation is how you define new operations:

```python
fn attention(q: [t, d], k: [t, d], v: [t, d]) -> [t, d]:
    s[i, j] = sum q[i, c] * k[j, c] / sqrt(d)      # c is summed: it is not on the left
    p = softmax(s)
    out[i, c] = sum p[i, j] * v[j, c]
    return out

fn conv2d(img: [b, ci, h, w], k: [co, ci, kh, kw]):
    out[n, o, y, x] = sum k[o, c, u, v] * img[n, c, y + u, x + v]   # output size inferred: h-kh+1
    return out
```

…and the standard optimizers are ordinary Anvil code:

```python
optimizer adam(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
    state m, v                                   # per-parameter, zero-initialised
    step(w, g, t):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        w -= lr * (m / (1 - beta1 ** t)) / (sqrt(v / (1 - beta2 ** t)) + eps)
```

---

## 3. Language design

### 3.1 Lexical structure

* UTF-8 source, `#` comments, Python-style significant indentation (INDENT/DEDENT); newlines
  inside brackets are ignored. A block may also be a single statement on the same line as the colon.
* Identifiers may use Unicode letters (`θ`, `η`, `λ` are fine).
* Numbers: `42`, `1_000`, `3.14`, `1e-3`. Strings: `"…"` with `{expr:fmt}` interpolation.
* Unicode aliases (optional, ASCII is canonical): `Σ ∑` → `sum`, `∇` → `grad`, `√x` → `sqrt(x)`,
  `→` → `->`, `≤ ≥ ≠`, `∞` → `inf`.

### 3.2 Types

| Type | Example | Notes |
|---|---|---|
| element types | `f32`, `i32` | `f32` is the default |
| tensor | `[64, 784]`, `i32[64]`, `f32[]` | shape = list of compile-time integers |
| compile-time values | `64`, `0.1`, `"path"`, `true` | constant-folded, may appear in shapes |
| models, functions, optimizers | `Linear(784, 128)` | compile-time objects, fully inlined |

* **Compile-time vs runtime.** A name that is assigned exactly once, at the top of its scope, to a
  constant expression *is* a compile-time constant (so `batch = 64` can be used in a shape). `const`
  makes that explicit and enforced. Everything else lives at runtime.
* **Promotion**: `i32 ∘ f32 → f32`; `/` is true division, `//` floor division (Python rules).
  Comparisons produce `f32` masks (0.0/1.0), so `mean(pred == y)` is accuracy.
* **Shape polymorphism**: in a signature, a dimension name that is not otherwise bound is a
  *shape variable* (`fn f(x: [n, 784])`). It is unified at each call site and can be used as a
  compile-time integer in the body (`sqrt(d)` above).

### 3.3 Declarations

```python
const B = 64                                   # compile-time constant
param W: [784, 128] ~ normal(0, sqrt(2/784))   # trainable, persistent, global
param b: [128]                                 # zero-initialised
fn relu(x) = max(x, 0)                         # expression function
fn mlp(x: [n, 784]) -> [n, 10]:                # block function, typed
    h = relu(x @ W + b)
    return h @ W2 + b2
model Linear(fan_in, fan_out): …               # parameterised bundle of params + methods
optimizer sgd(lr = 0.01, momentum = 0.0): …    # per-parameter update rule with state
```

* **Functions** are always inlined (monomorphised) at the call site. Recursion is rejected.
  An expression statement at the end of a function body is its return value.
* **Models** are instantiated at the top level (`net = MLP()`). Instances own their params
  (`net.l1.W`), and calling an instance calls its `forward`.
* **Optimizers** declare `state` tensors (one per parameter, zero-initialised) and a
  `step(w, g, t)` rule. `t` is the 1-based step count.

### 3.4 Tensor expressions

* Arithmetic `+ - * / // % **`, matmul `@`, comparisons, `and/or/not`, conditional
  `a if c else b` (elementwise select), pipeline `x |> f |> g(2)` (= `g(f(x), 2)`).
* **Broadcasting** follows NumPy's trailing-dimension rules.
* `@` handles vector/matrix/batched cases: `[..., k] @ [k, m] → [..., m]`.
* **Indexing** produces zero-copy *views*: `x[3]`, `x[a:b]`, `x[:, j]`, `x[::2]`, `x.T`,
  `x.reshape(…)`, and integer-tensor gathers `x[perm]`. A slice length must be static, but its
  start may be a runtime value: `images[s : s + 64]` type-checks because `(s + 64) - s = 64`.
* Uniform function-call syntax: `x.sum()` is `sum(x)`, and `logits.softmax()` is `softmax(logits)`.
* Tensor literals: `[[1, 0], [0, 1]]` (NumPy dtype rules). A list of same-shaped tensors stacks
  them along a new first axis, and `stack([a, b, c], axis=k)` along axis `k`.
* `nonzero(mask, size=K, fill_value=0)` gives the indices of the true elements, one `i32[K]`
  tensor per dimension, as in NumPy. As in JAX, the size is static: the result is padded with
  `fill_value`, or cut off after `K` indices. This is how a program turns "the legal moves" into a
  fixed-size list.

### 3.5 Index notation (tensor comprehensions)

```python
y[i] = sum W[i, j] * x[j]                 # j is reduced (it is not on the left-hand side)
m[i] = max x[i, j]                        # max / min / sum / mean / prod / argmax reductions
lse[i] = log(sum exp(x[i, j] - m[i])) + m[i]
loss = -mean(logp[i, labels[i]])          # gather + reduction over i, no LHS needed
out[..., j] = e[..., j] / sum e[..., k]   # ellipsis = all leading dimensions
pool[n, c, i, j] = max x[n, c, 2*i + u, 2*j + v] where u < 2, v < 2
counts[labels[i]] += 1                    # scatter-add: index names on the left loop, values add up
```

* **Binding rule.** Names on the left are the output indices. Any other name used in a subscript
  that is not a variable is an *index variable*. It is bound by the **innermost reduction that
  encloses all of its uses** (the lowest common ancestor). An index that no reduction encloses is
  an error that suggests the fix.
* **Precedence** follows math typesetting: `sum a[i]*b[i] + c` is `(Σ a·b) + c`.
* **Range inference.** A plain index `x[i]` fixes the range of `i` exactly, and every plain use must
  agree, which is the shape check. Affine uses (`x[i + t]`, `x[2*i + u]`) bound it: the compiler
  infers the largest in-bounds range ("valid" convolution falls out for free) and proves every
  access in-bounds. `where u < 2` gives a range explicitly.
* Runtime offsets (`x[s + i]`) are checked once at kernel entry. Gathers (`x[labels[i]]`) are
  checked per element.

### 3.6 Differentiation

```python
dW, db = grad(loss, W, b)                 # any tensors in scope, not just params
minimize loss with sgd(lr=0.1)            # all params that loss depends on
maximize elbo over encoder, decoder with adam()
x_adv = x + 0.1 * sign(grad(loss, x))     # input gradients (FGSM) work the same way
h = detach(h)                             # stop-gradient
```

* Reverse mode, over the kernel IR, at compile time. The loss must be a scalar, so the error
  message suggests `sum`/`mean` when it is not.
* Gradients flow through every computation in the current loop iteration. Values defined outside
  the loop are constants. Differentiating through runtime control flow is reported as an error;
  use `a if c else b`.
* If a parameter is modified between the forward pass and `grad`, the compiler raises an error
  instead of silently producing wrong gradients.

### 3.7 Randomness

```python
param W: [784, 128] ~ normal(0, 0.05)
eps: [64, 16] ~ normal(0, 1)              # fresh sample each time this line runs
keep: [64, 128] ~ bernoulli(0.9)          # dropout mask
seed(42)
```

The generator is a counter-based hash (`lowbias32` over element index × stream × seed). It is
vectorized like any other kernel, it is reproducible, and it is independent of evaluation order.
Normals use Box–Muller with inline NEON `log`/`cos`.

### 3.8 Data

* `idx("file.gz")` loads MNIST-family IDX files (gzipped or not). The compiler reads the header
  **at compile time** to get the shape, and the binary checks it again at run time.
* `batches(X, Y, size=64, shuffle=true)` yields zero-copy views (gathers through a permutation
  when shuffling). The partial last batch is dropped so that shapes stay static.
* `save(net, "net.weights")` writes a model's parameters (or one `param`) next to the program, and
  `load(net, "net.weights")` reads them back. `load` returns 1, or 0 when the file is missing or
  holds tensors of other sizes, and then it changes nothing. So a program can reuse what an
  earlier run learned: `if load(net, f) == 0: train()`.
* `csv("file.csv", sep=",")` loads a table of numbers as `f32[rows, cols]`. The compiler reads
  the file for its shape, checks that every field is a number and every row has the same width,
  and skips a header line. The program reads it again when it runs.
* Planned: `bin("file.f32", [n, d])` and memory-mapped datasets.

### 3.9 Control flow, variables, scoping

* `for v in range(…)`, `for x, y in batches(…)`, `while`, `if/elif/else`, `break`, `continue`.
  An `if` on a compile-time condition is resolved at compile time.
* **Locals have value semantics** (they are SSA under the hood). A variable assigned in a loop is
  *loop-carried*: it gets one canonical buffer, and the final value of each iteration is written back
  to it. The copy is elided whenever the producer can write in place. Python's "variables leak out of
  loops" behaviour is preserved.
* **Params and optimizer state have reference semantics**: an assignment mutates them in place at
  that point in the program.
* `x[ptr] = v` is an item assignment when `x` is already a variable of the current function (or of
  the top level) and `ptr` has a value. Otherwise `x[m] = …` is an index definition, and its index
  names shadow any outer variables called `m`.
* Functions are inlined, so recursion must end at compile time. A depth argument that counts down
  to a constant unrolls into nested loops, as in a game-tree `search(…, depth)`.

### 3.10 Output

`print("epoch {epoch}: loss {loss:.4f}  acc {acc:.1%}")`. Format specs are `[width][.prec][f|e|g|d|%]`.
Tensors print NumPy-style and are summarised with `…` when large. A list of strings can be indexed
with a run-time integer inside a string (`"{FILES[col]}{row}"`), and a function may return a
string for another one to include. `show(grid, glyphs)` draws small integers as text.

`input("your move: ")` prints the prompt, reads a line, and returns the first decimal number in it
as an `f32`, or `nan` if there is none. When the input ends, the program ends.

### 3.11 Standard library (`prelude.anvil`, written in Anvil)

| Area | Functions |
|---|---|
| activations | `relu leaky_relu sigmoid tanh gelu silu softplus` |
| normalisation | `softmax log_softmax layer_norm` |
| losses / metrics | `cross_entropy mse bce accuracy` |
| layers | `Linear`, `Conv2d(c_in, c_out, k, stride, pad)` (im2col), `max_pool2d`, `LayerNorm`, `Embedding` |
| optimizers | `sgd` (with momentum), `adam`, `adamw`, `rmsprop` |
| tensors | `zeros ones full arange eye randn rand reshape flatten transpose argmax` |

### 3.12 Diagnostics

Errors are rendered with the source line, a caret span, and notes:

```
error: shape mismatch in `@`
  ┌─ examples/mnist.anvil:14:12
  │
14│     return h @ W1 + b2
  │            ━━━━━━ cannot multiply [64, 128] by [784, 128]
  │
  = note: inner dimensions must agree (128 ≠ 784)
```

Undefined names get "did you mean …" suggestions. Runtime failures (out-of-bounds gather, data
file shape mismatch, a failed `assert`) report the `.anvil` source location they came from.

Every independent error in a file is reported, up to 20. When a statement fails, the names it
would have defined are bound to an error value, and later uses of them stay quiet instead of
cascading. Inside an inlined function, the first error ends the call.

---

## 4. Compiler architecture

```
 .anvil ──► Lexer ──► Parser ──► AST
                               │
                               ▼
                         Elaborator ── staging: consts folded, fns/models inlined,
                               │         shapes checked, comprehensions → kernels,
                               │         locals → SSA, views → affine offsets
                               ▼
                ┌──────── Kernel IR (blocks of kernels + structured control flow)
                │              │
       grad / minimize ──► Autodiff (reverse mode over kernels, symbolic scalar derivatives)
                               │
                               ▼
                         Optimizer ── DCE · write-back coalescing · elementwise inlining ·
                               │      epilogue fusion · horizontal fusion · dead-state removal
                               ▼
               ┌───────────────┴────────────────┐
               ▼                                ▼
       AArch64 backend                    NumPy interpreter
   schedule → MIR → regalloc → .s          (reference semantics,
               │                            differential testing)
               ▼
         cc model.s -lz  ──►  native executable
```

### 4.1 Front end
A hand-written lexer, with an indentation stack and Unicode aliases, feeds a recursive-descent /
Pratt parser. Every AST node carries a source span, and diagnostics are rendered from spans.

### 4.2 Elaboration (the staging interpreter)
The elaborator *executes the program at compile time over symbolic tensors*, in the style of
JAX tracing, but it keeps runtime loops as real loops:

* Compile-time values (`const`s, shape variables, model arguments, hyper-parameters) are folded.
* Function and method calls are inlined after unifying shapes against signatures.
* Every tensor-level operation becomes one **kernel**. Views (slice, transpose, reshape,
  broadcast, select, gather) become **affine offsets** on loads and are never copied.
* Comprehensions: index variables are collected, reductions are bound by the LCA rule, ranges are
  inferred and bounds proven, and nested reductions are hoisted into their own kernels.
* Variables are converted to SSA. Loop-carried and branch-merged locals get canonical buffers
  plus write-backs.

### 4.3 Kernel IR

```
Kernel   := domain vars (with extents)
            [ reduction: vars, op ∈ {sum,max,min,prod,argmax}, body: Expr ]
            stmts: (let name = Expr | store buf[Affine] (= | +=) Expr)*
            checks: runtime bounds assertions
Expr     := const | load buf[Affine] | index(Affine) | acc | let
          | unary(op, e) | binary(op, a, b) | select(c, a, b) | rand(salt)
Affine   := c₀ + Σ cᵢ·varᵢ + Σ dⱼ·runtime_scalarⱼ + Σ eₖ·gather(load)ₖ
Program  := blocks of: kernel | for | while | if | print | runtime call | break | continue
```

Using flat affine offsets means a view costs nothing, and the backend only has to understand
one addressing form.

### 4.4 Automatic differentiation
For each forward kernel `Y[I] = ⊕_J body(I, J)`, taken in reverse order, and each load
`X[f(I, J)]` of an *active* buffer (on a path from a target to the loss):

* `∂body/∂load` is computed symbolically (rules for `+ − × ÷ exp log sqrt tanh sigmoid max min pow select …`).
  The result is simplified, and forward outputs are reused where they appear (`d exp(a) = Y`).
* If `f` is an injective, dense map of the variables (the common case), the contribution is an
  ordinary **reduction kernel**. For matmul: `dW[i,o] += Σ_b x[b,i] · dY[b,o]`.
* Otherwise (convolution windows, gathers, runtime offsets) it becomes a **scatter-accumulate
  kernel** into a zero-initialised gradient buffer.
* `max`/`min` reductions route gradients through an equality mask, and `detach` stops gradients.
* A buffer may be written by several kernels (a copy, then `x[1] = …` or `x[ids[i]] += …`). All
  of them count for reachability. An accumulating store passes the gradient on unchanged to
  earlier writers, and an overwriting store zeroes the gradient of the elements it replaced.
* The first full write to a gradient buffer uses `=` instead of `+=`, so most gradients need no
  zero fill.

`minimize` then inlines the optimizer's `step` for each active parameter. Because the result is
just more kernels, the SGD update fuses into the epilogue of the weight-gradient GEMM, and
`dW` is never written to memory.

### 4.5 Optimizations
* **DCE**, including whole-program *dead state* removal. For example, `sgd` with `momentum = 0`
  folds its momentum buffer away entirely.
* **Write-back coalescing**: the final producer of a loop-carried variable writes straight into its
  canonical buffer. This is how in-place updates arise.
* **Elementwise inlining**: a producer is substituted into its consumer when no work is duplicated.
* **Epilogue fusion**: an elementwise consumer runs inside a reduction's store loop
  (`relu(x @ W + b)` is one kernel), and producers may sink or consumers hoist across independent
  kernels to make this possible.
* **Horizontal fusion**: adjacent elementwise kernels over the same domain become one loop (Adam's
  `m`, `v`, `w` updates). A matmul-like reduction takes at most a small epilogue (SGD's update),
  because a big one (Adam's) would cost it its register tile.
* **Window expansions stay materialized**: a copy that reads some input elements several times
  (im2col: `patches[n, i, j, c, u, v] = x[n, c, i + u, j + v]`) is never inlined, so the
  convolution and both of its gradients become matrix multiplies over `(c, u, v)`.
* **Loop collapsing** (last): adjacent loops that every access sees as one index, `a·n_b + b`,
  merge into one longer loop. `dW[o, c, u, v]` becomes `dW[o, cuv]`, a 4×16-tileable GEMM. The
  iteration order, and so every result, stays the same.
* **Dead random draws still count**: removing an unused `~ normal(…)` leaves a statement that
  advances the random stream, so later random numbers are the same as without optimization.
* **Static memory plan**: every buffer has a fixed address in `__bss`. Temporaries share one arena:
  the program is flattened into statement positions, and each temporary's lifetime runs from its
  first use to its last. A lifetime that reaches into or out of a loop covers the whole loop. Only
  temporaries whose first use writes all of them take part, and that write may not sit inside an
  `if` or loop that a later use is outside of. They are placed first-fit, largest first.

### 4.6 AArch64 backend
Each kernel becomes a leaf function. Its schedule is chosen per kernel:

| Mode | When | Inner loop |
|---|---|---|
| **vec-out** | innermost output dim is contiguous | 4×f32 NEON lanes × unroll 4, accumulators in registers across the whole reduction |
| **vec-red** | output dim strided, reduction dim contiguous | vector accumulate + horizontal `faddp`/`fmaxv` |
| **scalar** | otherwise / tails | same instruction selection on lane 0 |

* **GEMM-like kernels** get register tiling (R rows × U vectors of accumulators) with broadcast
  operands hoisted.
* **Addressing**: one pointer register per distinct access, advanced by its per-loop stride and
  rewound on loop exit. Gathers recompute their address at their dependency level.
* **LICM**: each sub-expression is evaluated at the outermost loop level it does not depend on.
  Constants live in registers.
* **FMA contraction** (`fmla`) for `acc + a*b`.
* **Inline vectorized math**: `exp` (Cody–Waite + degree-6 polynomial), `log` (exponent split +
  atanh series), `tanh` (via expm1), `sigmoid`, `sin`/`cos`. The hot loops never call into libm.
* **Register allocation**: the kernel is lowered to a machine IR over virtual registers, then linear
  scan with loop-extended live ranges. If it runs out of registers, the scheduler retries with a
  smaller tile.
* `main` holds the control flow. Kernels use an internal convention (they clobber anything,
  `main` keeps nothing live in registers), and libc calls follow the Apple arm64 ABI. Variadic
  `printf` arguments go on the stack.

### 4.7 Runtime
A small hand-written assembly runtime is appended to every output file. It provides tensor
printing, IDX/gzip loading (zlib), Fisher–Yates shuffling, a monotonic clock, runtime-error
reporting, and a **thread pool**. The only link dependencies are `libSystem` and `libz`.

**Threads.** A large kernel whose stores are disjoint across its outermost loop variable becomes a
function of a `[start, end)` range. The runtime splits that loop into chunks and hands them out
with compare-and-swap on a single `(generation, next chunk)` word. Idle workers sleep in `wfe`
rather than spinning. Each output element is computed by the same code whichever thread runs it,
so results are bit-identical for any thread count.

### 4.8 Toolchain
`anvil build` writes `model.s`, then runs `cc -arch arm64 model.s -lz -o model`. `anvil run` builds into
a cache directory and executes.

---

## 5. Tooling

```
anvil run   file.anvil              compile, assemble, link, run
anvil build file.anvil [-o exe]     native executable (+ memory report)
anvil asm   file.anvil [-o file.s]  annotated assembly (each kernel headed by its source + math form)
anvil ir    file.anvil              optimized kernel IR in math notation, with fusion decisions
anvil check file.anvil              parse + type/shape check only
anvil run --interp file.anvil       run on the NumPy reference interpreter
anvil repl                        try Anvil a line at a time (on the reference interpreter)
anvil run file.anvil --set N=V      give the program's `const N` the value V (repeatable): settings
                                without editing the file, e.g. `bin/checkers` is checkers.anvil
                                with `--set PLAY=1 --set WATCH=0`
```

Editor support: a TextMate grammar (`editors/vscode`) for highlighting in VS Code, Cursor, Zed, etc.

---

## 6. Testing strategy

1. **Unit tests**: lexer, parser, shape inference, range inference, symbolic derivatives.
2. **Error tests**: programs that must fail, checked against the expected message and span.
3. **Differential tests**: every program is run natively and on the NumPy interpreter, and the
   outputs must agree (to a float tolerance).
4. **Gradient checks**: AD gradients are compared with central finite differences in float64 on the
   interpreter.
5. **Golden examples**: `examples/*.anvil` with expected outputs. MNIST must reach ≥97% test accuracy.
6. **Codegen stress**: random shapes (tails, 1-sized dims, odd extents) for every schedule mode.

---

## 7. Performance targets (MVP)

* MNIST MLP 784-128-10, batch 64: **≥ 97% test accuracy**, an epoch in roughly a second on one core.
* GEMM-like kernels: a meaningful fraction of single-core NEON peak through register tiling.
  (BLAS/AMX parity is explicitly *not* an MVP goal.)
* Compile time under 1 s for the examples.

**Outcome:**
- **Accuracy.** After 5 epochs, plain SGD (lr 0.1) reaches 96.9% and `adam(lr=1e-3)` reaches
  97.3%. The 97% target is met with Adam; SGD would need a sixth epoch.
- **Speed.** An epoch takes 0.40 s on one core and 0.10 s on 8 threads.
- **GEMM.** About 70% of single-core NEON peak. With threads, the whole training loop is faster
  than NumPy on Accelerate/AMX.
- **Compile time.** 0.15–0.3 s for every example.

---

## 8. Roadmap

| Milestone | Contents |
|---|---|
| **MVP (this build)** | everything in §10; multithreaded AArch64/macOS backend; NumPy interpreter; examples (hello, linear regression, spirals, softmax regression, MNIST MLP, CNN, attention, DQN Snake, self-play checkers); diagnostics; VS Code grammar; `--profile` |
| **v0.2** | `bin` loader · optimizer state in checkpoints · memory reuse for variables, not just temporaries · a formatter |
| **v0.3** | Linux/AArch64 + x86-64/AVX2 backends · differentiation through run-time loops (`static for` is done) · parallel reductions |
| **v1.0** | dynamic batch dimension · mixed precision (f16/bf16) · packages (`use` of declarations is done) · language server · formatter |
| **v0.2 (done early)** | editor support: errors as you type, shapes on hover and as inlay hints (`anvil check --json`) · CUDA backend (one `.cu` file; tested in emulation and with clang's CUDA front end) · `anvil.function` (Anvil functions called from Python on NumPy arrays, zero-copy) · benchmarks and experiments suites |
| **next** | tiled MSL kernels for the GPU's non-product work; batch norm; differentiation through run-time loops; WebAssembly |
| **beyond** | running and tuning the CUDA backend on a GPU (shared-memory tiles for matmuls) · Metal backend (kernels map 1:1 to compute shaders) · auto-scheduling (search over tile sizes) · vmap/jvp transforms |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Codegen bugs (wrong answers fast) | NumPy interpreter as an oracle; differential + gradcheck tests on every example |
| Register pressure in large fused kernels | linear-scan allocator with automatic re-schedule at smaller tiles; split kernels as last resort |
| Fusion legality | buffer-level dependence checks (RAW/WAR/WAW) before any move; in-place only for pointwise accesses |
| AD across control flow / mutation | detected and reported as compile errors with suggestions |
| Static shapes feel rigid | shape variables + polymorphic functions; data shapes inferred from files; dynamic batch on roadmap |

---
