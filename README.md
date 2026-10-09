# Anvil

**A small language for machine learning that compiles to ARM64 assembly, the Apple GPU (Metal), or CUDA.**

*Anvil: **A**utodiff, **N**ative, **V**ectorized, **I**ndex **L**anguage.* You hammer the math out in
index notation; the compiler forges it into machine code.

You write a model the way it's written on paper: tensors, index notation, a loss to minimize.
Anvil compiles the whole program, including the forward pass, the backward pass, and the
optimizer, into **one self-contained assembly file** of fused, vectorized NEON loops. There is
no framework, no Python at run time, and no allocation while training. The same programs also
run on the Apple GPU (`--metal`) or compile to a single CUDA file. Any Anvil function can be called
from Python and Jupyter on NumPy arrays, or exported as a C library with its trained weights inside.

```python
# examples/mnist.anvil
images = idx("data/train-images-idx3-ubyte.gz") / 255          # f32[60000, 28, 28]
labels: i32[60000] = idx("data/train-labels-idx1-ubyte.gz")
test_images = idx("data/t10k-images-idx3-ubyte.gz") / 255
test_labels: i32[10000] = idx("data/t10k-labels-idx1-ubyte.gz")

X = images.reshape(60000, 784)
X_test = test_images.reshape(10000, 784)

model MLP:
    l1 = Linear(784, 128)
    l2 = Linear(128, 10)
    fn forward(x: [n, 784]) -> [n, 10] = x |> l1 |> relu |> l2

net = MLP()

start = clock()
for epoch in range(5):
    for x, y in batches(X, labels, size=64, shuffle=true):
        loss = cross_entropy(net(x), y)
        minimize loss with sgd(lr=0.1)
    acc = accuracy(net(X_test), test_labels)
    print("epoch {epoch + 1}   loss {loss:.4f}   test accuracy {acc:.2%}   ({clock() - start:.1f}s)")
```

```
$ bin/anvil run examples/mnist.anvil
epoch 1   loss 0.2079   test accuracy 93.55%   (0.1s)
epoch 2   loss 0.2904   test accuracy 95.32%   (0.1s)
epoch 3   loss 0.0991   test accuracy 96.25%   (0.2s)
epoch 4   loss 0.1284   test accuracy 96.80%   (0.3s)
epoch 5   loss 0.0741   test accuracy 96.84%   (0.4s)
```

That run compiled 30 lines of Anvil into 4,200 lines of AArch64 in 0.16 s, then trained five
epochs in under half a second. The same model written in NumPy on Apple's Accelerate BLAS takes
almost twice as long.

## Highlights

- **Index notation is the core of the language.** `y[i] = sum W[i, j] * x[j]` is valid Anvil.
  Indices on the left are outputs, and every other index is reduced. A convolution is one line,
  and its output size is inferred.
- **Shapes are types, with names.** Every shape is known at compile time, and every dimension
  remembers what it is: hovering shows `f32[BATCH, T, HEADS, DH] = [16, 64, 4, 16]`, and a call
  shows `forward(x: f32[BATCH, T, D]) -> f32[BATCH, T, 4*D]`. Two dimensions that must agree must
  be the same dimension, so swapping `T` and `D` is a compile error even when both are 64
  ([below](#named-dimensions)). Shape errors point at the exact operand.
- **Differentiation is built in.** `grad(loss, W)` and `minimize loss with adam()` generate the
  backward pass *at compile time*, so it gets fused and vectorized like everything else.
  The SGD update ends up inside the weight-gradient matrix multiply, and the gradient is never
  stored to memory.
- **Readable assembly.** Each kernel becomes one function, headed by the math it computes.
- **Multithreaded.** Large kernels split their outermost loop across cores. On a given machine,
  results are bit-identical for any number of threads. Another chip can round Accelerate's matrix
  products differently.
- **The standard library is written in Anvil.** Layers, losses, and optimizers live in
  [`anvil/prelude.anvil`](anvil/prelude.anvil), so you can read, copy, and change them.
- **Static and safe.** Every buffer has a fixed address. Index bounds are proven at compile time
  when possible, and otherwise checked once per kernel, or per element for gathers.
- **Small.** Temporaries whose lifetimes do not overlap share memory, so the checkers player
  needs 47 KB of scratch space instead of 332 KB. `anvil build` reports the memory by kind.
- **Callable from Python.** `anvil.function(source)` compiles an Anvil function for the shapes of the
  NumPy arrays you call it with. Softmax, layer norm, attention and a linear layer run 3–19× faster
  than the same NumPy code ([below](#from-python)).
- **On the GPU.** `anvil run --metal` runs the program on the Apple GPU, with matrix products on
  Metal Performance Shaders. It trains a 784-1024-1024-10 MLP 3.7× faster than the CPU, and the CPU
  is already 2× faster than NumPy ([below](#on-the-gpu-metal)).
- **CUDA too.** `anvil cuda file.anvil` writes the program as one `.cu` file: a kernel per Anvil kernel,
  host code for the control flow, and the same random numbers as the CPU ([below](#cuda)).
- **Finds bugs before and while they happen.** `--check` stops at the first NaN and shows the
  line and the operation that made it. The compiler warns about parameters the loss cannot train.
  `anvil cost` tells you a program's FLOPs, memory, and memory traffic, line by line, before it runs
  ([below](#finding-bugs-and-costs)).
- **Deploys without Python.** `anvil export` turns a function into a C library and header, with
  everything it reads (its trained weights) built in ([below](#deploying-anvil-export)).

## Quick start

You need an Apple Silicon Mac, Python 3.10 or later, and the Xcode Command Line Tools (for
`cc`). The compiler has no Python dependencies. NumPy and pytest are needed only for the tests and
the reference interpreter.

```bash
bin/anvil run examples/hello.anvil
```

```bash
bin/anvil run examples/mnist.anvil          # MNIST is in examples/data
```

```bash
bin/checkers                            # play checkers against a network that taught itself
bin/draw                                # draw digits in your browser; a network written in Anvil reads them
bin/draw --text                         # write English by hand; it reads your words (needs EMNIST)
```

Install the `anvil` command:

```bash
pip install -e .
```

## From Python

`anvil.function` turns Anvil source into a Python function on NumPy arrays. The first call with new
shapes compiles it to native code (a shared library, cached in `~/.cache/anvil`); the arrays are passed
by pointer, never copied.

```python
import numpy as np
import anvil

attention = anvil.function("""
fn attention(q: [b, t, d], k: [b, t, d], v: [b, t, d]):
    s[n, i, j] = sum q[n, i, c] * k[n, j, c] / sqrt(d) + (0.0 if j <= i else -1e9)
    return softmax(s) @ v
""")

q, k, v = (np.random.randn(16, 256, 64).astype(np.float32) for _ in range(3))
out = attention(q, k, v)            # [16, 256, 64], float32
```

- Floating-point arrays become `f32` tensors; integer and boolean arrays become `i32` tensors.
- Python numbers and strings are compile-time constants, so they can set shapes or loop counts.
  A new value compiles a new version.
- Functions can return several results, and they can use `grad`. For example, this returns a
  loss and its gradient for a training loop written in Python:

```python
step = anvil.function("""
fn loss_and_grad(w: [d], x: [n, d], y: [n]):
    loss = mean((x @ w - y) ** 2)
    return loss, grad(loss, w)
""")
loss, g = step(w, x, y)
```

| called from Python (M3 Max, busy) | `anvil.function` | NumPy + Accelerate | |
|---|---|---|---|
| layer norm, 4096×1024 | 0.25 ms | 4.7 ms | **19×** |
| softmax over rows, 4096×1024 | 0.76 ms | 8.3 ms | **11×** |
| causal attention, 16×256×64 | 1.30 ms | 4.8 ms | **3.7×** |
| `gelu(x @ W + b)`, 256×1024 → 1024 | 0.40 ms | 1.23 ms | **3.1×** |

NumPy makes a separate pass over memory for each operation, and runs most of them on one core.
Anvil fuses each function into a few multithreaded loops. Where NumPy spends its time in one large
matrix multiply, Apple's AMX coprocessor makes up the difference (last row).

**In Jupyter**, `%load_ext anvil` adds `%%anvil` cells. A cell of declarations defines its functions
in the notebook (as `anvil.function`s). Any other cell is compiled and run, and what it prints
appears as it runs:

```python
%%anvil
fn attention(q: [t, d], k: [t, d], v: [t, d]) = softmax(q @ k.T / sqrt(d)) @ v
```

```python
%%anvil --check
x = npy("activations.npy")
print(mean(log(x)))
```

A function's arguments are passed by pointer. The rest of the source is computed once, when the
function is compiled, and built in: models, weights from `load`, tables. A call then only computes
the function. The digit reader of `bin/draw` takes 0.04 ms a call.

## Deploying: anvil export

`anvil export` turns an Anvil function into a C library and header. Everything the function reads from
the rest of the program is computed once, when exporting, and built into the library: models,
their trained weights (from `load`), tables. The library needs no weights file and no Python.

```bash
bin/anvil export examples/digits_reader.anvil -o build/
```

```c
#include "digits_reader.h"            // void digits_reader_read(const float *image, float *out);

float image[DIGITS_READER_READ_IMAGE_SIZE], probs[DIGITS_READER_READ_OUT_SIZE];
digits_reader_read(image, probs);     // cc app.c -Lbuild -ldigits_reader -Wl,-rpath,build
```

[`examples/digits_reader.anvil`](examples/digits_reader.anvil) is the trained digit reader of
`bin/draw` as a function, and its library is 700 KB. A C program that reads MNIST's test images
with zlib gets 986 of the first 1,000 right with it. The arguments need concrete shapes
(`image: [28, 28]`); calls from several threads take turns.

## A tour of the language

### Tensors, shapes, broadcasting

```python
x = [1.0, 2.0, 3.0]                       # f32[3]
W: [2, 3] ~ normal(0, 1)                  # declared shape, randomly initialized
y = W @ x + 1                             # NumPy broadcasting rules
batch = 64                                # assigned once → a compile-time constant, usable in shapes
z: [batch, 10] ~ uniform(-1, 1)
```

Element types are `f32` (the default) and `i32`. Comparisons give `f32` masks, so
`mean(pred == y)` is accuracy. Slices, transposes (`x.T`), reshapes and integer-tensor gathers
(`table[ids]`) are zero-copy views.

### Named dimensions

Every integer `const` names a dimension, and shapes keep the names through everything the program
does with them: arithmetic, reshapes, slices, flattening, index notation, layers. Hovering over any
name shows them, and the numbers they stand for:

```python
const BATCH = 16
const T = 64
const D = 64
const HEADS = 4
const DH = D // HEADS                      # hover: 16 = D/HEADS

q = (h @ Wq).reshape(-1, T, HEADS, DH)     # hover: f32[BATCH, T, HEADS, DH] = [16, 64, 4, 16]
s[b, a, i, j] = sum q[b, i, a, e] * k[b, j, a, e]      # f32[BATCH, HEADS, T, T]; i < T (64)
y = l(x)                                   # hover on l: forward(x: f32[BATCH, T, D]) -> f32[BATCH, T, 4*D]
```

Two dimensions that have to agree (an index used on two tensors, the two sides of `+`, the inner
dimensions of `@`, a function's signature, an annotation, a variable across loop iterations) have to
be **the same dimension, not just the same number**. Here `T` and `D` are both 64, so a swapped
`P[d, t]` would compute something and train silently worse; it would fail as soon as either size
changed. A program that would fail with other sizes does not compile with these:

```
error: index `t` ranges over `T` in one place but `D` in another
48│         e[b, t, d] = E[codes[b, t], d] + P[d, t]
  │                        ─────────── `x[b, t]` dimension 1: size T
  │                                          ━━━━━━━ `net.P[d, t]` dimension 1: size D
  = note: `T` and `D` are both 64 here, but they are different dimensions: the program would fail with other sizes
  = help: if they are meant to be the same size, define one from the other (e.g. `const D = T`)
```

A size written as a number, or read from a data file, has no name and matches any dimension of that
size; an annotation can give it one (`images: [N, 28, 28] = idx(...)`). `anvil shapes file.anvil`
prints the whole program this way: its named dimensions, then every tensor it defines, line by line.
The names exist only at compile time: the generated code is the same with or without them.

[`examples/named_dims.anvil`](examples/named_dims.anvil) reads MNIST digits row by row with
attention. Its sizes collide the way real ones do (square images, a model as wide as the batch), and
it carries three one-line bugs behind `--set BUG=1` (2, 3): reading columns as rows, weights on the
wrong side of `@`, and an index typo. With unnamed sizes each one compiles and trains (two of them to
90% instead of 93%, by mixing the examples of a batch); with names, none of them compiles.

### Index notation

```python
y[i] = sum W[i, j] * x[j]                         # matrix-vector product
outer[i, j] = x[i] * x[j]
m[i] = max x[i, j]                                # max / min / sum / mean / prod / argmax
loss = -mean(logp[i, labels[i]])                  # gather + reduction, no left-hand side needed
out[..., j] = e[..., j] / sum e[..., k]           # `...` = all leading dimensions
conv[n, o, i, j] = sum k[o, c, u, v] * img[n, c, i + u, j + v]   # output size inferred
pool[n, c, i, j] = max x[n, c, 2*i + u, 2*j + v] where u < 2, v < 2
counts[labels[i]] += 1                            # scatter-add into an existing tensor: a histogram
conf[labels[i], pred[i]] += 1                     # a confusion matrix
x[b, t] = text[starts[b] + t] where t < 64        # windows at gathered offsets (bounds-checked)
```

Each index is bound by the innermost reduction that contains all of its uses. Index ranges are
inferred from the tensors they subscript, and every use must agree, which is the shape check.
`sum a[i] * b[i] + c` reads like math, `(Σ a·b) + c`, and `mean (x - m) ** 2` (with a space) is
the mean of the square. With `+=` (or `-=`), the index names in the subscripts loop over the
right-hand side and add into a tensor that already exists. Where several of them reach the same
element, the contributions add up, and gradients flow through it.

### Functions and shape polymorphism

```python
fn relu(x) = max(x, 0)

fn dense(x: [n, d], W: [d, k]) -> [n, k] = x @ W     # n, d, k are unified at each call

fn attention(x: [b, t, dm], Wq: [dm, dh], Wk: [dm, dh], Wv: [dm, dh]) -> [b, t, dh]:
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    s[..., i, j] = sum q[..., i, c] * k[..., j, c] / sqrt(dh)     # shape variables are constants
    out[..., i, c] = sum softmax(s)[..., i, j] * v[..., j, c]
    return out
```

Functions are inlined, which also makes them generic. A name in a signature is a shape
variable unless it is a `const`. The pipeline `x |> f |> g(2)` means `g(f(x), 2)`, and
`x.f(a)` means `f(x, a)`. Recursion is fine when it ends at compile time: a depth argument that
counts down to a constant unrolls into nested loops.

`static for t in range(T):` unrolls a loop at compile time. There is no run-time loop, so values
flow from step to step as ordinary variables, and `minimize` differentiates through every step.
That is backpropagation through time, and it is how
[`charrnn.anvil`](examples/charrnn.anvil) trains a recurrent network.

`use "model.anvil"` brings in another file's declarations: its functions, models, optimizers and
constants. A file is brought in once, however many times it is used. The digit reader's model
lives in [`examples/digits_model.anvil`](examples/digits_model.anvil). The program that trains it
`use`s it, and so does the drawing app that runs it.

### Models

```python
model Linear(fan_in, fan_out):                       # (this one is in the prelude)
    param W: [fan_in, fan_out] ~ normal(0, sqrt(2 / fan_in))
    param b: [fan_out]
    fn forward(x) = x @ W + b

model MLP:
    l1 = Linear(784, 128)
    l2 = Linear(128, 10)
    fn forward(x) = x |> l1 |> relu |> l2

net = MLP()          # params are named net.l1.W, net.l1.b, ...
```

The prelude also has `Conv2d(c_in, c_out, k, stride=1, pad=0)`, `max_pool2d(x, s=2)`,
`LayerNorm(d, eps=1e-5)` and `Embedding(n, d)`.

A list comprehension builds a list at compile time. A list of models is a stack of layers, named
the way PyTorch's `ModuleList` names them:

```python
model Encoder(n):
    layers = [Layer(k) for k in range(n)]            # params layers.0.q.W, layers.1.q.W, ...
    fn forward(x):
        static for layer in layers:
            x = layer(x)
        return x
```

### Training

```python
minimize loss with sgd(lr=0.1, momentum=0.9)        # every param the loss depends on
minimize loss over net.l2 with adam(lr=1e-3)        # or just some of them
minimize loss with adam(lr=warmup_cosine(step, STEPS, 3e-3, warmup=100), clip_norm=1.0)
h = dropout(h, 0.1, training=TRAINING)               # TRAINING: a constant (false to test)
loss = cross_entropy(logits, labels, smoothing=0.1)  # label smoothing
maximize elbo with adam()
dW, db = grad(loss, W, b)                           # gradients with respect to any tensors
h = detach(h)                                       # stop-gradient
```

An optimizer's state belongs to the parameters: two `minimize` statements with the same optimizer
on the same parameters (say, one for each length of input) continue one Adam, as one PyTorch
optimizer would.

Optimizers are plain Anvil:

```python
optimizer adam(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0):
    state m, v                                       # per-parameter, zero-initialized
    step(w, g, t):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        w -= lr * ((m / (1 - beta1 ** t)) / (sqrt(v / (1 - beta2 ** t)) + eps) + weight_decay * w)
```

### Randomness, data, control flow, output

```python
param W: [784, 128] ~ normal(0, 0.05)       # sampled once
eps: [64, 16] ~ normal(mu, sigma)           # fresh sample each time; reparameterized gradients
keep: [64, 128] ~ bernoulli(0.9)
seed(42)                                     # runs are deterministic by default

idx: i32[64] ~ randint(0, count)            # random integers

images = idx("train-images-idx3-ubyte.gz")  # IDX/MNIST loader (gzip or raw)
table = csv("iris.csv")                      # f32[rows, cols]; the shape is read at compile time
text = bytes("README.md")                    # i32[n], one element per byte
W1 = npy("w1.npy")                           # a NumPy file (from np.save, or PyTorch's .numpy())
Wq = safetensors("model.safetensors", "encoder.layer.0.attention.self.query.weight")   # by name
save_npy(probs, "probs.npy")                 # … and back: np.load reads it
save(net, "net.weights")                     # a checkpoint of the parameters, next to the program
if load(net, "net.weights") == 0: ...        # 0: no such file (or other sizes), nothing changed
for x, y in batches(X, Y, size=64, shuffle=true): ...
for i in range(10): ...   while loss > 0.1: ...   if/elif/else, break, continue

S[ptr] = state                               # item / slice assignment, in place when safe
counts[k] += 1
v = [danger, f32(dir == 0), f32(fy < hy)]    # list literals can hold run-time values
x = stack([mine, theirs, kings], axis=-2)   # NumPy's stack
src, dir = nonzero(legal, size=32)           # indices of the true elements, padded to a static size

print("epoch {epoch}: loss {loss:.4f}  acc {acc:.1%}")     # {expr:fmt} in any string
print(W)                                                    # NumPy-style tensor printing
show(board, " #o*")                                         # draw small integers as text
sleep(0.05)
print("{FILES[c]}{8 - r}")                                  # a list of strings, indexed at run time
print(decode(sample))                                       # byte values printed as text
print("-" * 72)
assert loss == loss, "loss is nan at step {step}"           # (nan ≠ nan) checked as it runs
k = input("your move: ")                                    # the first number on the line (or nan)
```

`use "model.anvil"` brings in another file's functions, models, optimizers and constants. File
names are relative to the `.anvil` file that names them, so a file brought in with `use` finds its
own data. `safetensors` reads the format Hugging Face publishes models in: like `npy`, a tensor's
shape is read when the program compiles, and its numbers when it runs.

Unicode is optional: `Σ` for `sum`, `∇` for `grad`, `√x`, `→`, `≤ ≥ ≠`, `∞`, and Greek identifiers such as `θ` and `η`.

## Errors

```
error: index `j` ranges over 784 in one place but 128 in another
  ┌─ examples/errors/index_range.anvil:3:22
  │
3│ y[i] = sum W[i, j] * x[j]
  │            ─────── `W[i, j]` dimension 1: size 784
  │                      ━━━━ `x[j]` dimension 0: size 128
  │
  = note: every use of an index must agree on its range (this is the shape check)
```

There are more in [`examples/errors/`](examples/errors/): broadcasting, `@`, unbound indices,
out-of-bounds windows, a non-scalar loss, and typos (which get "did you mean `softmax`?"). Two more
kinds are about meaning rather than size: dimensions of equal size with different names
([Named dimensions](#named-dimensions)), and a gradient that would have to pass through a run-time
`for` loop or `if` (Anvil differentiates through `static for` and compile-time conditions; the error
says how to rewrite it). The compiler reports every independent error in a file, up to 20, not just
the first. A name whose definition failed is not reported again where it is used.

## Finding bugs and costs

**`--check` finds where a NaN comes from.** After every operation, the program looks for a NaN in
what it just computed. The first one stops it, with the line and the operation:

```
$ bin/anvil run --check model.anvil
anvil: --check: nan in `t3` (f32[4, 3]) at [0, 0]
  made by `log` at model.anvil:3:1:
      y = log(x - 0.5) * w
  computing (k30):  t3[i, j] = log(t2[i, j])
  its inputs held no nan or infinity: this computation made it
```

`--check=inf` stops at the first infinity too. When a NaN comes from infinities made earlier (say
`inf - inf`), the message says so. `--check` compiles without fusion, so each operation is its own
kernel and has its own line.

**Warnings for training bugs.** The compiler warns when the objective does not depend on a
parameter (`minimize` cannot train it), and when no `minimize` trains a `param` at all.

**`anvil cost` tells you what a program costs before it runs.** Shapes and most loop counts are
known when compiling, so the compiler counts every kernel's arithmetic and memory traffic and how
often it runs:

```
$ bin/anvil cost examples/mnist.anvil
mnist.anvil: 101,770 parameters (397.5 KB), memory 215.5 MB (data 209.6 MB, temp 5.3 MB, …)
the whole run: 134.9 GFLOP, 12.7 GB of memory traffic (roughly 0.45 s at 300 GFLOP/s)

  share      FLOP     traffic        runs   line
  46.9%     63.2 G      6.0 GB       4,685   mnist.anvil:28     minimize loss with sgd(lr=0.1)
  45.5%     61.4 G      3.2 GB       4,685   mnist.anvil:27     loss = cross_entropy(net(x), y)
   7.5%     10.2 G    214.6 MB           5   mnist.anvil:29     acc = accuracy(net(X_test), test_labels)
```

## What the compiler does

```
.anvil → parse → elaborate ─────────► kernel IR ──► autodiff ──► optimize ──► AArch64 ──► cc ──► executable
              (shapes, inlining,    (tensor        (reverse      (fusion,      (NEON,
               index notation,      comprehensions  mode, at     in-place,     register tiles,
               SSA, views)          over affine     compile      dead code)    inline exp/log)
                                    offsets)        time)
```

Every operation becomes one *kernel*, `out[I] = ⊕_J body(I, J)`. Autodiff, fusion, and code
generation are each written once, against that form. Here is a full training step for the MNIST
model as optimized IR (`bin/anvil ir examples/mnist.anvil`). The SGD updates run inside the gradient
reductions:

```
k86    let t3 = Σ_k x[i, k] * net.l1.W[k, j]; let t4 = t3 + net.l1.b[j]; t4[i, j] = t4  ⟨matmul+add⟩
k112   let t6 = Σ_k max(t4[i, k], 0.0) * net.l2.W[k, j]; let t7 = t6 + net.l2.b[j]; t7[i, j] = t7  ⟨matmul+add⟩
       … log-softmax, cross-entropy and their gradients (8 kernels) …
k186   let dt5 = Σ_j dt7[i, j] * net.l2.W[k, j]; let dt4 = dt5 * (1.0 if t4[i, k] > 0.0 else 0.0); dt4[i, k] = dt4  ⟨∂matmul+∂max⟩
k208   let dnet.l1.W = Σ_i dt4[i, j] * x[i, k]; let net.l1.W = net.l1.W[k, j] - 0.1 * dnet.l1.W; net.l1.W[k, j] = net.l1.W  ⟨∂matmul+sub⟩
       … the same for b1, W2, b2 …
```

The first layer's matmul becomes a 4×16 register-tiled micro-kernel with 16 accumulators and
by-element FMAs, split across threads by rows. This is its header and inner loop as emitted by
`bin/anvil asm examples/mnist.anvil`; the `«…»` notes are mine:

```asm
// ── k86 · matmul+add · examples/mnist.anvil:27:9 ──────────────────────────
//   let t3 = Σ_k x[i, k] * net.l1.W[k, j]
//   let t4 = t3 + net.l1.b[j]
//   t4[i, j] = t4
//   loops: i<64 × j<128  reduce k<784  ·  vectorized over the output's last index, 4×16 register tile; parallel over the outermost loop
...
Lk86_loop3:
    ldr s21, [x2]                       «x[i, k]»
    ldr q22, [x3]                       «W[k, j:j+4]»
    ldr q23, [x3, #16]
    ldr q24, [x3, #32]
    ldr q25, [x3, #48]
    ldr s26, [x2, #3136]                «x[i+1, k]»
    ldr s27, [x2, #6272]
    ldr s28, [x2, #9408]
    fmla v5.4s, v22.4s, v21.s[0]
    fmla v6.4s, v23.4s, v21.s[0]
    ...                                 «16 fmla in all»
    fmla v20.4s, v25.4s, v28.s[0]
    add x2, x2, #4
    add x3, x3, #512
    add x8, x8, #1
    cmp x8, #784
    b.lt Lk86_loop3
```

There is more detail in [docs/PLAN.md](docs/PLAN.md), the design document.

## On the GPU (Metal)

```bash
bin/anvil run --metal examples/transformer.anvil
```

`--metal` runs the program on the Apple GPU. `anvil metal file.anvil` writes the program as one
Objective-C++ file, with the kernels in Metal Shading Language.
- **Kernels.** They are generated as for CUDA: one thread per output element.
- **Matrix products.** These, including stacks and sums of them, go to Metal Performance
  Shaders.
- **Memory.** Every tensor lives in shared memory.
- **No waiting.** The host never waits for the GPU except to read what it computed: a print, a
  branch on a computed value, a data file.
- **Scalars on the CPU.** Loop counters, and scalars computed from them alone (`step % 100 == 0`,
  Adam's step count), are computed on the CPU and copied into each kernel's arguments. A training
  loop therefore runs without the CPU waiting on any step, and an `if` on the step number does not
  stop the GPU.

| one epoch (M3 Max) | CPU | GPU (`--metal`) | |
|---|---|---|---|
| MLP 784-1024-1024-10, batch 64 | 1.47 s | 0.40 s | **3.7×** |
| MLP 784-2048-2048-10, batch 256 | 2.31 s | 0.45 s | **5.2×** |
| the transformer, 500 steps | 1.3 s | 0.8 s | **1.6×** |
| MLP 784-128-10 (small) | 0.067 s | 0.11 s | 0.6× |

Small models are faster on the CPU: there, each step is a few microseconds of work per kernel,
and a GPU dispatch costs about as much. The GPU's output matches the reference interpreter on every
example (tested), random numbers included.

## CUDA

```bash
bin/anvil cuda examples/mnist.anvil          # writes examples/mnist.cu
nvcc -O3 -o mnist examples/mnist.cu -lz  # on a machine with an NVIDIA GPU
bin/anvil run --cuda examples/mnist.anvil    # both steps, then run it
```

The `.cu` file is self-contained:
- **Kernels.** Each Anvil kernel becomes a `__global__` function with one thread per output
  element. A reduction is a loop inside the thread.
- **Long reductions.** When a kernel has few outputs and a long reduction, such as a loss summed
  over a batch, the reduction is split across up to 1,024 threads per output and the partial
  results are combined in a second kernel.
- **Scatters.** Scatter-adds (`counts[labels[i]] += 1`, and the gradients of gathers) use
  `atomicAdd`.
- **Order-dependent kernels.** A kernel whose result depends on the order of its elements runs
  on a single thread.
- **Memory and host code.** Tensors live in unified memory. The host code holds the control flow,
  printing, data loading, checkpoints and `input`, as in the ARM64 backend.
- **Random numbers.** The generator is the same counter-based hash, so a program draws the same
  random numbers on the GPU as on the CPU.

Shapes are fixed when the program is compiled, and data files are read from where they were
then. To run the `.cu` file on another machine, copy the same data files there and set `ANVIL_DATA` to
their directory.

There is no NVIDIA GPU on the machine this was developed on, so the generated code is tested two
other ways:
- `--cuda-emulate` compiles the same `.cu` file as C++ (`-DANVIL_EMULATE` turns each kernel launch
  into a loop). The tests run the examples this way and compare the output with the reference
  interpreter.
- clang's CUDA front end type-checks every example's `.cu` for the device and the host. This
  catches a host function called from a kernel, for instance.

Running it on a real GPU is the next step. Performance there is unmeasured, and the simple
one-thread-per-output kernels do not use shared memory yet.

## Tools

| Command | |
|---|---|
| `anvil run file.anvil` | compile to native code and run it (`--seed N`, `-O0`, `-v`) |
| `anvil run --profile file.anvil` | time every kernel; prints a breakdown by source line at exit |
| `anvil run --interp file.anvil` | run on the NumPy reference interpreter instead |
| `anvil run file.anvil --set N=V` | give the program's `const N` the value `V` instead (repeatable) |
| `anvil build file.anvil [-o exe]` | a standalone executable, plus a memory report |
| `anvil asm file.anvil [-o out.s]` | the generated assembly |
| `anvil run --metal file.anvil` | run on the Apple GPU (`anvil metal file.anvil`: write the Objective-C++ / Metal file) |
| `anvil run --check file.anvil` | stop at the first NaN (`--check=inf`: or infinity), with the line that made it |
| `anvil cost file.anvil` | parameters, memory, FLOPs and memory traffic, line by line, before running |
| `anvil export file.anvil [-o dir]` | a function as a C library and header, weights built in (`--fn NAME`) |
| `anvil cuda file.anvil [-o out.cu]` | the program as one CUDA file |
| `anvil run --cuda file.anvil` | build with `nvcc` and run (`--cuda-emulate`: as C++ on the CPU) |
| `anvil ir file.anvil` | the optimized kernel IR, in math notation |
| `anvil check file.anvil` | type- and shape-check only (`--json`: errors and every name's shape, for editors) |
| `anvil shapes file.anvil` | the program's named dimensions, then every tensor it defines with its shape, line by line |
| `anvil repl` | try Anvil a line at a time (on the reference interpreter) |
| `ANVIL_THREADS=n` | worker threads (default: performance cores, at most 8) |
| `ANVIL_BLAS=0` | matrix products in Anvil's own kernels instead of Accelerate (the AMX coprocessor) |

```
$ bin/anvil run --profile examples/mnist.anvil
  time%        ms      calls   kernel
  17.4%      89.4       4685   k70    copy                    mnist.anvil:26
  25.5%     131.0       4685   k86    matmul+add              mnist.anvil:27
  13.4%      69.0       4685   k186   ∂matmul+∂max            mnist.anvil:28
  25.7%     131.7       4685   k208   ∂matmul+sub             mnist.anvil:28
   ...
```

**In VS Code and Cursor**, the extension in [`editors/vscode/`](editors/vscode/) runs the
compiler as you type:
- **Errors** are underlined where they happen. An error inside the standard library is underlined
  at the call in your file that led to it.
- **Hovers.** Hovering over any name shows what it holds: `f32[64, 784]`, `index j < 10`, a
  function's signature, or a model's parameters and their count.
- **Shapes after definitions.** Every tensor definition shows its shape after the name, as
  `h: f32[64, 128] = …`.

It also highlights syntax (index notation, shapes, `where`, f-strings) and has snippets:

```bash
python3 editors/vscode/package_vsix.py
cursor --install-extension editors/vscode/anvil-language-0.3.1.vsix     # or: code --install-extension …
```

The editor gets all of this from `anvil check --json`, which elaborates the program without
generating code: about 50 ms for MNIST.

## Examples

| | | |
|---|---|---|
| [`hello.anvil`](examples/hello.anvil) | tensors, index notation, printing, `grad` | instant |
| [`linear_regression.anvil`](examples/linear_regression.anvil) | `param`, `mse`, `sgd`; recovers the hidden weights | 0.2 s |
| [`spirals.anvil`](examples/spirals.anvil) | data built with index notation; a tanh MLP with Adam | 99.5% in < 1 s |
| [`softmax_regression.anvil`](examples/softmax_regression.anvil) | the loss as one index-notation formula (10-714 hw0) | 7.97% test error, 1 s |
| [`mnist.anvil`](examples/mnist.anvil) | MLP, shuffled batches, models | 96.9%, 0.1 s/epoch |
| [`cnn.anvil`](examples/cnn.anvil) | convolution and pooling (`Conv2d`, `max_pool2d` from the prelude) | 97.9% (2 epochs), 0.9 s/epoch |
| [`attention.anvil`](examples/attention.anvil) | batched self-attention with `...` | loss → 0 in 600 steps, 0.5 s |
| [`snake.anvil`](examples/snake.anvil) | a Deep Q-Network that learns Snake, then plays it in the terminal | average score 17 after 300 games (1.9 s) |
| [`vae.anvil`](examples/vae.anvil) | a variational autoencoder (reparameterized `~ normal(mu, sigma)`, KL term); draws the digits it imagines as text | 5 epochs in 2.7 s |
| [`charrnn.anvil`](examples/charrnn.anvil) | a GRU language model trained on this README with backpropagation through time (`static for`), then sampled | 1.06 bits per byte after 3,000 steps (11 s) |
| [`transformer.anvil`](examples/transformer.anvil) | a small GPT: multi-head causal attention, layer norm, MLP, residuals, all index notation | 1.04 bits per byte on the README after 3,000 steps (10 s) |
| [`digits.anvil`](examples/digits.anvil) | a CNN for hand-drawn digits, trained with random shifts and saved for `bin/draw` | 99.0% after 6 epochs |
| [`letters.anvil`](examples/letters.anvil) | a CNN for handwritten letters and digits (EMNIST's 62 classes, 698,000 images), saved for `bin/draw --text` | 86.9% after 3 epochs (2 min) |
| [`diffusion.anvil`](examples/diffusion.anvil) | a denoising diffusion model (DDPM) with classifier-free guidance; draws each digit 0–9 on request | 20 epochs in 44 s |
| [`checkers.anvil`](examples/checkers.anvil) | checkers learned by self-play (TD learning) plus look-ahead search; play it with `bin/checkers` | beats a greedy player 94–6–0 after 4,000 games (14 s) |
| [`named_dims.anvil`](examples/named_dims.anvil) | MNIST read row by row with attention; three classic shape bugs behind `--set BUG=1` (2, 3) that train silently with plain numbers and do not compile with names | 92.9% after 3 epochs (1.5 s) |
| [`dream.anvil`](examples/dream.anvil) | gradients with respect to the input: the digit network's dreams of each digit, and real digits changed by at most 0.2 per pixel until it misreads all ten | 100% fooled, in under a second |
| [`sentiment.anvil`](examples/sentiment.anvil) | a pretrained 12-layer BERT ([`bge_small.anvil`](examples/bge_small.anvil), weights read from its Hugging Face file) fine-tuned to read the sentiment of a sentence (SST-2) | 91.97% after 2 epochs (9 min) |

### Reinforcement learning: Snake

[`examples/snake.anvil`](examples/snake.anvil) is a complete DQN in about 100 lines of code: the game, the
agent, experience replay, and an animated replay of the trained snake. The board is a tensor of
"lifetimes": each body cell stores how many more steps it stays occupied. Moving the snake is
therefore one index-notation statement:

```python
body[i, j] = length if i == ny and j == nx else (body[i, j] if eat > 0 else max(body[i, j] - 1, 0))
```

The learning step is the Bellman update, written as it is on paper:

```python
idx: i32[BATCH] ~ randint(0, count)
a = A[idx]
target = R[idx] + GAMMA * (1 - D[idx]) * detach(max(q(S2[idx]), axis=1))
chosen[n] = q(S[idx])[n, a[n]]
loss = mean((chosen - target) ** 2)
minimize loss over q with adam(lr=1e-3)
```

It reaches an average score of about 17 on a 12×12 board after 300 games, which take under 2
seconds. The agent sees only the 11 classic features (danger straight, left and right, its
heading, and the food's direction), so it eventually traps itself; a wider view of the board is
the natural next step.

### A recurrent network: charrnn

[`examples/charrnn.anvil`](examples/charrnn.anvil) trains a GRU, written out in about 15 lines of Anvil,
to predict the next byte of this README. The loop over time is a `static for`, so the compiler
unrolls the 32 steps and the backward pass runs through all of them:

```python
at: i32[BATCH] ~ randint(0, n - SEQ - 1)
x[b, t] = text[at[b] + t] where t < SEQ + 1           # 32 random windows of the text
h = zeros(BATCH, HIDDEN)
loss = 0.0
static for t in range(SEQ):
    h = net.gru(net.E[x[:, t]], h)
    loss = loss + cross_entropy(net.out(h), x[:, t + 1])
minimize loss / SEQ over net with adam(lr=3e-3)
```

The unrolled program has 1,252 kernels and compiles in about 3 seconds. Training reaches about 1
bit per byte in 11 seconds. (It was 0.8 when this README was 25 KB; at 38 KB it is harder to
memorize.) Then it writes, feeding each sampled byte back in. It mostly memorizes the text, so
what it writes is a remix of this README:

```
model Linear(fan_in, fnel; let dt shape is known at compile time: about 90 GFLOP/s, roughly
fused and vectorized like everything else.

## Testing

  needs 47 KB of scratch space instead one sees. A nater and each that core's
  NEON FMA peak.
```

### A transformer

[`examples/transformer.anvil`](examples/transformer.anvil) is a two-block GPT on the same text. Its
attention is three lines of index notation, with the causal mask added to the scores:

```python
s[b, a, i, j] = sum q[b, i, a, e] * k[b, j, a, e] / sqrt(DH) + (0.0 if j <= i else -1e9)
p = softmax(s)
o[b, i, a, e] = sum p[b, a, i, j] * v[b, j, a, e]
```

It trains in about 10 seconds. For the first 600 steps the loss sits at about 3.2 bits per byte,
which is roughly what predicting each byte from the one before it gives. Then attention starts to
pay off and the loss falls to about 1 bit per byte on the current 38 KB README (0.6 on the 25 KB
version).

### A pretrained transformer: sentiment

[`examples/bge_small.anvil`](examples/bge_small.anvil) is
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), a pretrained 12-layer BERT
with 33 million parameters, written out in about 60 lines of Anvil. Every weight comes from the
model's own `model.safetensors`, by the name PyTorch gives it:

```python
model Layer(number):
    at = "encoder.layer.{number}."
    q = Dense(at + "attention.self.query", D, D)
    ...
model Bert:
    param words: [VOCAB, D] = weight("embeddings.word_embeddings.weight")
    layers = [Layer(number) for number in range(LAYERS)]
```

After all 12 layers, its vector for every token of 16 test sentences is within 3.2·10⁻⁶ of
PyTorch's ([`check_clone.py`](examples/sentiment/check_clone.py), against
[`reference.py`](examples/sentiment/reference.py), Hugging Face's `BertModel` in plain PyTorch).

[`examples/sentiment.anvil`](examples/sentiment.anvil) puts one new layer on top, reading the
first token's vector, and fine-tunes the whole model on SST-2 (the Stanford Sentiment Treebank, as
in GLUE: 67,349 phrases from film reviews, labeled positive or negative):

```bash
python3 examples/sentiment/prepare.py     # link the model from the Hugging Face cache, tokenize SST-2
bin/anvil run examples/sentiment.anvil     # fine-tune, and save examples/sentiment.weights
python3 examples/sentiment/classify.py "not bad at all" "this was not good"
```

```
epoch 1   dev accuracy 91.63%   (270 s)
epoch 2   dev accuracy 91.97%   (536 s)

  positive  99.6%   not bad at all
  negative  99.5%   this was not good
```

prepare.py needs SST-2 in `examples/data/sst2`
([SST-2.zip](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip), 7.4 MB), and the model in the
Hugging Face cache (`huggingface-cli download BAAI/bge-small-en-v1.5`). The tokenizer is BERT's
WordPiece in plain Python ([`tokenizer.py`](examples/sentiment/tokenizer.py)). Most sentences are
short, so they come in three files, padded to 16, 32 and 64 tokens; the training loop is written
once, compiled for each length, and the three share one Adam.

The same fine-tuning in PyTorch 2.14 ([`torch_finetune.py`](examples/sentiment/torch_finetune.py):
the same model, data, batches, schedule and dropout), in milliseconds per step of 32 sentences on
an M3 Max:

| tokens | Anvil, CPU | PyTorch, CPU | Anvil, Metal | PyTorch, MPS |
|---|---|---|---|---|
| 16 | 98 | 201 | 38 | 81 |
| 32 | 167 | 294 | 58 | 88 |
| 64 | 310 | 530 | 94 | 110 |
| an epoch | 263 s | 504 s | 96 s | 176 s |

PyTorch's CPU times are at its best thread count (8; 12, its default, is 3–9% slower). The Metal
times come from a program of just these training steps, which takes about five minutes to compile.
(Without dropout, Metal's loss after those 105 steps equals the CPU's to six digits.) The whole of
sentiment.anvil does not build for Metal yet: clang crashes on the 36 MB of host code generated for
it.

### Draw a digit

```bash
bin/draw
```

`bin/draw` opens a page in your browser where you draw a digit with the mouse or a finger. A
convolutional network written in Anvil reads it as you draw: you see the digit it reads, how sure
it is of each of 0–9, and the 28×28 image it was given.
- **Training.** The first time, `bin/draw` trains the network
  ([`examples/digits.anvil`](examples/digits.anvil), about a minute) and saves it to
  `examples/digits.weights`. Every training digit is moved by up to 2 pixels in a random direction,
  because hand-drawn digits are never placed as exactly as MNIST's. The network reaches 99.0% on
  MNIST's test digits.
- **Your drawing.** The page sends your strokes, not pixels. The server
  ([`examples/draw/server.py`](examples/draw/server.py)) redraws them the way MNIST's digits look:
  scaled to fit a 20×20 box, with MNIST's stroke width, and centered by their center of mass.
  A digit drawn small in a corner reads the same as one drawn large in the middle.
- **Reading it.** The network runs as native code compiled by Anvil (`anvil.function`), with the
  weights the Anvil program saved. Each reading takes about 2 ms.

The tests draw every digit as a person might with a mouse, in 20 wobbly, slanted, stretched,
small and large versions each, and require the network to read at least 90% of them. The fully
trained network reads all of them.

### Write in English

```bash
bin/draw --text
```

The Text page reads handwritten English: write a few words or lines, printing the letters, and it
shows the text it reads, a box around each letter it found, and each letter as the network saw it.
- **The network.** [`examples/letters.anvil`](examples/letters.anvil) trains a CNN
  ([`letters_model.anvil`](examples/letters_model.anvil)) on EMNIST: 698,000 handwritten characters
  in 62 classes (0–9, A–Z, a–z), each turned the right way round and moved by up to 2 pixels
  in the training loop. Three epochs take two minutes and reach 86.9% on EMNIST's test set. That is
  about the best one character alone allows: o, O and 0, or l, I and 1, are often drawn exactly
  alike. (The data, 2.2 GB as floats, is too big for the program's static data section, so tensors
  of 64 MB or more are allocated when the program starts.)
- **Letters, words, lines** ([`examples/draw/text_reader.py`](examples/draw/text_reader.py)). Lines
  come from the tall strokes. Strokes that overlap from left to right make one letter: the dot of
  an i, the bar of a t, the three strokes of an E. Strokes that only touch might be one letter (the
  arches of an m) or two written close together, and the reader tries both. A gap that stands out
  from the gaps between letters separates words. Periods, commas and apostrophes are told apart
  by size and position.
- **Context.** For each word, a beam search tries the ways of dividing it into letters. Each
  division is read three ways: as the dictionary word the network's probabilities fit best
  (`/usr/share/dict/words`, plus suffixes like -s, -ed and -ing, and a bonus for common words), as
  the letters read one by one (for names), and as a number. The most probable reading wins. Alone,
  the network reads `hello world` as `he110 W0r1d`.
- **Case.** EMNIST scales every character to fill its frame, so the network can't tell c from C.
  The reader also uses each letter's height on its line: a letter that rises only as high as the
  small letters is lowercase. The letters of a word vote for CAPITALS, Capitalized or lowercase.

The tests write sentences with the same strokes a mouse would make. In 60 random sentences
(246 words), it reads 100% of words in neat writing, 98.8% in wobbly writing, 98.8% with letters
crowded together, and 88% when both happen at once. A sentence takes 15–30 ms.

### Diffusion

[`examples/diffusion.anvil`](examples/diffusion.anvil) is a denoising diffusion model:
- **Training.** A digit is drowned in a random amount of noise, and a network learns to recover
  it. The network is told the noise level and the digit.
- **Drawing.** It starts from pure noise and steps down through 200 noise levels.
- **Guidance.** The label is hidden 10% of the time during training. When drawing, each step
  then leans away from "any digit" towards the digit asked for (classifier-free guidance).

The noise schedule is two lines of index notation, computed when the program starts:

```python
beta[t] = (1e-4 + (0.02 - 1e-4) * t / (T - 1)) * 1000 / T where t < T
alpha_bar[t] = prod (alpha[s] if s <= t else 1.0) where t < T
```

Training takes 44 seconds on the CPU. Then it draws each digit, two pixels by two to a
character:

```
      ▟██▖               ▟▌           ▟██▖          ▄▄▄▄                   ▄
     ▐████▄             ▗█▘           ▀▀▀█         ▝█▀▀█▙                 ▐█
    ▗██▀▝▜█▖            █▛               █             ▐█            ▗▄   ▟▛
    ▟█▘   ██           ▐█▘              ▗█            ▟██▖           ▟▛  ▟█
   ▟█▘    ██          ▗█▌               ▟█           ▐███▙▖        ▗▟█▌  █▌
  ▗█▛     ██          ▟▛           ▗▄▄▄▄█▛               ▜█       ▟███▌ ▐█▌
  ▐█▘    ▟█▘         ▐█▘          ▗██▀███▘                █       █▛▘   ██
  ▐█   ▗██▘         ▗█▌           ▐█▌▄█▛▀                ▟▌       ▝     █▌
  ▝██▄▟██▘          ▟█            ▝██▛▀   ▗█▘      ▙▖  ▄▟█             ▐█
   ▝▜██▀            █▌                     ▀      ▝█████▛▘             ▐█

                       ▗█▌
                       ▟█▘                             ▗▄▄▄
        ▗▄▟██         ▟█▘         ▄▄▄▄▄               ▟█▀▀█▙          ▟█▌
      ▟██▛▀▜▘        ▗█▘          ▐█▜████▖           ▗█▘  ▐█▌        ▟▛▘█▖
     ▟█▀            ▗█▛                ▝█▌           ▐█  ▗██        ▐█▘ █▌
    ▟█▌             ██ ▗▄▄▄▖            █▌           ▐█▄██▀▘        ▐▌ ▗█▌
    ███▖           ▐█▘▗████▌            █            ▐██▀           ▐▙▟██▘
        ▙▖         ▐█▄█▛ ▟█▌           ▐█           ▗██▌            ▝▀▀ █
  ▐▙   ▐█▌         ▝██████▘            ▟▛          ▗█▛ █               ▐█
  ▐█████▀           ▀███▀▘             █           ▐█ ▗█               ▐▌
   ▀▀▀▀▘                              ▐█           ▐███▘               ▜▌
```

The network predicts the clean image rather than the noise. Predicting 784 numbers of white noise
through a 512-wide hidden layer cannot work, and the first version, which tried, stalled at a loss
no better than guessing.

### Self-play: checkers

[`examples/checkers.anvil`](examples/checkers.anvil) learns checkers from the rules alone. A small
network, the critic, scores positions. It plays thousands of games against itself, and after every
move the value of the position it was in is pulled towards the value of the best move from it.
That is temporal-difference learning, as in TD-Gammon. Arthur Samuel's 1959 checkers player, the
program that gave machine learning its name, learned in a similar way.

The rules are index notation over the 32 dark squares. The board is always seen from the side to
move, so one set of rules and one network play both colors. `NEXT[s, d]` and `JUMP[s, d]` are the
squares one and two steps along each diagonal:

```python
mine[s, d] = b[s] > 0 and (d < 2 or b[s] == 2) and (forced < 0 or s == forced) where d < 4    # men only go up
jumps[s, d] = mine[s, d] and b[NEXT[s, d]] < 0 and b[JUMP[s, d]] == 0
capture = sum(jumps) > 0                                       # captures are compulsory
legal[s, d] = jumps[s, d] if capture else mine[s, d] and b[NEXT[s, d]] == 0
src, dir = nonzero(legal, size=MOVES)                          # the legal moves, as a list
```

Every legal move's resulting board is built at once and turned around for the opponent
(`next[m, t] = -after[m, 31 - t]`), and the critic scores them all in one batch. Training looks one
move ahead. Playing looks further, with a search written as recursion that the compiler unrolls,
because `depth` is a constant:

```python
fn search(valid, next, next_forced, again, depth):
    if depth == 1:
        return evaluate(valid, next, next_forced, again)
    value: [MOVES] = -1.0
    for k in range(i32(sum(valid))):
        valid2, next2, forced2, again2 = moves(next[k], next_forced[k])
        reply = max(max(search(valid2, next2, forced2, again2, depth - 1)), -1.0)
        value[k] = reply if again[k] > 0 else -reply
    return where(valid, value, -inf)
```

Every few hundred games it plays 100 games against a random player and 100 against a greedy one.
The greedy player takes the most material it can and avoids moves that let the reply take some back:

```
$ bin/anvil run examples/checkers.anvil
  250 games   vs random:  98 won   1 drawn   1 lost   vs greedy:  32 won  50 drawn  18 lost   (1.5s)
  500 games   vs random:  99 won   1 drawn   0 lost   vs greedy:  32 won  47 drawn  21 lost   (2.9s)
 1000 games   vs random: 100 won   0 drawn   0 lost   vs greedy:  78 won  21 drawn   1 lost   (4.9s)
 2000 games   vs random: 100 won   0 drawn   0 lost   vs greedy:  88 won  12 drawn   0 lost   (8.1s)
 4000 games   vs random: 100 won   0 drawn   0 lost   vs greedy:  94 won   6 drawn   0 lost   (13.7s)
```

Then it plays one game against the greedy player on screen. Looking ahead matters: against the
greedy player, the fully trained critic wins 58, draws 39 and loses 3 of 100 games when it looks one
move ahead. It wins 87–13–0 at two moves, 94–6–0 at three and 96–4–0 at four. Besides the pieces,
the critic sees two attack maps: which pieces can be captured right now, for each side. In an
earlier version without them, the critic looked one move ahead, beat the random player, and lost
most of its games to the greedy one.

**Play it yourself** in a terminal:

```bash
bin/checkers
```

You are red and move first, choosing moves by number from a list like
`1: a3-b4  2: c3-b4  3: c3-d4`. The critic answers looking four moves ahead.
`bin/checkers --set PLAY_DEPTH=5` makes it stronger. The first time, it learns for about 15
seconds and saves the critic to `examples/checkers.weights`. After that, the game starts right
away. To train a fresh critic, delete that file or run the example again, which always trains
and saves. `bin/checkers` is just the example with `--set PLAY=1 --set WATCH=0`. `--set` gives
any `const` in a program a new value without editing the file.

## Performance

Everything here is measured on an Apple M3 Max. [`benchmarks/run.py`](benchmarks/run.py) produces
[`benchmarks/results.md`](benchmarks/results.md), and [`experiments/run.py`](experiments/run.py)
produces [`experiments/results.md`](experiments/results.md). Both files record the load average of
the run that made them. The table below was measured while other programs kept the machine busy
(load average about 16), which slows both sides down.

| Anvil against NumPy + Accelerate | Anvil | NumPy | |
|---|---|---|---|
| MNIST MLP training epoch (784-128-10, batch 64, SGD) | 0.066 s | 0.121 s | **1.8×** |
| matmul 1024³ | 0.94 ms (2,276 GFLOP/s) | 1.21 ms (1,775 GFLOP/s) | **1.3×** |
| `gelu(x @ W + b)`, 256×1024 → 1024 | 0.52 ms | 0.96 ms | **1.9×** |
| softmax over rows, 4096×1024 | 1.00 ms | 10.5 ms | **10×** |
| layer norm, 4096×1024 | 0.58 ms | 7.0 ms | **12×** |
| causal attention, 16×256×64 | 1.27 ms | 4.4 ms | **3.5×** |
| `gelu(1.01x + 0.1) · sigmoid(x)` on 4M floats | 5.6 ms | 17.4 ms | **3.1×** |

- **Fusion.** Anvil wins wherever the time goes to memory traffic and transcendental functions: it
  fuses each of these computations into one or two multithreaded passes. NumPy makes a pass per
  operation, mostly on one core.
- **Matrix products.** These go to Apple's AMX matrix coprocessor through Accelerate's
  `cblas_sgemm`, as NumPy's do, but with the epilogue (bias, activation, optimizer update) fused
  into Anvil's own kernels around it. Stacks of products (attention heads) and sums of products (a
  convolution's weight gradient) go there too. `ANVIL_BLAS=0` keeps every product in Anvil's own NEON
  kernels, which reach about 600 GFLOP/s on 8 cores; with Accelerate, a 1024³ product runs at about
  2,300 GFLOP/s.
- **Whole training steps.** Each step is a few fused kernels on static buffers, with no
  per-operation overhead.
- **The GPU.** Bigger models run faster still with `--metal` ([above](#on-the-gpu-metal)).

**What the optimizations are worth**, in seconds per MNIST epoch (from the ablation experiment,
before matrix products went to the AMX coprocessor):

| | MLP 784-128-10 | MLP 784-1024-1024-10 | LeNet CNN |
|---|---|---|---|
| everything on | 0.096 s | 1.98 s | 0.85 s |
| one thread | 0.28 s (2.9×) | 10.95 s (5.5×) | 3.20 s (3.8×) |
| no IR optimizations (`-O0`) | 0.137 s (1.4×) | 2.05 s (1.0×) | 2.35 s (2.8×) |
| `-O0`, one thread | 0.32 s (3.3×) | 10.23 s (5.2×) | 8.15 s (9.6×) |

- Fusion matters most where there are many small kernels, as in the CNN and the small MLP.
- In the wide MLP almost all the time is in large matrix multiplies, so threads (5.5×) are what
  count there.
- The experiments also found three performance bugs, now fixed:
  - In a fused optimizer update, the old weights were loaded before the matmul's loop, so the
    kernel fell back to a smaller register tile.
  - `relu(z)` was inlined into a matmul operand and recomputed once per 16 output columns.
  - Small but expensive kernels (the Adam update of a 784×128 matrix, the copy of a shuffled
    batch) ran on one thread.

  Together these cut the wide MLP from 2.23 to 1.98 s/epoch and Adam training at width 128 from
  0.150 to 0.100 s/epoch.
- **Row-tiled dot products.** The wide MLP's backward pass computes `dz @ Wᵀ`, where both operands
  are contiguous along the sum, so each output is a dot product. That schedule now computes 4
  outputs at once and shares the loads of `W`. The kernel went from 496 to 293 ms per epoch, the
  wide MLP to 1.78 s/epoch, and the transformer and char-RNN train about 10% faster.

**Hyperparameter sweeps are cheap.** At about 0.1 s per epoch, the two grids in the experiments (27
MNIST models, 3 epochs each, every one compiled separately with `--set` on
[`experiments/mnist_sweep.anvil`](experiments/mnist_sweep.anvil)) take under a minute:
- Adam at lr 1e-3 is best after 3 epochs (97.47%), just ahead of RMSProp (97.37%) and SGD at lr 0.3
  (97.21%).
- Widening one hidden layer from 32 to 1,024 units takes accuracy from 95.3% to 97.8%.
- A second hidden layer helps only up to 128 units.

Compiling from scratch takes 0.3 s for MNIST and about 0.9 s for the transformer and checkers.

**Against PyTorch.** [`benchmarks/vs_pytorch/`](benchmarks/vs_pytorch) holds nine small projects
written in both Anvil and PyTorch 2.14, with the same models, data and hyperparameters, and checked to
reach the same accuracy. The [report](benchmarks/vs_pytorch/REPORT.md) has every number. Speed-ups
of Anvil are geometric means over the nine:

| | Anvil is faster by |
|---|---|
| CPU, each at its defaults | 6.5× |
| CPU, PyTorch at its best thread count | 3.6× |
| CPU, one thread each | 2.3× |
| CPU, against `torch.compile` | 7.9× |
| GPU, Metal against PyTorch's MPS | 2.2× (but MPS is 5× faster on the larger CNN) |

Small models gain the most (a softmax-regression step takes 26 µs against 158–374 µs), large matrix
products the least (both use the AMX unit), and Anvil's Metal backend still loses to MPS on
convolutions.

## Testing

```bash
python3 -m pytest tests
```

There are 334 tests:
- Every autodiff rule is checked against float64 finite differences.
- Optimized programs must match unoptimized ones exactly.
- Native code must match the NumPy reference interpreter on tricky shapes (vector tails, gathers,
  strided views, every math function, integer ops, control flow, a full training loop).
- The error messages and their source spans are checked.
- The examples run, and MNIST must reach at least 96%.
- The checkers rules match an independent Python implementation of the game on every position of
  150 random games.
- Multithreaded results must be identical for 1, 3, 8, and 16 threads.
- Compiling the same program gives the same assembly in every process, whatever Python's hash seed.
- The CUDA output of the examples, compiled as C++, must match the interpreter. clang's CUDA front
  end must accept every example's `.cu`.
- `anvil.function` results must match NumPy: gradients, gathers, tuples, constants, and calls from
  several threads.
- The editor extension runs against a stand-in for the VS Code API. It must report errors at the
  right lines, and its hovers and shape hints must be right.
- The examples on the GPU (Metal) must match the interpreter. Inside a training loop, only a print
  may make the CPU wait for the GPU.
- Matrix products through Accelerate (every transpose, stacks and sums of products, epilogues) must
  match the interpreter.
- `--check` must find the first NaN and the line that made it. `anvil export`'s libraries must
  give the same results when called from C. A real notebook with `%%anvil` cells must run.
- Pretrained models: tensors read from a `.safetensors` file must match NumPy; lists of layers must
  train, save and load; two `minimize` statements must continue one Adam; `erf` must match
  `math.erf`; the command line must optimize; programs too large for a static segment must give the
  same results with their temporaries on the heap.
- Named dimensions: equal sizes with different names must not compile at any of the places two
  dimensions meet; the names must survive slices, reshapes, index notation and layers; hovers,
  call signatures and `anvil shapes` must show them.
- A gradient through a run-time loop or branch must be refused (it used to come out silently
  wrong), and the rewrites the error suggests must give exact gradients.
- Every math function must match the interpreter on the GPU at extreme inputs. A checkpoint must
  refuse to load into tensors of other shapes, and old checkpoints must still load.

## Layout

```
anvil/            compiler: lexer, parser, elaborate (+ ops, builtins), dims (named dimensions),
                autodiff, optimize, interp (reference semantics), prelude.anvil (standard library),
                cli, pyapi (anvil.function), export (C libraries), jupyter (%%anvil),
                ide (editors, anvil shapes), cost (anvil cost)
anvil/backend/    AArch64: kernel lowering, vecmath (inline exp/log/tanh/sin/cos),
                mir (register allocation), aarch64 (program codegen), runtime.s,
                blas (matrix products on the AMX coprocessor);
                Metal: metal.py, anvil_metal.h; CUDA: cuda.py, anvil_cuda.h; anvil_host.h (shared)
examples/       programs, MNIST data, examples/errors/, examples/draw/ (bin/draw),
                examples/sentiment/ (BERT's tokenizer, SST-2 preparation, the PyTorch reference)
benchmarks/     Anvil against NumPy (run.py → results.md); vs_pytorch/: nine projects in Anvil and
                PyTorch, and REPORT.md
bin/            anvil, draw, checkers
experiments/    ablations and MNIST sweeps (run.py → results.md)
editors/vscode/ syntax highlighting for VS Code and Cursor
tests/          pytest suite
docs/PLAN.md    design document and roadmap
docs/EPIC.md    a ranked plan of where to go next (docs/BRAINSTORM.md: the 270 ideas behind it)
docs/index.html the website: examples, named dimensions and the benchmarks (GitHub Pages: docs/)
```

## Status

It covers the language in the design document, with four backends: native multithreaded
AArch64/macOS (with the AMX coprocessor for matrix products), the Apple GPU (Metal), CUDA (tested in
emulation; not yet on an NVIDIA GPU), and the NumPy interpreter. Dimensions carry names, and
mismatched names are compile errors. Anvil functions can be called from Python and Jupyter, and
exported as C libraries. Not done yet: Linux/x86 backends, differentiation through run-time loops
(`static for` unrolls instead; a gradient through a run-time loop is a compile error), and dynamic
shapes. See the [roadmap](docs/PLAN.md#8-roadmap), [the list of ideas](docs/IDEAS.md), and
[where to go next](docs/EPIC.md).

## About this repository's history

This repository doubles as a classroom example of how git records history. Its commit dates were
set with `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` to spread the project's development over six
months; they are not the dates the code was written.
