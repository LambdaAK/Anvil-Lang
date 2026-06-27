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
- **Shapes are types.** Every shape is known at compile time, and shape errors point at the exact
  operand.
- **Differentiation is built in.** `grad(loss, W)` and `minimize loss with adam()` generate the
  backward pass *at compile time*, so it gets fused and vectorized like everything else.
  The SGD update ends up inside the weight-gradient matrix multiply, and the gradient is never
  stored to memory.
- **Readable assembly.** Each kernel becomes one function, headed by the math it computes.
- **Multithreaded.** Large kernels split their outermost loop across cores. Results are
  bit-identical for any number of threads.
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
`LayerNorm(d)` and `Embedding(n, d)`.

```python
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

`use "model.anvil"` brings in another file's functions, models, optimizers and constants.

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
out-of-bounds windows, a non-scalar loss, and typos (which get "did you mean `softmax`?"). The
compiler reports every independent error in a file, up to 20, not just the first. A name whose
definition failed is not reported again where it is used.

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

## Testing

```bash
python3 -m pytest tests
```

There are 289 tests:
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

## Layout

```
anvil/            compiler: lexer, parser, elaborate (+ ops, builtins), autodiff, optimize,
                interp (reference semantics), prelude.anvil (standard library), cli,
                pyapi (anvil.function), export (C libraries), jupyter (%%anvil), ide (editors),
                cost (anvil cost)
anvil/backend/    AArch64: kernel lowering, vecmath (inline exp/log/tanh/sin/cos),
                mir (register allocation), aarch64 (program codegen), runtime.s,
                blas (matrix products on the AMX coprocessor);
                Metal: metal.py, anvil_metal.h; CUDA: cuda.py, anvil_cuda.h; anvil_host.h (shared)
examples/       programs, MNIST data, examples/errors/
benchmarks/     Anvil against NumPy (run.py → results.md)
experiments/    ablations and MNIST sweeps (run.py → results.md)
editors/vscode/ syntax highlighting for VS Code and Cursor
tests/          pytest suite
docs/PLAN.md    design document and roadmap
```
