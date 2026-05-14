"""Native AArch64 code vs. the reference interpreter."""
import pytest

from util import check_native, run_native

SHAPES = [1, 3, 4, 5, 8, 17, 33]


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("m", [1, 6, 19])
def test_elementwise_tails(n, m):
    check_native(f"""
x: [{m}, {n}] ~ normal(0, 1)
y: [{n}] ~ uniform(-1, 1)
z = relu(x * 2 + y) - 0.5 * abs(x) + sqrt(abs(y) + 1)
print(z)
""")


@pytest.mark.parametrize("n,k,m", [(1, 1, 1), (3, 5, 7), (4, 16, 16), (5, 9, 33), (17, 33, 20), (2, 64, 3)])
def test_matmul_shapes(n, k, m):
    check_native(f"""
a: [{n}, {k}] ~ normal(0, 1)
b: [{k}, {m}] ~ normal(0, 1)
c = a @ b
print(c)
print(a @ b.T.T)
print(sum(c), mean(c, axis=1))
""")


@pytest.mark.parametrize("n", [1, 7, 16, 37])
def test_reductions(n):
    check_native(f"""
x: [3, {n}] ~ normal(0, 1)
print(max(x, axis=1), min(x, axis=1), sum(x, axis=1), mean(x, axis=0))
print(argmax(x), argmin(x, axis=1))
m[i] = max x[i, j]
s[j] = sum x[i, j] * x[i, j]
print(m, s, sum(x * x))
""")


def test_math_functions():
    check_native("""
x = linspace(-6, 6, 41)
print(exp(x))
print(tanh(x))
print(sigmoid(x))
print(sin(x), cos(x))
print(log(abs(x) + 0.001))
print(floor(x * 0.7), ceil(x * 0.7), round(x * 0.7))
print(x ** 2, abs(x) ** 0.5, abs(x) ** 1.3, sign(x))
print(gelu(x), softplus(x), silu(x), elu(x))
""")


def test_integer_ops():
    check_native("""
a = arange(-7, 9)
print(a // 3, a % 3, a * a - 4, -a, abs(a))
print(f32(a) / 2, i32(f32(a) * 1.7))
print(a > 0, a == 2, (a > 0) and (a < 5), not (a > 0))
""")


def test_softmax_cross_entropy():
    check_native("""
z: [5, 10] ~ normal(0, 3)
y = [1, 0, 9, 3, 3]
print(softmax(z))
print(log_softmax(z))
print(cross_entropy(z, y), accuracy(z, y))
print(onehot(y, 10))
""")


def test_gather_and_views():
    check_native("""
table: [10, 6] ~ normal(0, 1)
ids = [3, 1, 4, 1, 5, 9, 2, 6]
e = table[ids]
print(e)
f[i, j] = table[ids[i], j] * 2
print(f)
print(table[2:7:2, 1:5].T)
print(reshape(table, 6, 10)[1])
""")


def test_control_flow():
    check_native("""
total = 0.0
i = 0
while total < 100:
    total = total + i * 1.5
    i = i + 1
    if i % 3 == 0:
        continue
    if total > 80:
        break
print("{i} {total:.2f}")
acc = 0
for k in range(10):
    if k == 7:
        break
    acc = acc + k
print(acc)
""")


def test_conv_and_pool():
    check_native("""
img: [2, 3, 9, 9] ~ normal(0, 1)
k: [4, 3, 3, 3] ~ normal(0, 0.3)
out[n, o, y, x] = sum k[o, c, u, v] * img[n, c, y + u, x + v]
print(out[1, 2])
pool[n, c, i, j] = max out[n, c, 2*i + u, 2*j + v] where u < 2, v < 2
print(pool[0, 0])
""")


def test_print_formats():
    check_native("""
x = 3.14159
n = 42
v = [1.5, -2.25]
print("pi={x:.2f} e={x:.3e} g={x} n={n} n3={n:5d} pct={0.1234:.1%} {v}")
big: [40, 40] ~ normal(0, 1)
print(big)
print("done", end="!\\n")
""")


def test_training_mlp():
    check_native("""
x: [128, 4] ~ normal(0, 1)
labels = i32(argmax(x @ [[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]]))
model Net:
    l1 = Linear(4, 32)
    l2 = Linear(32, 3)
    fn forward(x) = x |> l1 |> relu |> l2
net = Net()
for epoch in range(20):
    for xb, yb in batches(x, labels, size=32, shuffle=true):
        loss = cross_entropy(net(xb), yb)
        minimize loss with adam(lr=0.01)
    if epoch % 5 == 4:
        print("epoch {epoch} loss {loss:.4f} acc {accuracy(net(x), labels):.3f}")
""")


def test_runtime_bounds_check():
    out, err, code, _ = run_native("""
x = arange(10)
for i in range(12):
    s = x[i]
    print(s)
""")
    assert code != 0
    assert "out of bounds" in err


def test_invariant_integer_division_vectorizes():
    """`k // 2` and `k % 2` do not vary along the vectorized index, so the kernel still vectorizes."""
    from anvil.backend.aarch64 import generate
    from util import compile_text
    src = """
b: i32[3, 32] ~ randint(-2, 3)
x[n, k, t] = f32(b[n, t] == (1 + k % 2) * (1 - 2 * (k // 2))) where k < 4
print(x)
"""
    check_native(src)
    asm, _ = generate(compile_text(src).program)
    header = [line for line in asm.splitlines() if "loops: n<3 × k<4 × t<32" in line]
    assert header and "vectorized" in header[0], header


def test_loops_collapse():
    """Adjacent loops that every access sees as one index merge into a single, longer vector loop."""
    from anvil.backend.aarch64 import generate
    from util import compile_text
    src = """
x: [6, 8, 5] ~ normal(0, 1)
param W: [8, 5] ~ normal(0, 1)
y[n, c, v] = x[n, c, v] * W[c, v] + 1
print(sum(y))
"""
    check_native(src)
    asm, _ = generate(compile_text(src).program)
    assert "loops: ncv<240" in asm or "loops: n<6 × cv<40" in asm, [l for l in asm.splitlines() if "loops:" in l]


def test_fused_update_keeps_the_big_register_tile(no_blas):
    """A weight gradient with the SGD update fused into it is still a 4×16 register-tiled matrix
    multiply: the old weights are loaded after the reduction loop, not before it (where they held
    16 vector registers through the loop), and `relu(z)`, its left operand, is computed once
    instead of once per 16 output columns."""
    from anvil.backend.aarch64 import generate
    from util import compile_text
    src = """
x: [64, 48] ~ normal(0, 1)
y: i32[64] ~ randint(0, 10)
param W1: [48, 64] ~ normal(0, 0.1)
param W2: [64, 64] ~ normal(0, 0.1)
param W3: [64, 10] ~ normal(0, 0.1)
for step in range(3):
    h = relu(relu(x @ W1) @ W2)
    loss = cross_entropy(h @ W3, y)
    minimize loss with sgd(lr=0.1)
    print(loss)
"""
    check_native(src)
    asm, _ = generate(compile_text(src).program)
    headers = [line for line in asm.splitlines() if "loops: k<64 × j<64  reduce i<64" in line]
    assert headers and all("4×16 register tile" in h for h in headers), headers


def test_expensive_kernels_are_threaded_sooner(no_blas):
    """Threads pay off by work, not by iteration count: the Adam update of a 784×128 matrix
    (100k iterations, each with four loads, three stores, a square root and two divisions) runs
    in parallel, though a plain update of that size does not."""
    from anvil import ir
    from anvil.backend.aarch64 import parallel_grain
    from util import compile_text
    adam = """
x: [64, 784] ~ normal(0, 1)
y: i32[64] ~ randint(0, 10)
param W: [784, 128] ~ normal(0, 0.05)
param V: [128, 10] ~ normal(0, 0.05)
for step in range(2):
    loss = cross_entropy(relu(x @ W) @ V, y)
    minimize loss with adam(lr=1e-3)
    print(loss)
"""
    plain = """
param W: [784, 128] ~ normal(0, 0.05)
for step in range(2):
    W = W + 1.0
print(sum(W))
"""
    check_native(adam)
    check_native(plain)

    def update_kernels(src):
        return [k for k in ir.all_kernels(compile_text(src).program)
                if k.red is None and ir.prod(v.extent for v in k.domain) == 784 * 128 and not k.uses_rand()]
    heavy, light = update_kernels(adam), update_kernels(plain)
    assert heavy and all(parallel_grain(k) is not None for k in heavy)
    assert light and all(parallel_grain(k) is None for k in light)


@pytest.mark.parametrize("n, m, r", [(64, 33, 100), (7, 5, 17), (13, 9, 3), (4, 16, 64)])
def test_row_tiled_dot_products(n, m, r, no_blas):
    """`a @ bᵀ`-shaped kernels (both operands contiguous along the reduction) compute 4 rows of
    dot products at once, sharing the loads of b; leftover rows and lanes are handled."""
    from anvil.backend.aarch64 import generate
    from util import compile_text
    src = f"""
a: [{n}, {r}] ~ normal(0, 1)
b: [{m}, {r}] ~ normal(0, 1)
d[i, k] = sum a[i, j] * b[k, j]
e[i, k] = max (a[i, j] - b[k, j])
print(d)
print(e)
"""
    check_native(src)
    asm, _ = generate(compile_text(src).program)
    if n >= 4 and r >= 8:
        assert any("4 rows ×" in line for line in asm.splitlines() if "loops:" in line), \
            [line for line in asm.splitlines() if "loops:" in line]


def run_checked(src, check="nan"):
    """Build with --check (no fusion: one kernel per operation) and run."""
    import os
    import subprocess
    import tempfile
    from anvil.backend.aarch64 import generate
    from anvil.backend.toolchain import assemble_and_link
    from util import compile_text
    asm, _ = generate(compile_text(src, path="checked.anvil", optimize=False).program, check=check)
    with tempfile.TemporaryDirectory() as d:
        s, exe = os.path.join(d, "p.s"), os.path.join(d, "p")
        with open(s, "w") as f:
            f.write(asm)
        assemble_and_link(s, exe)
        return subprocess.run([exe], capture_output=True, text=True)


def test_check_finds_the_first_nan():
    r = run_checked("x: [4, 3] ~ normal(0, 1)\nw = exp(x)\ny = log(x - 0.5) * w\nprint(sum(y))\n")
    assert r.returncode == 1 and r.stdout == ""
    assert "nan in `t3` (f32[4, 3]) at [0, 0]" in r.stderr
    assert "made by `log` at checked.anvil:3:1" in r.stderr and "y = log(x - 0.5) * w" in r.stderr
    assert "this computation made it" in r.stderr


def test_check_traces_infinities():
    src = "x: [4, 3] ~ normal(0, 1)\nz = log(x * 0)\nr = z - z\nprint(sum(r))\n"
    r = run_checked(src)                               # the nan appears in `z - z`, from infinities
    assert "nan in" in r.stderr and "its input `z` already held 0 nan and 12 infinite values" in r.stderr
    assert "--check=inf" in r.stderr
    r = run_checked(src, "inf")                        # … which --check=inf traces to the log
    assert "-inf in `z` (f32[4, 3]) at [0, 0]" in r.stderr and "`log`" in r.stderr


def test_check_passes_a_healthy_program():
    src = """
x: [64, 8] ~ normal(0, 1)
y: [64] ~ normal(0, 1)
param w: [8] ~ normal(0, 1)
for step in range(20):
    loss = mean((x @ w - y) ** 2)
    minimize loss with adam(lr=0.01)
print(loss)
"""
    r = run_checked(src, "inf")
    assert r.returncode == 0 and r.stderr == "", r.stderr


def test_large_tensors_live_on_the_heap(monkeypatch):
    """Tensors of 64 MB or more are allocated when the program starts (a static zero section of
    gigabytes can fail to map: EMNIST's 700k images are 2.2 GB); here the limit is lowered so a
    small training program keeps its data, parameters, gradients and Adam's moments on the heap."""
    import anvil.backend.aarch64 as a64
    from util import compile_text
    monkeypatch.setattr(a64, "HEAP_MIN", 1024)
    src = """
x: [64, 48] ~ normal(0, 1)
y: i32[64] ~ randint(0, 10)
param W1: [48, 32] ~ normal(0, 0.1)
param W2: [32, 10] ~ normal(0, 0.1)
for step in range(5):
    loss = cross_entropy(relu(x @ W1) @ W2, y)
    minimize loss with adam(lr=0.01)
    print(loss)
print(W1[0, 0:4], W2[3, 0:4])
"""
    check_native(src)
    asm, _ = a64.generate(compile_text(src).program)
    assert "_anvil_alloc_heap:" in asm and asm.count("bl _calloc") >= 4
