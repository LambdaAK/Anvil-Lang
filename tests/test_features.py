"""Language features from the README, run natively and checked against the interpreter."""
import pytest

from util import check_native

PROGRAMS = {
    "maximize_over_submodel": """
model Net:
    l1 = Linear(3, 4)
    l2 = Linear(4, 1)
    fn forward(x) = x |> l1 |> tanh |> l2
net = Net()
x: [16, 3] ~ normal(0, 1)
before = copy(net.l1.W)
for step in range(5):
    score = -mean(net(x) ** 2)
    maximize score over net.l2 with adam(lr=0.05)
print("l1 frozen:", sum(abs(net.l1.W - before)))
print("score", -mean(net(x) ** 2))
""",
    "reparameterization": """
param mu: [4]
param log_sigma: [4]
target = [1.0, -2.0, 0.5, 3.0]
for step in range(300):
    eps: [256, 4] ~ normal(0, 1)
    z = mu + exp(log_sigma) * eps
    loss = mean((z - target) ** 2)
    minimize loss with adam(lr=0.05)
print(mu, exp(log_sigma))
""",
    "batched_attention": """
fn attention(x: [b, t, dm], Wq: [dm, dh], Wk: [dm, dh], Wv: [dm, dh]) -> [b, t, dh]:
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    s[..., i, j] = sum q[..., i, c] * k[..., j, c] / sqrt(dh)
    out[..., i, c] = sum softmax(s)[..., i, j] * v[..., j, c]
    return out
x: [2, 5, 6] ~ normal(0, 1)
W: [6, 3] ~ normal(0, 0.5)
print(attention(x, W, W, W))
""",
    "greek_ufcs_while_pipe": """
θ: [3] ~ normal(0, 1)
η = 0.5
n = 0
while sum(θ * θ) > 0.01:
    θ = θ * (1 - η)
    n = n + 1
print(n, θ.sum(), θ.reshape(1, 3), θ |> softmax)
""",
    "dropout": """
h: [4, 8] ~ normal(0, 1)
keep: [4, 8] ~ bernoulli(0.5)
print(h * keep / 0.5)
""",
    "grad_wrt_input_fgsm": """
param W: [4, 3] ~ normal(0, 1)
x0: [5, 4] ~ normal(0, 1)
labels = [0, 2, 1, 1, 0]
x = copy(x0)
loss = cross_entropy(x @ W, labels)
g = grad(loss, x)
x_adv = x + 0.1 * sign(g)
print(loss, cross_entropy(x_adv @ W, labels))
""",
    "if_merge_and_seed": """
seed(7)
a: [3] ~ uniform(0, 1)
total = 0.0
for i in range(6):
    if i % 2 == 0:
        total = total + sum(a) * i
    else:
        total = total - 1
print(total, a)
""",
    "momentum_layernorm": """
x: [32, 8] ~ normal(0, 1)
y: [32] ~ normal(0, 1)
param W: [8] ~ normal(0, 0.3)
for step in range(50):
    loss = mse(layer_norm(x) @ W, y)
    minimize loss with sgd(lr=0.05, momentum=0.9)
print(loss)
""",
    "item_assignment": """
x = zeros(5)
x[2] = 7.0
x[0:2] += 1
y = x
x[4] = -1.0
print(x, y)
S: [4, 3] = 0
for t in range(6):
    S[t % 4] = [f32(t), f32(t * t), -1.0]
print(S)
counts: i32[3] = 0
for t in range(10):
    counts[t % 3] += 1
print(counts)
""",
    "runtime_literals_randint_show": """
a = 3
b = 2.5
print([a, b, a * b, 1], [[a, 0], [0, a + 1]])
idx: i32[12] ~ randint(0, 5)
k: i32 ~ randint(10, 13)
print(idx, k)
board[i, j] = i32(i == j) + 2 * i32(i + j == 3) where i < 4, j < 4
show(board, ".#@")
show(board, ["  ", "██", "<>"])
""",
    "nonzero_and_stack": """
x = [0.0, 3.0, 0.0, -1.0, 2.0, 0.0]
print(nonzero(x, size=4)[0], nonzero(x, size=2)[0])
m[i, j] = (i + j) % 3 == 0 where i < 3, j < 4
r, c = nonzero(m, size=6, fill_value=-1)
print(r, c)
k: i32[5] = [0, 7, 0, 0, 1]
print(nonzero(k, size=3)[0])
for t in range(2):
    y[i] = i % 2 == t where i < 5
    print(nonzero(y, size=4)[0])
a = [1.0, 2.0, 3.0]
B[n, t] = n * 10 + t where n < 2, t < 3
print(stack([a, a * 2]), stack([a, a * 2], axis=1), stack([a, a * 2], axis=-1).shape)
print(flatten(stack([B == 1, B > 10, B * 2], axis=-2), -2))
""",
    "string_lists": """
NAMES = ["zero", "one", "two"]
FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
fn square_name(t) = "{FILES[t % 8]}{t // 8 + 1}"
for k in range(-1, 4):
    print("{k}: {NAMES[k]}, square {square_name(k + 9)}")
print(NAMES[2], square_name(63))
""",
    "scatter_add": """
labels: i32[1000] ~ randint(0, 10)
pred: i32[1000] ~ randint(0, 10)
counts: i32[10] = 0
counts[labels[i]] += 1
conf: i32[10, 10] = 0
conf[labels[i], pred[i]] += 1
print(counts, sum(conf), conf[3])
ids = [2, 0, 2, 1]
g[n, d] = n * 10 + d where n < 4, d < 6
E: [3, 6] = 0
E[ids[n], d] += g[n, d]
print(E)
y = [1.0, 2.0, 3.0]
x[i, j] = i + j where i < 3, j < 5
y[i] -= sum x[i, j]
z = zeros(5)
w = z
z[2 * k] += 1 where k < 3
v = [1.0, 2.0, 3.0, 4.0]
v[i] += v[3 - i]
print(y, z, w, v)
for t in range(3):
    counts[labels[t * 7 + i]] += 2 where i < 5
print(sum(counts))
""",
    "static_for_and_windows": """
data[k] = k * 10 where k < 50
starts = [0, 7, 42]
x[b, t] = data[starts[b] + t] where t < 8
y[b, t] = data[starts[b] - t + 7] where t < 8
z[b, t] = data[2 * t + starts[b] + 1] where t < 4
print(x, y[2], z)
static = 3
total = 0
static for k in [1, 2, 3]:
    total = total + k * k
rows = 0.0
M: [3, 2] ~ normal(0, 1)
static for r in M:
    rows = rows + sum(r)
xs: [5, 4] ~ normal(0, 1)
param W: [4, 4] ~ normal(0, 0.5)
for step in range(30):
    h = zeros(4)
    static for t in range(5):
        h = tanh(xs[t] @ W + h)
    loss = mean((h - 0.5) ** 2)
    minimize loss with adam(lr=0.05)
print(total, static, rows - sum(M), loss, "ab" * 3 + "-" * 2)
""",
    "local_names_shadow_outer_ones": """
fn ramp(n):
    next[m] = 2 * m where m < 4          # `next` and `m` also exist outside: still a definition
    return next
next = [9, 9, 9, 9]
m = 1
print(ramp(3), next, m)
next[m] = 5                           # here `next` is a local variable and m has a value
print(next)
""",
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_feature(name):
    check_native(PROGRAMS[name])


def test_input():
    src = """
NAMES = ["zero", "one", "two"]
total = 0.0
for i in range(20):
    x = input("number {i + 1}? ")
    if x != x:
        print("(not a number)")
    else:
        total = total + x
        print("got {x:g}, total {total:g}, {NAMES[i32(x)]}")
print("not reached")
"""
    lines = ["5", "abc 12xyz", "hello", "-3.5e1", "+.25", "info 7", "0x10", "", "1e3", "-0X5", "2"]
    out = check_native(src, stdin="\n".join(lines) + "\n")
    assert "got 12, total 17" in out                # the first number on the line
    assert out.count("(not a number)") == 2         # "hello" and the empty line
    assert "got 7, total" in out                    # "info" is not a number (nor inf)
    assert "got 0, total" in out                    # decimal only: "0x10" reads as 0
    assert "got 2, total 991.25, two" in out
    assert out.rstrip().endswith("number 12?")      # the input ended: so did the program
    assert "not reached" not in out


def test_checkpoints_check_shapes(tmp_path):
    """A checkpoint loads only into tensors of the same shapes in the same places. Version 1 files
    stored element counts only, so a transposed weight, or two same-shaped layers declared in the
    other order, loaded silently into the wrong places. Renaming the model's variable is fine, and
    version 1 files still load (checked by count)."""
    import struct

    import numpy as np

    from util import check_cuda, check_metal
    f = tmp_path / "w.weights"
    src = f"""
model A:
    param W: [4, 8] ~ normal(0, 1)
model B:
    param W: [8, 4] ~ normal(0, 1)
model P:
    l1 = Linear(4, 4)
    l2 = Linear(4, 4)
model Q:
    l2 = Linear(4, 4)
    l1 = Linear(4, 4)
a = A()
save(a, "{f}")
print("transposed", load(B(), "{f}"), "same", load(A(), "{f}"))
p = P()
save(p, "{f}")
renamed = P()
print("swapped", load(Q(), "{f}"), "renamed", load(renamed, "{f}"), sum(abs(renamed.l2.W - p.l2.W)))
"""
    for check in (check_native, check_metal, check_cuda):
        out = check(src)
        assert "transposed 0.0000 same 1.0000" in out and "swapped 0.0000 renamed 1.0000 0.0000" in out, out
    old = tmp_path / "old.weights"                       # a version 1 file: "EINW", 1, n, counts, data
    data = np.arange(32, dtype=np.float32)
    old.write_bytes(struct.pack("<4sIQQ", b"EINW", 1, 1, 32) + data.tobytes())
    out = check_native(f"""
model A:
    param W: [4, 8]
a = A()
print(load(a, "{old}"), sum(a.W))
""")
    assert out.split() == ["1.0000", "496.0000"], out


def test_save_and_load(tmp_path):
    """A roundtrip restores the parameters; a missing file or one with other sizes changes nothing;
    a file written by native code loads in the interpreter."""
    import io

    import numpy as np

    from util import Interpreter, compile_text, run_native
    net_file, other_file = tmp_path / "net.weights", tmp_path / "other.weights"
    model = """
model Net:
    l1 = Linear(3, 4)
    l2 = Linear(4, 2)
    fn forward(x) = x |> l1 |> relu |> l2
net = Net()
"""
    src = model + f"""
param other: [5] ~ normal(0, 1)
print("missing", load(net, "{tmp_path / 'never-written.weights'}"))
before = net.l2.W + 0
save(net, "{net_file}")
save(other, "{other_file}")
x: [8, 3] ~ normal(0, 1)
for step in range(20):
    loss = mean(net(x) ** 2)
    minimize loss with sgd(lr=0.1)
print("changed", sum(abs(net.l2.W - before)) > 0)
print("loaded", load(net, "{net_file}"), "restored", sum(abs(net.l2.W - before)))
print("other sizes", load(net, "{other_file}"), "unchanged", sum(abs(net.l2.W - before)))
print(net.l2.W)
"""
    out = check_native(src)
    assert "missing 0.0000" in out and "changed 1.0000" in out
    assert "loaded 1.0000 restored 0.0000" in out and "other sizes 0.0000 unchanged 0.0000" in out
    native_w = out.split("unchanged 0.0000")[1]

    out, err, code, _ = run_native(src)                 # leave the native file behind
    assert code == 0, err
    reader = model + f"""
print(load(net, "{net_file}"))
print(net.l2.W)
"""
    buf = io.StringIO()
    Interpreter(compile_text(reader).program, out=buf).run()
    flag, w = buf.getvalue().split("\n", 1)
    assert flag == "1.0000"
    assert np.allclose([float(v) for v in w.replace("[", " ").replace("]", " ").replace(",", " ").split()],
                       [float(v) for v in native_w.replace("[", " ").replace("]", " ").replace(",", " ").split()])


def test_csv(tmp_path):
    """csv(): shape from the file at compile time, the same numbers on both backends as NumPy reads."""
    import numpy as np

    from anvil.diagnostics import AnvilError
    from util import compile_text
    rng = np.random.default_rng(0)
    table = rng.normal(size=(2000, 7)).astype(np.float32) * rng.choice([1e-3, 1, 1e4], size=(2000, 7))
    path = tmp_path / "big.csv"
    with open(path, "w") as f:
        f.write("a,b,c,d,e,f,label\n")
        for row in table:
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    expect = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    src = f"""
t = csv("{path}")
print(t.shape)
print(sum(t[:, 0]), max(t[:, 3]), t[1999, 6], t[0])
"""
    out = check_native(src)
    assert out.splitlines()[0] == "(2000, 7)"
    s0 = float(out.splitlines()[1].split()[0])
    assert np.isclose(s0, expect[:, 0].sum(dtype=np.float64), rtol=1e-4)

    bad = tmp_path / "bad.csv"
    for text, msg in [("1,2\n3,x\n", "line 2: `x` is not a number"),
                      ("1,2\n3,4,5\n", "line 2: 3 fields, but the table has 2 columns"),
                      ("1;2\n3;4\n", "`3;4` is not a number")]:
        bad.write_text(text)
        try:
            compile_text(f'x = csv("{bad}")\n')
            raise AssertionError("no error for " + repr(text))
        except AnvilError as e:
            assert msg in e.message, e.message
    bad.write_text("1;2\n3;4\n")
    assert check_native(f'print(csv("{bad}", sep=";"))\n').startswith("[[1.0000, 2.0000]")


def test_bytes_and_decode(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("Hello, Anvil!\nsecond line\n")
    out = check_native(f"""
text = bytes("{path}")
print(text.shape, text[0:5])
print("[{{decode(text[0:13])}}]", decode(text[14:20]))
up[k] = text[k] - 32 if text[k] >= 97 and text[k] <= 122 else text[k]
print(decode(up), end="")
""")
    assert out == "(26) [72, 101, 108, 108, 111]\n[Hello, Anvil!] second\nHELLO, ANVIL!\nSECOND LINE\n"


def test_window_out_of_bounds_is_caught():
    from util import run_native
    src = "data[k] = k where k < 50\ns = [0, 43]\nx[b, t] = data[s[b] + t] where t < 8\nprint(x)\n"
    out, err, code, _ = run_native(src)
    assert code != 0 and "out of bounds" in err and "got 43" in err


def test_conv2d_padding_and_stride_match_numpy():
    import numpy as np

    from util import run_text
    src = """
img: [2, 3, 9, 9] ~ normal(0, 1)
conv = Conv2d(3, 4, 3, stride=2, pad=1)
out = conv(img)
"""
    _, it, c = run_text(src, float_dtype=np.float64)
    g = c.elab.globals.vars
    arr = lambda b: it.arr(b)[:b.numel].reshape(b.shape)
    x, out = arr(g["img"].val.buf), arr(g["out"].val.buf)
    model = g["conv"].val.scope.vars
    W, b = arr(model["W"].ref), arr(model["b"].ref)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    want = np.zeros((2, 4, 5, 5))
    for i in range(5):
        for j in range(5):
            win = xp[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3]
            want[:, :, i, j] = np.einsum("ncuv,ocuv->no", win, W) + b
    assert np.allclose(out, want)


def test_assert():
    from util import run_native
    src = """
x: [5] ~ normal(0, 1)
assert len(x) == 5
for i in range(4):
    assert i < 3, "i = {i}, mean {mean(x):.3f}"
    print("step", i)
"""
    out, err, code, _ = run_native(src)
    assert code == 1 and out == "step 0\nstep 1\nstep 2\n"
    assert "assertion failed at <test>.anvil:5: i = 3, mean" in err


def test_nan_survives_simplification():
    """0 / x is only folded to 0 for a known non-zero x: `loss == loss` must see a nan."""
    from util import run_native
    src = 'z = zeros(1)[0]\nloss = z / z\nassert loss == loss, "loss is nan"\nprint("not reached")\n'
    out, err, code, _ = run_native(src)
    assert code == 1 and "loss is nan" in err and out == ""


def test_repl_session():
    from anvil.repl import Session
    s = Session()
    assert s.run("x = [1.0, 2.0, 3.0]") == ("", "")
    assert s.run("x * 2") == ("[2.0000, 4.0000, 6.0000]\n", "")
    out, err = s.run("sofmax(x)")
    assert out == "" and "did you mean `softmax`" in err          # dropped from the session
    assert s.run("for i in range(2):\n    print(i + sum(x))") == ("6.0000\n7.0000\n", "")
    assert s.run("mean(x)") == ("2.0000\n", "")                    # earlier output is not repeated
    assert "sofmax" not in "\n".join(s.lines)
