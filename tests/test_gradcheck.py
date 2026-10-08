"""Autodiff vs. float64 central finite differences."""
import pytest

from util import gradcheck

PRE = """
param W: [4, 5] ~ normal(0, 0.5)
param b: [5] ~ normal(0, 0.5)
param v: [5] ~ normal(0, 0.5)
x: [3, 4] ~ normal(0, 1)
labels = [1, 0, 4]
rows = [1, 0, 3]
"""

CASES = {
    "matmul_bias": "loss = sum((x @ W + b) ** 2)",
    "relu_mlp": "h = relu(x @ W + b)\nloss = sum(h @ v)",
    "tanh_sigmoid": "loss = mean(tanh(x @ W) * sigmoid(b))",
    "cross_entropy": "loss = cross_entropy(x @ W + b, labels)",
    "softmax": "p = softmax(x @ W)\nloss = sum(p * p * b)",
    "exp_log_sqrt": "loss = sum(log(1 + exp(x @ W)) + sqrt(b * b + 1))",
    "division": "loss = sum((x @ W) / (b * b + 1))",
    "pow_general": "loss = sum(abs(x @ W + 2) ** 1.5)",
    "max_reduce": "m[i] = max (x @ W)[i, j]\nloss = sum(m * m)",
    "min_axis": "loss = sum(min(x @ W, axis=0) * b)",
    "select": "z = x @ W\nloss = sum((z if z > 0 else 0.1 * z) * b)",
    "index_notation": "y[i, o] = sum x[i, k] * W[k, o] + b[o]\nloss = mean(y * y)",
    "transpose_view": "loss = sum((W.T @ x.T) * 0.5)",
    "slice_view": "loss = sum(W[1:3, ::2] ** 2) + sum(b[2:])",
    "conv1d_scatter": "k = b[0:3]\nc[t] = sum v[t + s] * k[s]\nloss = sum(c * c)",
    "gather_rows": "e[i, j] = W[rows[i], j] * b[j]\nloss = sum(e * e)",
    "embedding": "e = W[rows]\nloss = sum(e * e * b)",
    "layer_norm": "loss = sum(layer_norm(x @ W + b) * v)",
    "mean_axis_keepdims": "z = x @ W\nloss = sum((z - mean(z, axis=1, keepdims=true)) ** 2)",
    "reshape_view": "loss = sum((reshape(W, 2, 10) @ reshape(W, 10, 2).T.T) ** 2)",
    "gelu_softplus": "loss = sum(gelu(x @ W) + softplus(b))",
    "mse": "loss = mse(x @ W + b, x @ W * 0.5)",
    "batched_matmul": "A = reshape(W, 2, 2, 5)\nloss = sum((A @ reshape(b, 5, 1)) ** 2)",
    "scatter_add": "s = tanh(W)\ns[rows[i], o] += x[i, 0] * b[o]\ns[k, o] -= sum W[k, j] * v[j]\n"
                   "loss = sum(s * s * s)",
    "overwritten_rows": "s = tanh(W * 2)\ns[1] = b * 3\nloss = sum(s * s * s)",
    "through_time": "h = zeros(5)\nstatic for t in range(3):\n    h = tanh(x[t] @ W + h * v + b)\n"
                    "loss = sum(h * h)",
    "gather_window": "w[i, t] = b[rows[i] + t] where t < 2\nloss = sum(w * w * v[0])",
    "prod_reduce": "p[i] = prod (x @ W)[i, k] * 0.5\nz = b * (b > 0)\nloss = sum(p) + prod(z) * prod(v)",
    "conv2d_im2col": "img = reshape(W, 1, 1, 4, 5)\nc = Conv2d(1, 2, 2)\nloss = sum(c(img) * c(img) * b[0])\n"
                     "dWc = grad(loss, c.W)",
    "conv2d_pad_stride": "img = reshape(W, 1, 1, 4, 5)\nc = Conv2d(1, 2, 3, stride=2, pad=1)\n"
                         "loss = sum(c(img) * c(img) * b[0])",
    "layernorm_embedding": "ln = LayerNorm(5)\ne = Embedding(4, 5)\nloss = sum(ln(e(rows) + x[0, 0] * b) * v)",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_gradcheck(name):
    src = PRE + CASES[name] + "\ndW, db, dv = grad(loss, W, b, v)\n"
    gradcheck(src, ["W", "b", "v"], ["dW", "db", "dv"])


def test_input_gradient():
    src = PRE + "x2 = copy(x)\nloss = cross_entropy(x2 @ W + b, labels)\ndx = grad(loss, x2)\n"
    gradcheck(src, ["x2"], ["dx"])


@pytest.mark.parametrize("backend", ["interp", "native"])
def test_reparameterized_samples(backend):
    """z ~ normal(mu, sigma) is mu + sigma * noise: d z / d sigma is that same noise. (The gradient's
    kernel used to draw fresh random numbers, so the gradient with respect to sigma was wrong, and a
    VAE's variance learned from noise.) The same for uniform(lo, hi)."""
    import io
    from anvil.interp import Interpreter
    from util import compile_text, run_native
    src = """
mu: [6] = [0.5, -0.2, 0.3, 1.0, 0.0, 2.0]
lv: [6] = [0.1, -0.4, 0.2, 0.0, 1.0, -1.0]
z: [6] ~ normal(mu, exp(0.5 * lv))
dmu, dlv = grad(sum(z * z * z), mu, lv)
eps = (z - mu) / exp(0.5 * lv)
print(max(abs(dmu - 3 * z * z)))
print(max(abs(dlv - 3 * z * z * eps * 0.5 * exp(0.5 * lv))))
lo: [6] = [0.0, 1.0, -1.0, 0.5, 2.0, 0.0]
hi = lo + exp(lv)
u: [6] ~ uniform(lo, hi)
dlo, dhi = grad(sum(u * u), lo, hi)
frac = (u - lo) / (hi - lo)
print(max(abs(dhi - 2 * u * frac)))
print(max(abs(dlo - 2 * u * (1 - frac))))
"""
    if backend == "interp":
        out = io.StringIO()
        Interpreter(compile_text(src).program, out=out).run()
        text = out.getvalue()
    else:
        text, err, code, _ = run_native(src)
        assert code == 0, err
    errors = [float(x) for x in text.split()]
    assert len(errors) == 4 and max(errors) < 1e-4, text


def test_gradients_through_run_time_control_flow_are_refused():
    """Gradients are taken through `static for` and compile-time `if` only. A gradient whose path
    runs through a run-time loop or branch used to come out silently wrong ([1, 1, 1, 1] for
    d sum(x⁴)/dx through a run-time `for`, and zero through a run-time `if`); now it is an error that
    says how to write it, and the ways it suggests give the right gradients."""
    from anvil.diagnostics import AnvilError
    from util import compile_text, run_native
    loop = """
x: [4] = [1.0, 2.0, 3.0, 4.0]
y = x
for k in range(3):
    y = y * x
print(grad(sum(y), x))
"""
    branch = """
c: [4] = [1.0, -2.0, 3.0, -4.0]
x: [4] = [1.0, 2.0, 3.0, 4.0]
if sum(c) < 0:
    z = c * c
else:
    z = c
print(grad(sum(z * x), c))
"""
    after = """
param w: [3] ~ normal(0, 1)
param v: [3] ~ normal(0, 1)
loss = sum(w * v)
if sum(v) < 100:
    v[0] = 0.0
minimize loss over w with sgd(lr=0.1)
"""
    for src, says in ((loop, "run-time `for` loop"), (branch, "run-time `if`"), (after, "is modified after it was used")):
        with pytest.raises(AnvilError) as e:
            compile_text(src)
        assert says in e.value.message, e.value.message
    out, err, code, _ = run_native(loop.replace("for k", "static for k") + branch.replace(
        "if sum(c) < 0:\n    z = c * c\nelse:\n    z = c\n", "z = c * c if sum(c) < 0 else c\n"))
    assert code == 0, err
    assert out.split("\n")[:2] == ["[4.0000, 32.0000, 108.0000, 256.0000]", "[2.0000, -8.0000, 18.0000, -32.0000]"], out
    # what a run-time branch computes without derivatives (an action chosen by argmax) is fine
    compile_text("""
param w: [3] ~ normal(0, 1)
x: [3] = [1.0, 2.0, 3.0]
a: i32[1] = 0
if sum(x) > 0:
    a[0] = argmax(x * w)
loss = sum(w * w) * f32(a[0] + 1)
minimize loss with sgd(lr=0.1)
""")
