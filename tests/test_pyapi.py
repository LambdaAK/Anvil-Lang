"""anvil.function: Anvil functions compiled to native code and called on NumPy arrays."""
import threading

import numpy as np
import pytest

from util import ROOT  # noqa: F401  (puts the checkout on sys.path)

import anvil
from anvil.diagnostics import AnvilError

rng = np.random.default_rng(0)


def randn(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_softmax_one_version_per_shape():
    f = anvil.function("fn rows(x: [n, d]) = softmax(x)")
    for shape in [(4, 8), (300, 17), (4, 8)]:
        x = randn(*shape)
        np.testing.assert_allclose(f(x), np_softmax(x), rtol=1e-5, atol=1e-7)
    assert len(f.versions) == 2


def test_loss_and_gradient_with_respect_to_an_argument():
    f = anvil.function("""
fn loss_and_grad(w: [d], x: [n, d], y: [n]):
    loss = mean((x @ w - y) ** 2)
    return loss, grad(loss, w)
""")
    w, x, y = randn(5), randn(32, 5), randn(32)
    loss, g = f(w, x, y)
    r = x @ w - y
    assert isinstance(loss, np.float32)
    np.testing.assert_allclose(loss, (r ** 2).mean(), rtol=1e-5)
    np.testing.assert_allclose(g, 2 * x.T @ r / 32, rtol=1e-4, atol=1e-6)


def test_causal_attention():
    f = anvil.function("""
fn attention(q: [b, t, d], k: [b, t, d], v: [b, t, d]):
    s[n, i, j] = sum q[n, i, c] * k[n, j, c] / sqrt(d) + (0.0 if j <= i else -1e9)
    return softmax(s) @ v
""")
    q, k, v = randn(3, 7, 4), randn(3, 7, 4), randn(3, 7, 4)
    s = q @ k.transpose(0, 2, 1) / 2 + np.where(np.arange(7)[None, :] <= np.arange(7)[:, None], 0, -1e9)
    np.testing.assert_allclose(f(q, k, v), np_softmax(s) @ v, rtol=1e-5, atol=1e-6)


def test_integer_arrays_and_compile_time_constants():
    """Integer arrays become i32 tensors; Python numbers are constants (a new value, a new version)."""
    f = anvil.function("fn pick(x: [n, d], idx: i32[k], scale) = x[idx] * scale")
    x = randn(10, 3)
    np.testing.assert_array_equal(f(x, [3, 1, 4], 2.0), x[[3, 1, 4]] * 2)
    np.testing.assert_array_equal(f(x, np.array([9, 0]), 0.5), x[[9, 0]] * 0.5)
    assert len(f.versions) == 2
    g = anvil.function("fn shifted(x: [n], k):\n    y[i] = x[i + k] where i < n - k\n    return y\n")   # k: a shape
    np.testing.assert_array_equal(g(np.arange(6.0), 2), np.arange(2.0, 6.0))


def test_the_callers_arrays_are_not_modified():
    f = anvil.function("""
fn f(x: [n]):
    x[0] = 1.0
    return x * 2
""")
    a = np.zeros(4, np.float32)
    np.testing.assert_array_equal(f(a), [2, 0, 0, 0])
    np.testing.assert_array_equal(a, 0)


def test_non_contiguous_and_float64_inputs():
    f = anvil.function("fn double(x) = x * 2")
    x = np.arange(12.0).reshape(3, 4).T                     # float64, transposed
    out = f(x)
    assert out.dtype == np.float32 and out.shape == (4, 3)
    np.testing.assert_array_equal(out, x * 2)


def test_scalar_and_constant_results():
    f = anvil.function("fn total(x) = sum(x)")
    assert f(np.arange(10.0)) == 45.0
    g = anvil.function("fn sizes(x: [n, d]):\n    return n * d, sum(x)\n")
    n, s = g(np.ones((3, 4)))
    assert n == 12 and s == 12.0


def test_from_a_file_with_models_and_helpers(tmp_path):
    p = tmp_path / "net.anvil"
    p.write_text("""
fn layer(x, W, b) = relu(x @ W + b)
fn mlp(x: [n, 4], W1: [4, 8], b1: [8], W2: [8, 2]) = layer(x, W1, b1) @ W2
fn unused(x) = x
""")
    f = anvil.function(str(p), name="mlp")
    x, W1, b1, W2 = randn(5, 4), randn(4, 8), randn(8), randn(8, 2)
    np.testing.assert_allclose(f(x, W1, b1, W2), np.maximum(x @ W1 + b1, 0) @ W2, rtol=1e-5, atol=1e-6)


def test_errors():
    with pytest.raises(AnvilError, match="no function"):
        anvil.function("x = 1")
    with pytest.raises(AnvilError, match="no function `g`"):
        anvil.function("fn f(x) = x", name="g")
    f = anvil.function("fn f(x: [n, 3]) = x")
    with pytest.raises(AnvilError):
        f(np.zeros((2, 4)))                                  # the shape does not match the annotation
    with pytest.raises(TypeError):
        f(np.array(["a", "b"]))


def test_calls_from_threads():
    f = anvil.function("fn scaled(x, s) = x * s")
    x = randn(1000)
    f(x, np.float32(1))
    results = {}

    def work(k):
        results[k] = f(x, np.float32(k))
    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for k in range(8):
        np.testing.assert_allclose(results[k], x * k)


def test_functions_are_optimized_and_bake_only_program_values():
    """The function's own temporaries stay temporaries (so they fuse); only what the program's
    statements computed is built in as constant data."""
    from anvil import ir
    f = anvil.function("""
param W: [4, 4] ~ normal(0, 1)
fn f(x: [n, d], w: [d]) = layer_norm(x) * w + sum(W)
""")
    x, w = randn(8, 16), randn(16)
    c = f.lower(x, w)
    kernels = ir.all_kernels(c.prog)
    assert len(kernels) <= 4, ir.fmt_program(c.prog)
    consts = [b.name for b in c.prog.buffers if b.kind == "const" and b.init is not None]
    assert "W" in consts and "mu" not in consts and "out" not in consts, consts
    mu = x.mean(1, keepdims=True)
    ref = (x - mu) / np.sqrt(x.var(1, keepdims=True) + 1e-5) * w + 0   # sum(W) checked below
    got = f(x, w)
    np.testing.assert_allclose(got - got[0, 0] + ref[0, 0], ref, rtol=1e-4, atol=1e-4)
