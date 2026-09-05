"""Standard-library additions: dropout, learning-rate schedules, label smoothing, gradient clipping."""
import math

import numpy as np
import pytest

from util import check_native, run_text


def test_dropout():
    out, it, c = run_text("""
x = ones(100, 100)
y = dropout(x, 0.25)
z = dropout(x, 0.25, training=false)
print(mean(f32(y == 0)), mean(y), sum(z))
""")
    zeros, mean, total = map(float, out.split())
    assert abs(zeros - 0.25) < 0.02 and abs(mean - 1) < 0.03 and total == 10000
    check_native("x = ones(20, 30)\ny = dropout(x, 0.5)\nprint(y)\n")


def test_warmup_cosine():
    out, _, _ = run_text("""
for s in range(0, 120, 10):
    print(warmup_cosine(s, 100, 2.0, warmup=20, floor=0.1))
print(exponential_decay(30, 1.0, 0.5, every=10))
""")
    got = [float(v) for v in out.split()]
    for s, v in zip(range(0, 120, 10), got):
        want = 2.0 * (s + 1) / 20 if s < 20 else 0.1 + 1.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - 20) / 80)))
        assert abs(v - want) < 1e-4, (s, v, want)
    assert abs(got[-1] - 0.125) < 1e-6


def test_label_smoothing():
    out, _, _ = run_text("""
z = [[2.0, 0.5, -1.0], [0.0, 1.0, 3.0]]
y: i32[2] = [0, 2]
print(cross_entropy(z, y), cross_entropy(z, y, smoothing=0.1))
""")
    plain, smooth = map(float, out.split())
    z = np.array([[2.0, 0.5, -1.0], [0.0, 1.0, 3.0]])
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    nll = -logp[[0, 1], [0, 2]].mean()
    assert abs(plain - nll) < 1e-4
    assert abs(smooth - (0.9 * nll - 0.1 * logp.mean())) < 1e-4


@pytest.mark.parametrize("clip", [0.0, 1.0, 100.0])
def test_gradient_clipping(clip):
    """One SGD step with lr = 1: w moves by the gradient, scaled down to norm `clip` if it is larger."""
    src = f"""
param a: [3] = [3.0, 0.0, 4.0]
param b: [2] = [0.0, 12.0]
for s in range(1):
    loss = 0.5 * (sum(a * a) + sum(b * b))
    minimize loss with sgd(lr=1.0, clip_norm={clip})
print(a, b)
"""
    out, it, c = run_text(src)
    vals = [float(v) for v in out.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    g = np.array([3.0, 0.0, 4.0, 0.0, 12.0])                    # the gradient of loss is the parameters
    scale = min(1.0, clip / np.linalg.norm(g)) if clip > 0 else 1.0
    np.testing.assert_allclose(vals, g - scale * g, atol=1e-4)
    check_native(src)
