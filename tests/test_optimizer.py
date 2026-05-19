"""The optimizer must not change program output."""
import numpy as np
import pytest

from test_gradcheck import CASES, PRE
from util import run_text

TRAIN = """
x: [64, 4] ~ normal(0, 1)
labels = i32(argmax(x @ [[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]]))
model Net:
    l1 = Linear(4, 16)
    l2 = Linear(16, 3)
    fn forward(x) = x |> l1 |> relu |> l2
net = Net()
for step in range(30):
    loss = cross_entropy(net(x), labels)
    minimize loss with OPT
print("{loss:.6f} {accuracy(net(x), labels):.4f}")
print(net.l2.b)
"""


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_output(name):
    src = PRE + CASES[name] + "\ndW, db, dv = grad(loss, W, b, v)\nprint(loss, dW, db, dv)\n"
    a, _, _ = run_text(src, optimize=False, float_dtype=np.float64)
    b, _, _ = run_text(src, optimize=True, float_dtype=np.float64)
    assert a == b


@pytest.mark.parametrize("opt", ["sgd(lr=0.5)", "sgd(lr=0.1, momentum=0.9)", "adam(lr=0.01)", "rmsprop(lr=0.01)"])
def test_training_same_output(opt):
    src = TRAIN.replace("OPT", opt)
    a, _, _ = run_text(src, optimize=False, float_dtype=np.float64)
    b, _, c = run_text(src, optimize=True, float_dtype=np.float64)
    assert a == b
    first = float(a.split()[0])
    assert first < 1.1, a     # it actually trains


def test_fill_then_item_assignment_in_loop():
    """The fill before the loop must not be fused into the read after it: the loop's item
    assignments are what the read sees."""
    src = """
fn fill(valid):
    value: [6] = -1.0
    for k in range(i32(sum(valid))):
        value[k] = 10.0 * k
    return where(valid, value, -inf)
print(fill([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
"""
    a, _, _ = run_text(src, optimize=False)
    b, _, _ = run_text(src, optimize=True)
    assert a == b == "[0.0000, 10.0000, 20.0000, -inf, -inf, -inf]\n"


def test_compilation_is_deterministic():
    """The same program compiles to the same assembly in every process, whatever Python's string
    hashing (PYTHONHASHSEED) does to the order of sets: a reduction's index order once came from a
    set of names, so the CNN's convolution was tiled differently (and ran slower) from run to run."""
    import os
    import subprocess
    import sys
    from util import ROOT
    outs = set()
    for seed in ("1", "2", "3"):
        r = subprocess.run([sys.executable, "-m", "anvil", "asm", "cnn.anvil", "-o", "-"], capture_output=True, text=True,
                           cwd=os.path.join(ROOT, "examples"), env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=ROOT))
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1
