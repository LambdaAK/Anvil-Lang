"""The Metal backend on this Mac's GPU: outputs must match the reference interpreter."""
import os
import subprocess

import pytest

from util import ROOT, check_metal, compile_text, run_metal

from anvil.backend.metal import MetalGen, generate


def have_metal():
    if subprocess.run(["uname", "-s"], capture_output=True, text=True).stdout.strip() != "Darwin":
        return False
    r = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
    return "Metal" in r.stdout


pytestmark = pytest.mark.skipif(not have_metal(), reason="needs a Metal GPU")


def example(name, replace=()):
    path = os.path.join(ROOT, "examples", f"{name}.anvil")
    src = open(path).read()
    for old, new in replace:
        assert old in src, old
        src = src.replace(old, new)
    return src, path


@pytest.mark.parametrize("name", ["hello", "linear_regression", "attention"])
def test_example(name):
    src, path = example(name)
    check_metal(src, path=path)


def test_snake_and_checkers():
    src, path = example("snake", [("sleep(0.05)", "sleep(0.0)")])
    check_metal(src, path=path, consts={"EPISODES": 4})
    src, path = example("checkers", [("sleep(0.15)", "sleep(0.0)"), ("{clock() - start:.1f}s", "{0 * start:.1f}s")])
    consts = {"GAMES": 3, "MATCH": 2, "MAX_PLY": 30, "PLAY": 1, "PLAY_DEPTH": 2, "SAVE": "/nonexistent/c.weights"}
    assert "the critic plays" in check_metal(src, path=path, stdin="3\n9\nx\n1\n1\n", consts=consts)


def test_transformer_and_diffusion():
    src, path = example("transformer", [("{clock() - start:.1f}s", "{0 * start:.1f}s"),
                                        ("step % 250 == 249", "step % 5 == 4")])
    check_metal(src, path=path, consts={"STEPS": 20, "BATCH": 4, "T": 8, "D": 16, "HEADS": 2, "SAMPLE": 10})
    src, path = example("diffusion", [("{clock() - start:.1f}s", "{0 * start:.1f}s")])
    check_metal(src, path=path, consts={"EPOCHS": 1, "T": 8, "H": 16, "BATCH": 3000, "PER_DIGIT": 1})


def test_products_through_mps():
    src = """
A: [96, 80] ~ normal(0, 1)
B: [80, 72] ~ normal(0, 1)
C = A @ B
D[i, j] = sum A[k, i] * C[k, j]              # a transposed operand
E = relu(A @ B + 1)                          # an epilogue after the product
q: [2, 64, 3, 32] ~ normal(0, 1)
s[b, h, i, j] = sum q[b, i, h, e] * q[b, j, h, e]      # a stack of products (heads)
x: [4, 64, 48] ~ normal(0, 1)
g[c, o] = sum x[n, c, i] * x[n, o, i]       # a sum of products
print(sum(C), sum(D), sum(E), s[1, 2, 3, 0:3], g[5, 0:3])
"""
    check_metal(src)
    gen = MetalGen(compile_text(src).program)
    gen.generate()
    assert len(gen.gemms) >= 4


def test_loop_counters_stay_on_the_host():
    """`if step % 10 == 0` is decided on the host, so the GPU never waits for it; the optimizer's
    step count too."""
    src = """
x: [64, 8] ~ normal(0, 1)
param w: [8] ~ normal(0, 1)
for step in range(30):
    loss = mean((x @ w - 1) ** 2)
    minimize loss with adam(lr=0.05)
    if step % 10 == 0:
        print("step {step} loss {loss:.4f}")
"""
    check_metal(src)
    gen = MetalGen(compile_text(src).program)
    code = gen.generate()
    assert gen.hosts, "the loop counter and what depends only on it live on the host"
    body = code.split("int main(void)")[1]
    loop = body[body.index("for (b"):body.index("anvil_end:")]
    # inside the training loop, the only wait is the print's (Adam's step count is on the host too)
    assert loop.count("anvil_check_faults();") == 1, loop
    assert "if (b" in loop.split("anvil_check_faults();")[0].rstrip().splitlines()[-1]


def test_bounds_fault_on_the_gpu():
    out, err, code, _ = run_metal("idx: i32[4] = [0, 1, 7, 2]\nE: [5, 3] ~ normal(0, 1)\nr[i, d] = E[idx[i], d]\nprint(r)\n")
    assert code != 0 and "index out of bounds" in err and "got 7" in err


def test_generated_source_compiles_standalone(tmp_path):
    src, path = example("mnist")
    mm = generate(compile_text(src, path=path).program)
    assert "kernel void" in mm and "MPSMatrixMultiplication" in mm


def test_kernels_with_more_tensors_than_metal_binds():
    """Metal binds at most 31 buffers to a kernel; the rest go through an argument buffer of GPU
    addresses. A sum of 40 unrolled losses fuses into one kernel that reads 40 tensors."""
    src = """
x: [40, 16] ~ normal(0, 1)
param w: [16] ~ normal(0, 0.1)
for step in range(3):
    loss = 0.0
    static for t in range(40):
        loss = loss + mean((x[t] * w - 0.5) ** 2) * (t + 1)
    minimize loss with sgd(lr=0.001)
    print(loss)
print(w[0:4])
"""
    check_metal(src)
    assert "_more &anvil_more" in generate(compile_text(src).program)
