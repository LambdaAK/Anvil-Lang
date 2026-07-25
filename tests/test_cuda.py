"""The CUDA backend. With no GPU here, each generated .cu file is compiled as C++ (-DANVIL_EMULATE runs
every launch as a loop) and its output compared with the interpreter's; clang's CUDA front end
type-checks the same files for the device and the host."""
import os
import shutil
import subprocess

import pytest

from util import ROOT, check_cuda, compile_text, run_cuda

from anvil.backend.cuda import generate

STUB = os.path.join(ROOT, "tests", "cuda_stub")


def example(name, replace=()):
    path = os.path.join(ROOT, "examples", f"{name}.anvil")
    src = open(path).read()
    for old, new in replace:
        assert old in src, old
        src = src.replace(old, new)
    return src, path


@pytest.mark.parametrize("name", ["hello", "linear_regression", "attention"])
def test_example_matches_interpreter(name):
    src, path = example(name)
    check_cuda(src, path=path)


def test_snake():
    src, path = example("snake", [("sleep(0.05)", "sleep(0.0)")])
    out = check_cuda(src, path=path, consts={"EPISODES": 4})
    assert "watching the trained snake" in out


def test_checkers_with_input():
    """Self-play, search, the game on screen, and a game against scripted input."""
    src, path = example("checkers", [("sleep(0.15)", "sleep(0.0)"), ("{clock() - start:.1f}s", "{0 * start:.1f}s")])
    consts = {"GAMES": 3, "MATCH": 2, "MAX_PLY": 30, "PLAY": 1, "PLAY_DEPTH": 2, "SAVE": "/nonexistent/c.weights"}
    out = check_cuda(src, path=path, stdin="3\n9\nx\n1\n1\n", consts=consts)
    assert "the critic plays" in out


def test_transformer():
    src, path = example("transformer", [("{clock() - start:.1f}s", "{0 * start:.1f}s"),
                                        ("step % 250 == 249", "step % 5 == 4")])
    check_cuda(src, path=path, consts={"STEPS": 20, "BATCH": 4, "T": 8, "D": 16, "HEADS": 2, "SAMPLE": 10})


SPLIT = """
x: [300000] ~ normal(0, 1)
print("sum {sum(x):.2f}  max {max(x):.4f}  min {min(x):.4f}  mean {mean(x):.5f}")
W: [3, 50000] ~ uniform(-1, 1)
y: [50000] ~ normal(0, 1)
v[i] = sum W[i, j] * y[j]
print(v)
print(argmax(x))
R = rand(3, 50000)
noisy[i] = sum W[i, j] * R[i, j]
print(noisy)
"""


def test_split_reductions():
    """Few outputs with long reductions: each output is reduced by many threads, then combined."""
    check_cuda(SPLIT)
    src = generate(compile_text(SPLIT).program)
    assert "split reduction, part 1 of 2 (1024 threads per output)" in src


def test_scatter_add_and_gather():
    """`+=` through an index becomes atomicAdd; gathers are bounds-checked."""
    src = """
labels: i32[1000] ~ randint(0, 10)
counts: i32[10] = 0
counts[labels[i]] += 1
print(counts)
E: [10, 4] ~ normal(0, 1)
rows[i, d] = E[labels[i], d]
print(rows[0:5])
"""
    check_cuda(src)
    assert "atomicAdd(&" in generate(compile_text(src).program)


def test_runtime_bounds_check():
    out, err, code, _ = run_cuda("""
x = arange(10)
for i in range(12):
    s = x[i]
    print(s)
""")
    assert code != 0 and "out of bounds" in err
    assert out.split() == [str(i) for i in range(10)]


def test_gather_out_of_bounds_inside_a_kernel():
    out, err, code, _ = run_cuda("""
idx: i32[4] = [0, 1, 7, 2]
E: [5, 3] ~ normal(0, 1)
r[i, d] = E[idx[i], d]
print(r)
""")
    assert code != 0 and "index out of bounds" in err and "got 7" in err


def test_save_load_and_assert(tmp_path):
    ckpt = str(tmp_path / "w.weights")
    src = f"""
param W: [3, 4] ~ normal(0, 1)
loaded = load(W, "{ckpt}")
if loaded == 0:
    save(W, "{ckpt}")
print("loaded {{loaded}} sum {{sum(W):.4f}}")
assert sum(W) < 100, "the sum is small"
"""
    first, err, code, _ = run_cuda(src)            # learns nothing, saves W
    assert code == 0, err
    second, err, code, _ = run_cuda(src)           # loads it back
    assert code == 0, err
    assert "loaded 0.0000" in first and "loaded 1.0000" in second
    assert first.split("sum")[1] == second.split("sum")[1]
    out, err, code, _ = run_cuda("x = sum(rand(3)) + 3\nassert x < 2, \"x is too big\"\nprint(\"unreachable\")\n")
    assert code != 0 and "x is too big" in err and "unreachable" not in out


def clang_cuda():
    cxx = shutil.which("clang++")
    if cxx is None:
        return None
    r = subprocess.run([cxx, "-x", "cuda", "--cuda-gpu-arch=sm_70", "-nocudainc", "-nocudalib", "--cuda-device-only",
                        "-fsyntax-only", "-"], input="__attribute__((global)) void k() {}\n",
                       capture_output=True, text=True)
    return cxx if r.returncode == 0 else None


@pytest.mark.parametrize("name", ["hello", "checkers", "transformer", "vae"])
def test_clang_cuda_typechecks(name, tmp_path):
    """The generated .cu is valid CUDA C++ on both sides: no host function is called from a kernel."""
    cxx = clang_cuda()
    if cxx is None:
        pytest.skip("clang without CUDA support")
    src, path = example(name)
    cu = tmp_path / f"{name}.cu"
    cu.write_text(generate(compile_text(src, path=path).program))
    for side in ("--cuda-device-only", "--cuda-host-only"):
        r = subprocess.run([cxx, "-x", "cuda", "-std=c++17", "--cuda-gpu-arch=sm_70", "-nocudainc", "-nocudalib", side,
                            "-fsyntax-only", "-Werror", f"-I{STUB}", str(cu)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]


def test_diffusion():
    """Product reductions (the noise schedule), embedding gathers and their scatter-add gradients,
    bernoulli, and a run-time loop with guidance."""
    src, path = example("diffusion", [("{clock() - start:.1f}s", "{0 * start:.1f}s")])
    out = check_cuda(src, path=path, consts={"EPOCHS": 1, "T": 8, "H": 16, "BATCH": 3000, "PER_DIGIT": 1})
    assert "digits it drew" in out
