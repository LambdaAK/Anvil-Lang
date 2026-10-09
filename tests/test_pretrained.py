"""What running a pretrained model takes: weights read from .safetensors files by name, lists of layers,
one optimizer state across `minimize` statements, erf (for BERT's exact GELU), names built from
constants, data paths relative to the file that names them, and programs whose memory does not fit
in a static segment. examples/bge_small.anvil uses all of them."""
import json
import math
import os
import re

import numpy as np
import pytest

from util import ROOT, check_native, compile_text, run_native, run_text

from anvil.diagnostics import AnvilError


def write_safetensors(path, tensors: dict):
    """A .safetensors file: an 8-byte header length, a JSON header, then the raw little-endian data."""
    codes = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int64"): "I64",
             np.dtype("int32"): "I32", np.dtype("uint8"): "U8"}
    header, blobs, at = {}, [], 0
    for name, a in tensors.items():
        raw = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": codes[a.dtype], "shape": list(a.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    header["__metadata__"] = {"format": "pt"}
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + b"".join(blobs))


@pytest.fixture
def weights(tmp_path):
    rng = np.random.default_rng(0)
    t = {"layer.0.weight": rng.normal(size=(3, 4)).astype(np.float32),     # [out, in], as PyTorch keeps it
         "layer.0.bias": rng.normal(size=3).astype(np.float32),
         "layer.1.weight": rng.normal(size=(2, 3)).astype(np.float32),
         "layer.1.bias": np.array([0.5, -0.25], dtype=np.float32),
         "half": np.array([1.5, -2.0, 65504.0], dtype=np.float16),
         "ids": np.array([[1, 2], [3, -4]], dtype=np.int64)}
    path = tmp_path / "model.safetensors"
    write_safetensors(path, t)
    return path, t


def test_safetensors_reads_tensors_by_name(weights):
    path, t = weights
    src = f"""
const M = "{path}"
W0 = safetensors(M, "layer.0.weight")
print(shape(W0), W0)
print(safetensors(M, "half"), safetensors(M, "ids"))
print(safetensors(M, "layer.1.bias"))
"""
    out = check_native(src)                     # native code, and the interpreter, read the same numbers
    assert out.startswith("(3, 4)")
    assert "[1.5000, -2.0000, 65504.0000]" in out and "[[1, 2],\n [3, -4]]" in out and "[0.5000, -0.2500]" in out


def test_a_model_from_a_safetensors_file(weights):
    """Layers named by the numbers of a list comprehension (`layers.0`, `layers.1`), their weights from
    the file by name; the forward pass is NumPy's."""
    path, t = weights
    src = f"""
const M = "{path}"
fn weight(name) = safetensors(M, name)
model Dense(name, fan_in, fan_out):
    param W: [fan_in, fan_out] = transpose(weight(name + ".weight"))
    param b: [fan_out] = weight(name + ".bias")
    fn forward(x) = x @ W + b
model Net:
    layers = [Dense("layer.{{k}}", 4 - k, 3 - k) for k in range(2)]
    fn forward(x):
        static for layer in layers:
            x = layer(x)
        return x
net = Net()
x = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]]
print(net(x))
print(net.layers[1].b)
"""
    out = check_native(src)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 2.0]], dtype=np.float32)
    want = (x @ t["layer.0.weight"].T + t["layer.0.bias"]) @ t["layer.1.weight"].T + t["layer.1.bias"]
    got = np.array([float(v) for v in re.findall(r"-?\d+\.\d+", out.split("]]")[0])]).reshape(2, 2)
    assert np.allclose(got, want, atol=1e-3), (got, want)
    assert out.strip().endswith("[0.5000, -0.2500]")


def test_safetensors_errors(weights):
    path, _ = weights
    with pytest.raises(AnvilError) as e:
        compile_text(f'x = safetensors("{path}", "layer.0.wieght")\n')
    assert "has no tensor `layer.0.wieght`" in e.value.render() and "did you mean `layer.0.weight`?" in e.value.render()
    with pytest.raises(AnvilError) as e:
        compile_text('x = safetensors("no-such-model.safetensors", "w")\n')
    assert "model file not found" in e.value.render()


def model_list_program(save_to=None, load_from=None):
    return f"""
model Layer(k):
    param W: [3, 3] = f32(k + 1) * eye(3)
    fn forward(x) = relu(x @ W)
model Net(n):
    layers = [Layer(k) for k in range(n)]
    fn forward(x):
        static for layer in layers:
            x = layer(x)
        return x
net = Net(3)
{f'print("loaded", load(net, "{load_from}"))' if load_from else ''}
x: [4, 3] ~ normal(0, 1)
for step in range(5):
    minimize mean((net(x) - 1) ** 2) with sgd(lr=0.01)
print(net.layers[2].W)
{f'save(net, "{save_to}")' if save_to else ''}
"""


def test_lists_of_models_train_save_and_load(tmp_path):
    check_native(model_list_program())
    f = tmp_path / "net.weights"
    out, err, code, c = run_native(model_list_program(save_to=f))
    assert code == 0, err
    names = sorted(b.name for b in c.elab.model_params(c.elab.globals.vars["net"].val))
    assert names == ["net.layers.0.W", "net.layers.1.W", "net.layers.2.W"]
    again, err, code, _ = run_native(model_list_program(load_from=f))
    assert code == 0 and "loaded 1.0000" in again, err


def test_list_comprehensions_of_values():
    out, _, _ = run_text("sq = [k * k for k in range(5)]\nprint(sq)\nprint(sq[3] + 1)\n")
    assert out == "(0, 1, 4, 9, 16)\n10\n"


def test_names_built_from_constants_are_known_at_compile_time():
    out, _, _ = run_text('const K = 7\nname = "encoder.layer.{K}." + "attention"\nprint(name == "encoder.layer.7.attention")\n')
    assert out == "1\n" or out == "true\n" or out.strip() in ("1", "1.0000", "true")


def test_one_optimizer_state_across_minimize_statements():
    """Two `minimize` statements with adam on the same parameter continue one Adam (one m, v and step
    count), as in PyTorch with one optimizer."""
    out = check_native("""
param w: [3] = 1.0
a = [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]]
b = [[1.0, 1.0, 1.0]]
for k in range(3):
    minimize sum(a @ w) ** 2 with adam(lr=0.1)
    minimize sum(b @ w) ** 2 with adam(lr=0.1)
print(w)
""")
    a = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
    b = np.array([[1.0, 1.0, 1.0]])
    w, m, v, t = np.ones(3), np.zeros(3), np.zeros(3), 0
    for _ in range(3):
        for M in (a, b):
            g = 2 * (M @ w).sum() * M.sum(0)
            t += 1
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            w -= 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    got = [float(x) for x in out.strip()[1:-1].split(",")]
    assert np.allclose(got, w, atol=2e-4), (got, w)


def test_erf_and_the_exact_gelu():
    out = check_native("""
x = linspace(-4.0, 4.0, 9)
print(erf(x))
print(gelu(x, exact=true))
""")
    xs = np.linspace(-4, 4, 9)
    lines = out.strip().split("\n")
    erf = [float(v) for v in lines[0].strip("[]").split(",")]
    gelu = [float(v) for v in lines[1].strip("[]").split(",")]
    assert np.allclose(erf, [math.erf(v) for v in xs], atol=2e-4)
    assert np.allclose(gelu, [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in xs], atol=2e-4)


def test_data_paths_are_relative_to_the_file_that_names_them(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    np.save(lib / "table.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
    (lib / "model.anvil").write_text('fn table() = npy("table.npy")\n')
    prog = tmp_path / "main.anvil"
    prog.write_text('use "lib/model.anvil"\nprint(sum(table()))\n')
    out, _, _ = run_text(prog.read_text(), path=str(prog))
    assert out.strip() == "15.0000"


def test_a_program_too_big_for_a_static_segment(monkeypatch):
    """Past STATIC_MAX, the arena of temporaries (then the largest tensors) is allocated when the program
    starts, and reached through pointers: the results do not change."""
    import anvil.backend.aarch64 as a64
    src = """
x: [64, 96] ~ normal(0, 1)
param W1: [96, 128] ~ normal(0, 0.1)
param W2: [128, 10] ~ normal(0, 0.1)
for step in range(3):
    h = relu(x @ W1)
    y = softmax(h @ W2)
    loss = mean((y - 0.1) ** 2) + mean(h * h) / 100
    minimize loss with adam(lr=0.01)
print(W2[0:2, 0:4], sum(W1))
"""
    monkeypatch.setattr(a64, "STATIC_MAX", 1 << 14)
    c = compile_text(src)
    asm, gen = a64.generate(c.program)
    assert gen.arena_heap is not None and "in the arena on the heap" in asm
    check_native(src)


def test_layer_norm_takes_an_epsilon():
    out, _, _ = run_text("ln = LayerNorm(4, 1e-12)\nx = [[1.0, 1.0, 1.0, 1.0]]\nprint(ln(x))\n")
    assert out.strip() == "[[0.0000, 0.0000, 0.0000, 0.0000]]"


BGE = os.path.join(ROOT, "examples", "data", "bge-small", "model.safetensors")


@pytest.mark.skipif(not os.path.exists(BGE), reason="needs BAAI/bge-small-en-v1.5 (examples/sentiment/prepare.py)")
def test_the_bge_small_clone_compiles():
    path = os.path.join(ROOT, "examples", "sentiment", "clone_check.anvil")
    c = compile_text(open(path).read(), path=path)
    params = [b for b in c.program.buffers if b.kind == "param"]
    layer = 4 * (384 * 384 + 384) + (384 * 1536 + 1536) + (1536 * 384 + 384) + 2 * 2 * 384
    embeddings = 30522 * 384 + 64 * 384 + 384 + 2 * 384           # words, 64 places, segment 0, layer norm
    assert len(params) == 5 + 12 * 16 and sum(b.numel for b in params) == embeddings + 12 * layer


def test_the_command_line_optimizes(tmp_path):
    """`anvil run`, `build` and `ir` fuse kernels (and -O0 does not): an Adam update is two kernels, not
    one per operation."""
    import subprocess
    prog = tmp_path / "adam.anvil"
    prog.write_text("param W: [64, 64] ~ normal(0, 0.1)\nx: [16, 64] ~ normal(0, 1)\n"
                    "for k in range(3):\n    minimize mean((x @ W) ** 2) with adam(lr=0.01)\n")
    count = lambda *flags: subprocess.run([os.path.join(ROOT, "bin", "anvil"), "ir", str(prog), *flags],
                                          capture_output=True, text=True).stdout.count("⟨")
    assert count() < 12 < count("-O0")
