"""Compile errors: the message, and where it points."""
import glob
import os

import pytest

from util import ROOT, compile_text, run_text
from anvil.diagnostics import AnvilError

EXPECT = {
    "shape_mismatch.anvil": ("shape mismatch in `@`", "h @ W1"),
    "broadcast.anvil": ("cannot broadcast shapes [64, 128] and [10]", "a + b"),
    "index_range.anvil": ("index `j` ranges over 784 in one place but 128 in another", "x[j]"),
    "unbound_index.anvil": ("index `j` is not bound", "j"),
    "typo.anvil": ("`softmx` is not defined", "softmx"),
    "nonscalar_loss.anvil": ("the loss must be a scalar", "loss"),
    "fn_shapes.anvil": ("`d` is 3 from an earlier argument but 4", "dense(a, b)"),
    "bounds.anvil": ("index out of bounds", "x[i + 4]"),
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_error_example(name):
    path = os.path.join(ROOT, "examples", "errors", name)
    text = open(path).read()
    with pytest.raises(AnvilError) as e:
        compile_text(text, path=path)
    msg, snippet = EXPECT[name]
    assert msg in e.value.message
    assert e.value.span.text == snippet or snippet in e.value.span.text
    assert e.value.render(color=False)       # renders without crashing


def test_every_error_example_is_covered():
    names = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "examples", "errors", "*.anvil"))}
    assert names == set(EXPECT)


@pytest.mark.parametrize("src,msg", [
    ("x = [1.0, 2.0]\ny = x[5]\n", "out of bounds"),
    ("param w: [3]\nloss = 3.0\nminimize loss with sgd()\n", "constant"),
    ("x: [3] ~ normal(0, 1)\ny = reshape(x, 2, 2)\n", "cannot reshape"),
    ("fn f(x) = f(x)\ny = f(1)\n", "recursive"),
    ("x = 1\nfor i in range(3):\n    i = 2\n", "loop variable"),
    ("y[i] = 1\n", "cannot infer the range"),
    ("const n = 3\nn = 4\n", "constant"),
    ("x = zeros(3)\nk = 1\nx[k] = [1.0, 2.0]\n", "cannot assign a value of shape [2]"),
    ("f = 2\nf[0] = 1\n", "is a constant, not a tensor"),
    ("x = zeros(4)\nx[9] = 1.0\n", "out of bounds"),
    ("x = [1.0, 0.0]\ni = nonzero(x)\n", "needs a static `size`"),
    ("x = zeros(3)\nsave(x, \"x.weights\")\n", "takes a model or a `param`"),
    ("ids = [1, 0]\ncounts[ids[i]] += 1\n", "`counts` is not defined"),
    ("x = zeros(3, 4)\ny = zeros(3)\ny[i] += x[i, j]\n", "index `j` is not bound"),
    ("x = zeros(3, 4)\nx[0:2, i] += 1\n", "cannot be mixed with index names"),
    ("n: i32 ~ randint(1, 5)\nstatic for t in range(n):\n    print(t)\n", "needs a range known at compile time"),
    ("for i in range(3):\n    static for t in range(2):\n        break\n", "inside a `static for`"),
    ("x = \"ab\" * 2.5\n", "repeating"),
    ("assert 2 + 2 == 5\n", "this assertion is false"),
    ("x = zeros(3)\nassert sum(x) == 0, \"{x}\"\n", "not whole tensors"),
    ("N = [\"a\", \"b\"]\nx = 0.5 + zeros(1)[0]\nprint(N[x])\n", "indexed by an integer"),
])
def test_semantic_errors(src, msg):
    with pytest.raises(AnvilError) as e:
        compile_text(src)
    assert msg in e.value.message, e.value.message


def test_set_names_a_constant():
    c = compile_text("const N = 3\nconst NAME = \"a\"\nprint(N, NAME)\n", consts={"N": 5, "NAME": "b"})
    assert c is not None
    with pytest.raises(AnvilError) as e:
        compile_text("const N = 3\nprint(N)\n", consts={"M": 1})
    assert "has no constant `M`" in e.value.message and "its constants: N" in (e.value.help or "")
    with pytest.raises(AnvilError) as e:
        compile_text("const BATCH = 3\nprint(BATCH)\n", consts={"BTACH": 1})
    assert "did you mean `BATCH`" in (e.value.help or "")


def test_set_values():
    from anvil.cli import parse_set
    assert parse_set("GAMES=500") == ("GAMES", 500) and parse_set("LR=3e-4") == ("LR", 3e-4)
    assert parse_set("ON=true") == ("ON", True) and parse_set("SAVE=a/b.weights") == ("SAVE", "a/b.weights")
    # quotes that no shell removed (subprocess arguments, or `--set 'OPT="adam"'`) are not part of the value
    assert parse_set('OPT="adam"') == parse_set("OPT='adam'") == parse_set("OPT=adam") == ("OPT", "adam")
    assert parse_set('N="3"') == ("N", "3")
    with pytest.raises(ValueError):
        parse_set("3=4")


def test_several_errors_are_reported_together():
    from anvil.diagnostics import AnvilErrors
    src = """x = [1.0, 2.0, 3.0]
W: [4, 3] ~ normal(0, 1)
print(sofmax(x))
for step in range(3):
    loss = mean(x @ W)
    minimize loss with sgd(lr=0.1)
q = x + undefined_thing
print(q + 1)
"""
    with pytest.raises(AnvilErrors) as e:
        compile_text(src)
    msgs = [x.message for x in e.value.errors]
    assert len(msgs) == 3, msgs                  # `minimize loss` and `q + 1` are not reported again
    assert "`sofmax` is not defined" in msgs[0] and "shape mismatch in `@`" in msgs[1]
    assert "`undefined_thing` is not defined" in msgs[2]
    assert e.value.render().endswith("3 errors")


def test_strings_compare_at_compile_time():
    """`if OPT == "adam":` picks a branch while compiling (configuration constants for --set)."""
    src = """const OPT = "adam"
if OPT == "adam":
    print("adam")
elif OPT != "sgd":
    print("other")
else:
    print("sgd")
"""
    for value, want in [("adam", "adam"), ("sgd", "sgd"), ("rmsprop", "other")]:
        out, _, _ = run_text(src) if value == "adam" else run_text(src.replace('"adam"\nif', f'"{value}"\nif'))
        assert out == want + "\n", (value, out)
    with pytest.raises(AnvilError, match="strings support"):
        compile_text('x = "a" < "b"\n')


def test_use_brings_in_declarations(tmp_path):
    """`use "file.anvil"`: another file's functions, models, optimizers and constants, once."""
    (tmp_path / "lib.anvil").write_text("const WIDTH = 3\nfn double(x) = 2 * x\nmodel Tiny:\n    l = Linear(2, WIDTH)\n"
                                      "    fn forward(x) = relu(l(x))\n")
    (tmp_path / "main.anvil").write_text('use "lib.anvil"\nuse "lib.anvil"\nnet = Tiny()\nprint(double(WIDTH), net([[1.0, 2.0]]).shape)\n')
    out, _, _ = run_text((tmp_path / "main.anvil").read_text(), path=str(tmp_path / "main.anvil"))
    assert out == "6 (1, 3)\n"
    (tmp_path / "bad.anvil").write_text("x = 1\n")
    with pytest.raises(AnvilError, match="may only declare"):
        compile_text('use "bad.anvil"\n', path=str(tmp_path / "m.anvil"))
    (tmp_path / "a.anvil").write_text('use "b.anvil"\n')
    (tmp_path / "b.anvil").write_text('use "a.anvil"\n')
    with pytest.raises(AnvilError, match="uses itself"):
        compile_text('use "a.anvil"\n', path=str(tmp_path / "m.anvil"))
    with pytest.raises(AnvilError, match="cannot read"):
        compile_text('use "missing.anvil"\n', path=str(tmp_path / "m.anvil"))
    with pytest.raises(AnvilError, match="top level"):
        compile_text('fn f(x):\n    use "lib.anvil"\n    return x\ny = f(1)\n', path=str(tmp_path / "m.anvil"))


def test_warnings_for_parameters_that_cannot_learn():
    c = compile_text("""
x: [8, 3] ~ normal(0, 1)
param w: [3] ~ normal(0, 1)
param unused: [3] ~ normal(0, 1)
param frozen: [3] ~ normal(0, 1)
for step in range(2):
    loss = mean((x @ w) ** 2) + sum(detach(frozen))
    minimize loss with sgd(lr=0.1)
""")
    msgs = [w.message for w in c.warnings]
    assert any("does not depend on `unused`, `frozen`" in m for m in msgs), msgs
    assert any("`unused` is a parameter, but no `minimize` trains it" in m for m in msgs), msgs
    assert not any("`w`" in m for m in msgs)
    clean = compile_text("param w: [3] ~ normal(0, 1)\nfor s in range(2):\n    loss = sum(w * w)\n"
                         "    minimize loss with sgd(lr=0.1)\n")
    assert clean.warnings == []


def test_compile_time_if_in_a_model():
    src = """
const DEPTH = 1
model Net:
    a = Linear(4, 4)
    if DEPTH == 2:
        b = Linear(4, 4)
    fn forward(x):
        h = a(x)
        if DEPTH == 2:
            h = b(h)
        return h
net = Net()
print(net(ones(1, 4)).shape)
"""
    for depth, params in [(1, 2), (2, 4)]:
        c = compile_text(src, consts={"DEPTH": depth})
        assert len([b for b in c.program.buffers if b.kind == "param"]) == params
    with pytest.raises(AnvilError, match="known at compile time"):
        compile_text("model M:\n    if sum(rand(2)) > 1:\n        a = Linear(2, 2)\nm = M()\n")
