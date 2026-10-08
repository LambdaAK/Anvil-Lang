"""Named dimensions (anvil/dims.py): every integer `const` names a dimension, shapes carry the names
through arithmetic, reshapes, slices and index notation, and two dimensions that must agree must be
the same dimension, not just the same number. A program that would fail with other sizes does not
compile with these."""
import os
import subprocess

import pytest

from util import ROOT, compile_text

from anvil.diagnostics import AnvilError
from anvil.dims import DIM_NAMES, Dim, Sym, show
from anvil.ide import analyze, shape_sheet

MISMATCHES = {
    "index notation": ("""
const T = 4
const D = 4
P: [T, D] ~ normal(0, 1)
x: [T, D] ~ normal(0, 1)
y[t, d] = x[t, d] + P[d, t]
""", "ranges over `T` in one place but `D` in another"),
    "broadcasting": ("""
const T = 4
const D = 4
a: [T] ~ normal(0, 1)
b: [D] ~ normal(0, 1)
c = a + b
""", "dimension 0 is `T` on one side but `D` on the other"),
    "matmul": ("""
const T = 4
const D = 4
x: [2, T] ~ normal(0, 1)
W: [D, 3] ~ normal(0, 1)
y = x @ W
""", "inner dimensions must be the same dimension"),
    "a function's shape variable": ("""
const T = 4
const D = 4
fn f(x: [n], y: [n]) = x * y
a: [T] ~ normal(0, 1)
b: [D] ~ normal(0, 1)
c = f(a, b)
""", "`n` is `T` from an earlier argument but `D`"),
    "a function's fixed dimension": ("""
const T = 4
const D = 4
fn g(x: [T]) = sum(x)
b: [D] ~ normal(0, 1)
c = g(b)
""", "should be [T] but has shape [D]"),
    "an annotation": ("""
const T = 4
const D = 4
x: [D, T] ~ normal(0, 1)
y: [T, D] = x
""", "declared as f32[T, D] but the value has shape [D, T]"),
    "a loop-carried variable": ("""
const T = 4
const D = 4
h: [T] ~ normal(0, 1)
c: [D] ~ normal(0, 1)
for k in range(3):
    h = c * 2
""", "changes dimensions inside the loop: [T] → [D]"),
}


@pytest.mark.parametrize("site", list(MISMATCHES))
def test_equal_sizes_with_different_names_do_not_compile(site):
    src, says = MISMATCHES[site]
    with pytest.raises(AnvilError) as e:
        compile_text(src)
    text = e.value.render()
    assert says in text, text
    assert "both 4 here, but they are different dimensions" in text, text


def test_same_size_on_purpose_and_unnamed_sizes_compile():
    compile_text("const T = 4\nconst D = T\na: [T] ~ normal(0, 1)\nb: [D] ~ normal(0, 1)\nc = a + b\n")
    compile_text("const T = 4\na: [T] ~ normal(0, 1)\nb: [4] ~ normal(0, 1)\nc = a + b\n")


def test_the_transformers_swapped_positions_do_not_compile():
    """In examples/transformer.anvil T = D = 64, so `P[d, t]` for `P[t, d]` used to compile and train
    silently worse."""
    path = os.path.join(ROOT, "examples", "transformer.anvil")
    src = open(path).read()
    good = "e[b, t, d] = E[codes[b, t], d] + P[t, d]"
    assert good in src
    compile_text(src, path=path)
    with pytest.raises(AnvilError) as e:
        compile_text(src.replace(good, "e[b, t, d] = E[codes[b, t], d] + P[d, t]"), path=path)
    assert "ranges over `T` in one place but `D` in another" in e.value.render()


def test_formulas():
    DIM_NAMES.set({})                       # (the names of derived sizes belong to the latest compilation)
    T, D, H = Dim(64, Sym.atom("T")), Dim(64, Sym.atom("D")), Dim(4, Sym.atom("HEADS"))
    assert show(T + 1) == "T + 1" and show(2 * D) == "2*D" and show(T * D) == "D*T"
    assert show(D // H) == "D/HEADS" and show((T * D) // (D // H)) == "HEADS*T"
    assert show(T - T + 5) == "5" and not isinstance(T - T, Dim)       # a formula with no names is a number
    assert D // 3 == 21 and not isinstance(D // 3, Dim)                # an inexact division keeps only the number
    assert T == D and hash(T) == hash(64) and f"{T}" == "64" and repr(T) == "64"


def test_names_flow_through_the_program(tmp_path):
    p = tmp_path / "prog.anvil"
    p.write_text("""const BATCH = 8
const T = 16
const D = 32
const HEADS = 4
const DH = D // HEADS
x: [BATCH, T + 1] ~ normal(0, 1)
codes = x[:, 0:T]
h: [BATCH, T, D] ~ normal(0, 1)
q = h.reshape(-1, T, HEADS, DH)
o = q.reshape(-1, T, D)
s[b, a, i, j] = sum q[b, i, a, e] * q[b, j, a, e]
l = Linear(D, 4 * D)
u = l(h)
images: [N, 28, 28] = zeros(100, 28, 28)
flat = images.reshape(-1, 784)
""".replace("[N, 28, 28]", "[100, 28, 28]"))
    r = analyze(str(p))
    assert r["diagnostics"] == []
    text = {h["range"][0]: h["text"] for h in r["hovers"] if h["definition"]}
    assert text[6] == "f32[BATCH, T] = [8, 16]"                       # a slice 0:T has length T
    assert text[8] == "f32[BATCH, T, HEADS, DH] = [8, 16, 4, 8]"      # DH = D/HEADS, shown by its name
    assert text[9] == "f32[BATCH, T, D] = [8, 16, 32]"                # -1 is BATCH, not a number
    assert text[10] == "f32[BATCH, HEADS, T, T] = [8, 4, 16, 16]"     # index notation
    assert text[12] == "f32[BATCH, T, 4*D] = [8, 16, 128]"
    assert text[4] == "8 = D/HEADS (compile-time constant, i32)"
    calls = [h["text"] for h in r["hovers"] if "called here as" in h["text"]]
    assert any("forward(x: f32[BATCH, T, D]) -> f32[BATCH, T, 4*D]" in c for c in calls), calls


def test_an_annotation_names_unnamed_dimensions(tmp_path):
    p = tmp_path / "prog.anvil"
    p.write_text("const N = 100\nimages: [N, 28, 28] = zeros(100, 28, 28)\nflat = images.reshape(N, 784)\n")
    r = analyze(str(p))
    text = {h["range"][0]: h["text"] for h in r["hovers"] if h["definition"]}
    assert text[1] == "f32[N, 28, 28] = [100, 28, 28]" and text[2] == "f32[N, 784] = [100, 784]"


def test_shape_sheet():
    out = shape_sheet(os.path.join(ROOT, "examples", "transformer.anvil"))
    assert "DH = D/HEADS = 16" in out.splitlines()[0]
    assert any(line.split()[1:3] == ["q", "f32[BATCH,"] for line in out.splitlines() if line.strip()), out
    r = subprocess.run([os.path.join(ROOT, "bin", "anvil"), "shapes", os.path.join(ROOT, "examples", "transformer.anvil")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "f32[BATCH, HEADS, T, T]" in r.stdout


def test_the_named_dims_example():
    """examples/named_dims.anvil trains; each of its three bugs (one changed line, each of which runs
    with unnamed sizes) is refused at compile time."""
    path = os.path.join(ROOT, "examples", "named_dims.anvil")
    src = open(path).read()
    compile_text(src, path=path)
    expect = {1: "`ROWS` and `COLS` are both 28", 2: "`D` and `BATCH` are both 64", 3: "`D` and `BATCH` are both 64"}
    for bug, says in expect.items():
        with pytest.raises(AnvilError) as e:
            compile_text(src, path=path, consts={"BUG": bug})
        assert says in e.value.render(), e.value.render()
