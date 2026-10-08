"""Editor support: `anvil check --json` (anvil/ide.py) and the VS Code / Cursor extension."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from util import ROOT, compile_text

from anvil.diagnostics import AnvilError
from anvil.ide import analyze


def hover(result, line, col):
    """The innermost hover at a 0-based position."""
    hits = [h for h in result["hovers"] if h["range"][0] <= line <= h["range"][2]
            and (line, col) >= (h["range"][0], h["range"][1]) and (line, col) <= (h["range"][2], h["range"][3])]
    hits.sort(key=lambda h: (h["range"][2] - h["range"][0], h["range"][3] - h["range"][1]))
    return hits[0]["text"] if hits else None


SRC = """const B = 8
x: [B, 4] ~ normal(0, 1)
param W: [4, 3] ~ normal(0, 1)
fn layer(h: [n, d], M) = relu(h @ M)
y = layer(x, W)
z = layer(x[0:2], W)
s[i] = sum y[i, j]
model Net:
    l = Linear(4, 2)
    fn forward(v) = l(v)
net = Net()
out = net(x)
"""


def test_hovers_show_shapes_constants_indices_and_models(tmp_path):
    p = tmp_path / "prog.anvil"
    p.write_text(SRC)
    r = analyze(str(p))
    assert r["diagnostics"] == []
    assert hover(r, 0, 6) == "8 (compile-time constant, i32)"
    # a dimension named by a constant shows its name, then the numbers
    assert hover(r, 1, 0) == "f32[B, 4] = [8, 4]"
    assert hover(r, 2, 6) == "param f32[4, 3]"
    assert hover(r, 4, 0) == "f32[B, 3] = [8, 3]"
    assert hover(r, 4, 4).startswith("fn layer(h: [n, d], M) = relu(h @ M)")
    # a function's parameter shows every shape it is called with
    assert hover(r, 3, 9) == "f32[B, 4] = [8, 4] | f32[2, 4]"
    assert hover(r, 6, 2) == "index i < B (8)" and hover(r, 6, 16) == "index j < 3"
    assert hover(r, 10, 0).startswith("Net (model, 10 parameters)")
    assert "l.W: f32[4, 2]" in hover(r, 10, 0)
    # inlay hints: after definitions of tensors, not where the line states the shape already
    inlays = {h["range"][0]: h["inlay"] for h in r["hovers"] if h.get("inlay")}
    assert inlays[4] == "f32[B, 3]" and inlays[5] == "f32[2, 3]" and inlays[6] == "f32[B]"
    assert 1 not in inlays and 2 not in inlays


def test_errors_point_at_the_users_line(tmp_path):
    """A shape error inside the standard library is reported at the call in this file."""
    p = tmp_path / "bad.anvil"
    p.write_text("x: [8, 4] ~ normal(0, 1)\nl = Linear(5, 2)\ny = l(x)\nq = y + undefined_thing\nw = 1 +\n")
    r = analyze(str(p))
    assert len(r["diagnostics"]) == 1 and r["diagnostics"][0]["severity"] == "error"
    assert "syntax" in r["diagnostics"][0]["message"] or "expected" in r["diagnostics"][0]["message"]
    p.write_text("x: [8, 4] ~ normal(0, 1)\nl = Linear(5, 2)\ny = l(x)\nq = 3 + undefined_thing\n")
    r = analyze(str(p))
    msgs = [(d["range"][0], d["message"]) for d in r["diagnostics"]]
    assert msgs[0][0] == 2 and "@" in msgs[0][1] and "prelude.anvil" in msgs[0][1], msgs
    assert msgs[1][0] == 3 and "undefined_thing" in msgs[1][1], msgs
    assert hover(r, 0, 0) == "f32[8, 4]"            # what compiled before the errors is still described


def test_call_sites_in_error_messages():
    with pytest.raises(AnvilError) as info:
        compile_text("fn f(a) = a @ a\nfn g(b) = f(b)\nx: [2, 3] ~ normal(0, 1)\ny = g(x)\n")
    text = info.value.render()
    assert "in this call to `f`" in text and "in this call to `g`" in text, text


def test_cli_json_from_stdin():
    src = "x: [3, 2] ~ normal(0, 1)\ny = x @ x\n"
    r = subprocess.run([sys.executable, "-m", "anvil", "check", "--json", "--stdin", os.path.join(ROOT, "x.anvil")],
                       input=src, capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
    data = json.loads(r.stdout)
    assert data["diagnostics"][0]["range"][:2] == [1, 4] and "cannot multiply" in data["diagnostics"][0]["label"]


def test_every_example_analyzes_cleanly():
    for name in sorted(os.listdir(os.path.join(ROOT, "examples"))):
        if name.endswith(".anvil"):
            r = analyze(os.path.join(ROOT, "examples", name))
            assert [d for d in r["diagnostics"] if d["severity"] == "error"] == [], name
            assert r["hovers"], name


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_vscode_extension():
    r = subprocess.run(["node", os.path.join(ROOT, "editors", "vscode", "test", "extension.test.js")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "extension ok" in r.stdout, r.stdout + r.stderr
