"""examples/handwriting: the diffusion model that draws characters, and the page it writes them on."""
import os
import sys

import numpy as np
import pytest

from util import ROOT, compile_text



def test_the_model_trains_and_draws():
    """The U-Net, a training step and the sampler compile (the trained weights are not needed)."""
    src = f'''use "{os.path.join(ROOT, "examples", "handwriting_model.anvil")}"
net = UNet()
x: [2, 1, 28, 28] ~ normal(0, 1)
t: i32[2] = 3
c: i32[2] = 5
minimize mean((net(x, t, c) - x) ** 2) with adam(lr=1e-3)
labels: i32[3] = 10
noise: [T + 1, 3, 1, 28, 28] ~ normal(0, 1)
drawn, frames = draw(net, labels, noise, 2.0)
'''
    c = compile_text(src, path=os.path.join(ROOT, "examples", "handwriting", "t.anvil"))
    params = sum(b.numel for b in c.program.buffers if b.kind == "param")
    assert 1_000_000 < params < 1_300_000


def test_the_page():
    """Characters are set by size (capitals tall, small letters small, g below the line), in order,
    and wrapped at the width; punctuation is drawn with a pen."""
    pytest.importorskip("PIL")
    import importlib.util                      # (examples/synth has a write.py too: load this one by path)
    spec = importlib.util.spec_from_file_location("handwriting_write", os.path.join(ROOT, "examples", "handwriting", "write.py"))
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    rng = np.random.default_rng(0)
    text = "Hello, big gap!"
    chars = [c for c in text if c in W.CLASSES]
    glyphs = [np.pad(np.ones((20, 12)), 2) for _ in chars]
    lines = W.layout(text, [12 / 20] * len(chars), rng, width=10_000)
    assert len(lines) == 1
    placed = {p["ch"]: p for p in lines[0] if p["ch"] != "l"}
    assert placed["H"]["h"] > 1.4 * placed["e"]["h"] and placed["g"]["depth"] > 0 == placed["b"]["depth"]
    xs = [p["x"] for p in lines[0]]
    assert xs == sorted(xs)
    assert len(W.layout(" ".join(["word"] * 40), [0.6] * 160, rng, width=600)) > 3
    page = W.compose(lines, glyphs, 0.1, 5.0)
    assert page.shape[0] > W.H and 0 < page.mean() < 0.5
