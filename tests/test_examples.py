"""The example programs compile, run natively, and agree with the interpreter."""
import os
import re

import pytest

from util import ROOT, check_native, run_native

FAST = ["hello", "linear_regression", "spirals", "attention"]


@pytest.mark.parametrize("name", FAST)
def test_example_native_matches_interpreter(name):
    path = os.path.join(ROOT, "examples", f"{name}.anvil")
    check_native(open(path).read(), path=path)


@pytest.mark.slow
def test_mnist_accuracy():
    from util import run_native
    path = os.path.join(ROOT, "examples", "mnist.anvil")
    out, err, code, _ = run_native(open(path).read(), path=path)
    assert code == 0, err
    last = out.strip().splitlines()[-1]
    acc = float(last.split("test accuracy")[1].split("%")[0])
    assert acc >= 96.0, out


def test_snake_native_matches_interpreter():
    """A few games of the DQN snake (with the replay drawn): item assignment, runtime
    literals, randint, show, nested runtime control flow and minimize all at once."""
    path = os.path.join(ROOT, "examples", "snake.anvil")
    src = open(path).read()
    src = src.replace("const EPISODES = 300 ", "const EPISODES = 4 ")
    src = src.replace("sleep(0.05)", "sleep(0.0)")
    assert "const EPISODES = 4 " in src
    out = check_native(src, path=path)
    assert "watching the trained snake" in out and "score" in out


def test_checkers_native_matches_interpreter():
    """A few self-play games, test matches with a two-move search, the game on screen, and a game
    against "you" from scripted input: nonzero, stack, compile-time recursion, input and string
    lists, against the interpreter."""
    path = os.path.join(ROOT, "examples", "checkers.anvil")
    src = open(path).read()
    for old, new in [("sleep(0.15)", "sleep(0.0)"), ("{clock() - start:.1f}s", "{0 * start:.1f}s")]:
        assert old in src, old
        src = src.replace(old, new)
    # nowhere to keep the critic: both runs learn (and warn that they cannot save it)
    consts = {"GAMES": 3, "MATCH": 2, "MAX_PLY": 30, "PLAY": 1, "PLAY_DEPTH": 2, "SAVE": "/nonexistent/c.weights"}
    out = check_native(src, path=path, stdin="3\n9\nx\n1\n1\n", consts=consts)
    assert "3 games   vs random:" in out and "a b c d e f g h" in out
    assert "1: a3-b4  2: c3-b4  3: c3-d4" in out                # the seven opening moves
    assert "a number from 1 to" in out and "the critic plays" in out


def test_checkers_plays_the_saved_critic(tmp_path):
    """`bin/checkers`: the first run learns and saves the critic, the next one loads it."""
    path = os.path.join(ROOT, "examples", "checkers.anvil")
    src = open(path).read()
    save = str(tmp_path / "critic.weights")
    first, err, code, _ = run_native(src, path=path, consts={"GAMES": 5, "MATCH": 2, "WATCH": 0, "SAVE": save})
    assert code == 0, err
    assert "5 games   vs random" in first and os.path.getsize(save) == 16 + 8 * 4 + 4 * (224 * 64 + 64 + 64 + 1)
    second, err, code, _ = run_native(src, path=path, stdin="1\n",
                                      consts={"PLAY": 1, "WATCH": 0, "PLAY_DEPTH": 2, "SAVE": save})
    assert code == 0, err
    assert "from an earlier run" in second and "games   vs random" not in second
    assert "the critic plays" in second


def test_charrnn_native_matches_interpreter():
    """A few steps of the GRU language model, unrolled with `static for`, then a short sample."""
    path = os.path.join(ROOT, "examples", "charrnn.anvil")
    src = open(path).read().replace("{clock() - start:.1f}s", "{0 * start:.1f}s")
    out = check_native(src, path=path, consts={"STEPS": 500, "SEQ": 4, "BATCH": 8, "HIDDEN": 16, "SAMPLE": 40})
    assert "step   500" in out and "bits per byte" in out


def test_transformer_native_matches_interpreter():
    """A few steps of the GPT: multi-head causal attention, layer norm, gelu, windows, sampling."""
    path = os.path.join(ROOT, "examples", "transformer.anvil")
    src = open(path).read().replace("{clock() - start:.1f}s", "{0 * start:.1f}s")
    src = src.replace("step % 250 == 249", "step % 5 == 4")
    out = check_native(src, path=path, consts={"STEPS": 20, "BATCH": 4, "T": 8, "D": 16, "HEADS": 2, "SAMPLE": 10})
    assert "step    20" in out


def test_vae_runs():
    """One epoch of the VAE (reparameterized sampling, bce, KL), then its imagined digits."""
    path = os.path.join(ROOT, "examples", "vae.anvil")
    out, err, code, _ = run_native(open(path).read(), path=path, consts={"EPOCHS": 1})
    assert code == 0, err
    recon = float(out.split("reconstruction")[1].split()[0])
    assert recon < 160, out                       # an untrained decoder scores about 540
    assert "digits it imagined" in out and "@" in out or "%" in out


def test_diffusion_native_matches_interpreter():
    """A tiny DDPM: the noise schedule as products, Embedding of noise levels and labels, label
    dropout with bernoulli, a run-time loop over noise levels with guidance, and the drawing."""
    path = os.path.join(ROOT, "examples", "diffusion.anvil")
    src = open(path).read().replace("{clock() - start:.1f}s", "{0 * start:.1f}s")
    out = check_native(src, path=path, consts={"EPOCHS": 1, "T": 8, "H": 16, "BATCH": 3000, "PER_DIGIT": 1})
    assert "epoch  1" in out and "digits it drew" in out


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "examples", "digits.weights")),
                    reason="needs examples/digits.weights (bin/draw trains it)")
def test_dream_fools_the_network():
    """examples/dream.anvil: gradients with respect to the input image. The dreams must convince the
    network, and changing each pixel by at most 0.2 must make it misread most of the ten digits."""
    out, err, code, _ = run_native(open(os.path.join(ROOT, "examples", "dream.anvil")).read(),
                                   path=os.path.join(ROOT, "examples", "dream.anvil"),
                                   consts={"DREAM_STEPS": 150})
    assert code == 0, err
    fooled = int(out.split("% now read as")[0].split()[-1])
    assert fooled >= 80, out
    sure = [float(x) for x in re.findall(r"\d: ([\d.]+)%", out.split("sure it is:")[1].splitlines()[0])]
    assert len(sure) == 10 and min(sure) > 50, out
