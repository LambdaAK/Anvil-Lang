"""Experiments with Anvil: what the compiler's optimizations are worth, and MNIST hyperparameter sweeps.

    python3 experiments/run.py              # everything (a few minutes); writes experiments/results.md
    python3 experiments/run.py ablation     # one experiment: ablation | threads | optimizers | width

Every number comes from compiled Anvil programs run as separate processes: the training programs
time themselves with clock() and print their test accuracy. Sweeps use `--set`, so each
configuration is the same source file compiled with different constants.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANVIL = os.path.join(ROOT, "bin", "anvil")
SWEEP = os.path.join(HERE, "mnist_sweep.anvil")


def run(args, env=None, cwd=HERE) -> str:
    r = subprocess.run([ANVIL, "run", *args], capture_output=True, text=True, cwd=cwd,
                       env=dict(os.environ, **(env or {})))
    if r.returncode != 0:
        raise RuntimeError(f"anvil {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def sweep(**consts):
    """(test accuracy, seconds per epoch, final batch loss) of mnist_sweep.anvil with these constants."""
    sets = []
    for k, v in consts.items():
        sets += ["--set", f"{k}={v}"]
    env = consts.pop("_env", None) if "_env" in consts else None
    out = run([SWEEP, *sets], env=env)
    acc, t, loss = map(float, out.split("result")[1].split())
    return acc, t, loss


def per_epoch(path, flags=(), env=None, consts=None, runs=2) -> float:
    """Seconds per epoch of an example that prints `(12.3s)` after each epoch (best of `runs`)."""
    sets = []
    for k, v in (consts or {}).items():
        sets += ["--set", f"{k}={v}"]
    best = float("inf")
    for _ in range(runs):
        out = run([*flags, path, *sets], env=env, cwd=os.path.dirname(path))
        times = [float(x) for x in re.findall(r"\(([\d.]+)s\)", out)]
        best = min(best, times[-1] / len(times))
    return best


def sweep_time(flags=(), env=None, runs=2, **consts) -> float:
    best = float("inf")
    sets = []
    for k, v in consts.items():
        sets += ["--set", f"{k}={v}"]
    for _ in range(runs):
        out = run([*flags, SWEEP, *sets], env=env)
        best = min(best, float(out.split("result")[1].split()[1]))
    return best


# ----------------------------------------------------------------------------- experiments
def ablation():
    """Seconds per epoch with fusion and the other IR optimizations off (-O0), with one thread,
    and with both. -O0 still vectorizes and tiles each kernel; it only stops merging them."""
    cnn = os.path.join(ROOT, "examples", "cnn.anvil")
    configs = [("everything on", (), None), ("one thread", (), {"ANVIL_THREADS": "1"}),
               ("no IR optimizations (-O0)", ("-O0",), None),
               ("-O0, one thread", ("-O0",), {"ANVIL_THREADS": "1"})]
    rows = []
    for name, flags, env in configs:
        mlp = sweep_time(flags, env, EPOCHS=2)
        wide = sweep_time(flags, env, EPOCHS=1, HIDDEN=1024, DEPTH=2)
        conv = per_epoch(cnn, flags, env, runs=1)
        rows.append((name, mlp, wide, conv))
        print(f"  {name:28s} mlp {mlp:.3f}s  wide {wide:.3f}s  cnn {conv:.3f}s", flush=True)
    base = rows[0]
    lines = ["| configuration | MLP 784-128-10 | MLP 784-1024-1024-10 | LeNet CNN |", "|---|---|---|---|"]
    for name, a, b, c in rows:
        lines.append(f"| {name} | {a:.3f} s ({a / base[1]:.1f}×) | {b:.3f} s ({b / base[2]:.1f}×) | "
                     f"{c:.2f} s ({c / base[3]:.1f}×) |")
    return ("Ablation: seconds per training epoch (and how much slower than with everything on)", lines)


def threads():
    """Seconds per epoch as worker threads are added (ANVIL_THREADS)."""
    rows = []
    for n in (1, 2, 4, 8, 12):
        env = {"ANVIL_THREADS": str(n)}
        small = sweep_time((), env, EPOCHS=2)
        wide = sweep_time((), env, EPOCHS=1, HIDDEN=1024, DEPTH=2)
        rows.append((n, small, wide))
        print(f"  {n:2d} threads   mlp {small:.3f}s   wide {wide:.3f}s", flush=True)
    lines = ["| threads | MLP 784-128-10 | speed-up | MLP 784-1024-1024-10 | speed-up |", "|---|---|---|---|---|"]
    for n, a, b in rows:
        lines.append(f"| {n} | {a:.3f} s | {rows[0][1] / a:.2f}× | {b:.3f} s | {rows[0][2] / b:.2f}× |")
    return ("Thread scaling: seconds per epoch", lines)


def optimizers():
    """Test accuracy after 3 epochs for each optimizer and learning rate (784-128-10, batch 64)."""
    grid = {"sgd": [0.01, 0.03, 0.1, 0.3, 1.0], "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "rmsprop": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}
    lines = ["| optimizer | " + " | ".join(f"lr #{i + 1}" for i in range(5)) + " |", "|---|---|---|---|---|---|"]
    best = (0, "")
    for opt, lrs in grid.items():
        cells = []
        for lr in lrs:
            acc, t, loss = sweep(OPT=f'"{opt}"', LR=lr, EPOCHS=3)
            cells.append(f"{acc:.2%} (lr {lr:g})")
            best = max(best, (acc, f"{opt}, lr {lr:g}"))
            print(f"  {opt:8s} lr {lr:<7g} acc {acc:.4f}  loss {loss:.4f}", flush=True)
        lines.append(f"| {opt} | " + " | ".join(cells) + " |")
    lines += ["", f"Best: {best[1]} at {best[0]:.2%}."]
    return ("Optimizers and learning rates: test accuracy after 3 epochs", lines)


def width():
    """Test accuracy and time per epoch as the network widens and deepens (Adam, lr 1e-3, 3 epochs)."""
    lines = ["| hidden width | 1 hidden layer | 2 hidden layers |", "|---|---|---|"]
    for h in (32, 64, 128, 256, 512, 1024):
        cells = []
        for depth in (1, 2):
            acc, t, loss = sweep(OPT='"adam"', LR=1e-3, HIDDEN=h, DEPTH=depth, EPOCHS=3)
            cells.append(f"{acc:.2%}, {t:.2f} s/epoch")
            print(f"  width {h:4d} depth {depth}: acc {acc:.4f}  {t:.3f}s/epoch", flush=True)
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    return ("Width and depth: test accuracy after 3 epochs, and seconds per epoch (Adam, lr 1e-3)", lines)


EXPERIMENTS = {"ablation": ablation, "threads": threads, "optimizers": optimizers, "width": width}


def main():
    pick = sys.argv[1:] or list(EXPERIMENTS)
    sections = []
    t0 = time.perf_counter()
    load = os.getloadavg()[0]
    for name in pick:
        print(f"{name}:", flush=True)
        title, lines = EXPERIMENTS[name]()
        sections.append(f"## {title}\n\n" + "\n".join(lines) + "\n")
    if not sys.argv[1:]:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        head = (f"# Experiments\n\nRun with `python3 experiments/run.py` on an {cpu}, {time.strftime('%Y-%m-%d')} "
                f"({time.perf_counter() - t0:.0f} s in all; load average {load:.1f} at the start, "
                f"{os.getloadavg()[0]:.1f} at the end). MNIST: 60,000 training images, accuracy on the "
                f"10,000 test images.\n\n")
        with open(os.path.join(HERE, "results.md"), "w") as f:
            f.write(head + "\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
