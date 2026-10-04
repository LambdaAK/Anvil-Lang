"""Anvil against PyTorch: the same small projects (anvil/*.anvil, torch/*.py), timed on this machine.

    python3 benchmarks/vs_pytorch/run.py                 # everything, 3 runs each (about 40 minutes)
    python3 benchmarks/vs_pytorch/run.py --quick         # tiny settings: does everything run?
    python3 benchmarks/vs_pytorch/run.py mlp cnn         # some projects
    python3 benchmarks/vs_pytorch/run.py --configs anvil,torch --runs 5

Every program prints `@epoch <seconds>` after each epoch (or chunk of steps): the training time
alone, with the GPU's work finished, and data loading and evaluation left out. Then `@metric`, the
accuracy or loss it reached, to check that both sides learned the same thing. Around that, this
records each process's wall time and peak memory (`/usr/bin/time -l`), and for Anvil the compile
time (`anvil build`, separately). The configurations of a project run round-robin, so a burst of
load from other programs hits all of them alike; the load average is recorded with each run.

Results go to benchmarks/vs_pytorch/results.json (every run) and results.md (the tables).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ANVIL = os.path.join(ROOT, "bin", "anvil")
TORCH_PY = os.path.join(ROOT, "benchmarks", ".venv-torch", "bin", "python")
BUILD = os.path.join(HERE, "build")

PROJECTS = ["logreg", "mlp", "wide_mlp", "cnn", "emnist_cnn", "vae", "charrnn", "gpt", "spirals"]
CONFIGS = {
    "anvil": "Anvil, CPU (default threads)",
    "anvil-1t": "Anvil, CPU, 1 thread",
    "anvil-metal": "Anvil, Metal GPU",
    "torch": "PyTorch eager, CPU (default threads)",
    "torch-1t": "PyTorch eager, CPU, 1 thread",
    "torch-compile": "PyTorch + torch.compile, CPU",
    "torch-mps": "PyTorch eager, MPS GPU",
}
DEFAULT_CONFIGS = ["anvil", "anvil-1t", "anvil-metal", "torch", "torch-1t", "torch-compile", "torch-mps"]
# --quick: settings small enough to check that every program runs (the names are each program's consts)
QUICK = {"logreg": {"EPOCHS": 1}, "mlp": {"EPOCHS": 1}, "wide_mlp": {"EPOCHS": 1}, "cnn": {"EPOCHS": 1},
         "emnist_cnn": {"EPOCHS": 1}, "vae": {"EPOCHS": 1}, "charrnn": {"STEPS": 40, "CHUNK": 20},
         "gpt": {"STEPS": 40, "CHUNK": 20}, "spirals": {"EPOCHS": 4, "CHUNK": 2}}


def build_anvil(project, metal, consts):
    """Compile the Anvil program; returns (executable, seconds)."""
    os.makedirs(BUILD, exist_ok=True)
    exe = os.path.join(BUILD, project + ("-metal" if metal else "") + ("-quick" if consts else ""))
    cmd = [ANVIL, "build", os.path.join(HERE, "anvil", project + ".anvil"), "-o", exe]
    if metal:
        cmd.insert(2, "--metal")
    for k, v in consts.items():
        cmd += ["--set", f"{k}={v}"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"anvil build failed for {project}:\n{r.stderr}")
    return exe, time.perf_counter() - t


def command(config, project, exes, consts):
    env = dict(os.environ)
    threads = re.fullmatch(r"(?:anvil|torch)-(\d+)t", config)        # anvil-4t, torch-8t: that many threads
    if config.startswith("anvil"):
        cmd = [exes["metal" if config == "anvil-metal" else "cpu"]]
        if threads:
            env["ANVIL_THREADS"] = threads.group(1)
    else:
        cmd = [TORCH_PY, os.path.join(HERE, "torch", project + ".py")]
        if threads:
            cmd += ["--threads", threads.group(1)]
        if config == "torch-compile":
            cmd += ["--compile"]
        if config == "torch-mps":
            cmd += ["--device", "mps"]
        for k, v in consts.items():
            cmd += [f"--{k}", str(v)]
    return cmd, env


def run_once(config, project, exes, consts):
    cmd, env = command(config, project, exes, consts)
    load = os.getloadavg()[0]
    t = time.perf_counter()
    r = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True, env=env,
                       cwd=os.path.join(HERE, "torch") if config.startswith("torch") else HERE)
    wall = time.perf_counter() - t
    if r.returncode != 0:
        raise RuntimeError(f"{config} {project} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    epochs = [float(x) for x in re.findall(r"^@epoch ([\d.eE+-]+)", r.stdout, re.M)]
    metric = re.search(r"^@metric ([\d.eE+-]+|nan|inf)", r.stdout, re.M)
    rss = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    return {"config": config, "project": project, "wall": wall, "epochs": epochs,
            "train": sum(epochs), "metric": float(metric.group(1)) if metric else None,
            "rss_mb": int(rss.group(1)) / 2**20 if rss else None, "load": load}


def steady(epochs):
    """The time of a typical epoch after the first (which pays for warm-up and torch.compile)."""
    return statistics.median(epochs[1:]) if len(epochs) > 1 else epochs[0]


def summarize(results, builds, configs, projects):
    """Markdown tables of the medians over runs."""
    def cell(p, c, f):
        runs = [r for r in results if r["project"] == p and r["config"] == c and "error" not in r]
        vals = [f(r) for r in runs if f(r) is not None]
        return statistics.median(vals) if vals else None

    out = []
    head = "| project | " + " | ".join(configs) + " |"
    sep = "|---|" + "---|" * len(configs)
    tables = [
        ("Steady-state time per epoch (s; the median epoch after the first)", lambda r: steady(r["epochs"]), "{:.3f}"),
        ("Total training time (s; all epochs, the first included)", lambda r: r["train"], "{:.2f}"),
        ("First epoch (s; includes warm-up and torch.compile's compilation)", lambda r: r["epochs"][0], "{:.3f}"),
        ("Process wall time (s; start-up, data loading, training and evaluation; Anvil's compile time not included)",
         lambda r: r["wall"], "{:.2f}"),
        ("Peak memory (MB)", lambda r: r["rss_mb"], "{:.0f}"),
        ("Result (test accuracy, or loss / bits per byte: lower is better for vae, charrnn, gpt)",
         lambda r: r["metric"], "{:.4f}"),
    ]
    for title, f, fmt in tables:
        out += [f"### {title}", "", head, sep]
        for p in projects:
            vals = [cell(p, c, f) for c in configs]
            out.append(f"| {p} | " + " | ".join("—" if v is None else fmt.format(v) for v in vals) + " |")
        out.append("")
    if "anvil" in configs and "torch" in configs:
        out += ["### Speed-up of Anvil over PyTorch (steady-state epoch time; >1 means Anvil is faster)", "",
                "| project | CPU (default threads) | CPU, 1 thread | CPU vs torch.compile | GPU (Metal vs MPS) |",
                "|---|---|---|---|---|"]
        for p in projects:
            def ratio(a, b):
                x, y = cell(p, a, lambda r: steady(r["epochs"])), cell(p, b, lambda r: steady(r["epochs"]))
                return "—" if x is None or y is None else f"{y / x:.2f}×"
            out.append(f"| {p} | {ratio('anvil', 'torch')} | {ratio('anvil-1t', 'torch-1t')} | "
                       f"{ratio('anvil', 'torch-compile')} | {ratio('anvil-metal', 'torch-mps')} |")
        out.append("")
    if builds:
        out += ["### Anvil compile time (s; `anvil build`, source to executable)", "", "| project | CPU | Metal |", "|---|---|---|"]
        for p in projects:
            b = builds.get(p, {})
            out.append(f"| {p} | " + " | ".join(f"{b[k]:.2f}" if k in b else "—" for k in ("cpu", "metal")) + " |")
        out.append("")
    return "\n".join(out)


def machine():
    def sysctl(name):
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    torch_version = subprocess.run([TORCH_PY, "-c", "import torch; print(torch.__version__, torch.get_num_threads())"],
                                   capture_output=True, text=True).stdout.split()
    return {"cpu": sysctl("machdep.cpu.brand_string"), "p_cores": sysctl("hw.perflevel0.physicalcpu"),
            "e_cores": sysctl("hw.perflevel1.physicalcpu"), "memory_gb": int(sysctl("hw.memsize")) / 2**30,
            "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "torch": torch_version[0] if torch_version else None,
            "torch_threads": torch_version[1] if len(torch_version) > 1 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    projects = [p for p in PROJECTS if not args.projects or p in args.projects]
    configs = args.configs.split(",")
    subprocess.run([sys.executable, os.path.join(HERE, "prepare.py")], check=True, capture_output=True)

    results, builds = [], {}
    started = time.time()
    for p in projects:
        consts = QUICK[p] if args.quick else {}
        exes = {}
        builds[p] = {}
        failed = {}
        cpu_configs = {c for c in configs if c.startswith("anvil") and c != "anvil-metal"}
        for target, wanted in (("cpu", cpu_configs), ("metal", {"anvil-metal"})):
            if wanted & set(configs):
                try:
                    exes[target], builds[p][target] = build_anvil(p, target == "metal", consts)
                except RuntimeError as e:                  # recorded, and the other configurations go on
                    for c in wanted:
                        failed[c] = str(e).strip().splitlines()[-1]
                    print(f"{p:11s} {target}: {failed[next(iter(wanted))]}", flush=True)
        for c, why in failed.items():
            results.append({"config": c, "project": p, "error": why})
        for run in range(args.runs):
            for c in configs:
                if c in failed:
                    continue
                r = run_once(c, p, exes, consts)
                r["run"] = run
                results.append(r)
                ep = steady(r["epochs"]) if r["epochs"] else float("nan")
                print(f"{p:11s} {c:14s} run {run + 1}: epoch {ep:8.4f}s  train {r['train']:7.2f}s  wall {r['wall']:6.2f}s  "
                      f"metric {r['metric']}  {r['rss_mb'] or 0:5.0f} MB  load {r['load']:.1f}", flush=True)
        with open(args.out + ".json", "w") as f:
            json.dump({"machine": machine(), "configs": {c: CONFIGS.get(c, c) for c in configs}, "builds": builds,
                       "results": results, "quick": args.quick, "minutes": (time.time() - started) / 60}, f, indent=1)
    table = summarize(results, builds, configs, projects)
    with open(args.out + ".md", "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
