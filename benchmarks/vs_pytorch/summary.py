"""Summary statistics of results.json (and kernels.json): the numbers REPORT.md quotes.

    python3 benchmarks/vs_pytorch/summary.py
"""
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def steady(epochs):
    return statistics.median(epochs[1:]) if len(epochs) > 1 else epochs[0]


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    with open(os.path.join(HERE, "results.json")) as f:
        data = json.load(f)
    runs = [r for r in data["results"] if "error" not in r]
    projects = list(dict.fromkeys(r["project"] for r in runs))
    configs = list(dict.fromkeys(r["config"] for r in runs))

    def med(p, c, f):
        vals = [f(r) for r in runs if r["project"] == p and r["config"] == c]
        return statistics.median(vals) if vals else None

    print("machine:", data["machine"])
    print(f"runs: {len(runs)}, {data['minutes']:.0f} minutes; load average {min(r['load'] for r in runs):.1f}–"
          f"{max(r['load'] for r in runs):.1f} (median {statistics.median(r['load'] for r in runs):.1f})")
    print("\nrun-to-run spread of the steady epoch time (max/min over runs), worst per config:")
    for c in configs:
        spreads = []
        for p in projects:
            ts = [steady(r["epochs"]) for r in runs if r["project"] == p and r["config"] == c]
            if len(ts) > 1:
                spreads.append((max(ts) / min(ts), p))
        if spreads:
            print(f"  {c:14s} worst {max(spreads)[0]:.3f} ({max(spreads)[1]}), median {statistics.median(s for s, _ in spreads):.3f}")

    pairs = [("anvil", "torch"), ("anvil-1t", "torch-1t"), ("anvil", "torch-compile"), ("anvil-metal", "torch-mps"),
             ("anvil", "torch-mps"), ("anvil-metal", "torch"), ("anvil", "anvil-1t"), ("torch", "torch-1t"), ("anvil", "anvil-metal")]
    print("\nspeed-ups (steady epoch time of the second over the first; >1: the first is faster)")
    for a, b in pairs:
        if a not in configs or b not in configs:
            continue
        ratios = []
        for p in projects:
            x, y = med(p, a, lambda r: steady(r["epochs"])), med(p, b, lambda r: steady(r["epochs"]))
            if x and y:
                ratios.append((y / x, p))
        if ratios:
            print(f"  {a:10s} vs {b:14s} geomean {geomean([r for r, _ in ratios]):6.2f}×   "
                  f"min {min(ratios)[0]:.2f}× ({min(ratios)[1]})   max {max(ratios)[0]:.2f}× ({max(ratios)[1]})")
            print("     " + "  ".join(f"{p} {r:.2f}" for r, p in ratios))

    print("\ntotal training time, all epochs (median run), and end-to-end wall:")
    for p in projects:
        print(f"  {p:11s} " + "  ".join(f"{c} {med(p, c, lambda r: r['train']):.2f}/{med(p, c, lambda r: r['wall']):.2f}"
                                       for c in configs if med(p, c, lambda r: r['train']) is not None))
    print("\npeak memory (MB):")
    for p in projects:
        print(f"  {p:11s} " + "  ".join(f"{c} {med(p, c, lambda r: r['rss_mb']):.0f}" for c in configs
                                       if med(p, c, lambda r: r['rss_mb']) is not None))
    print("\nresults (metric) per config:")
    for p in projects:
        print(f"  {p:11s} " + "  ".join(f"{c} {med(p, c, lambda r: r['metric']):.4f}" for c in configs
                                       if med(p, c, lambda r: r['metric']) is not None))
    print("\nfirst epoch over steady epoch (warm-up cost):")
    for p in projects:
        print(f"  {p:11s} " + "  ".join(
            f"{c} {med(p, c, lambda r: r['epochs'][0] - steady(r['epochs'])):+.2f}s" for c in configs
            if med(p, c, lambda r: r['epochs'][0]) is not None))
    print("\nAnvil compile times:", {p: {k: round(v, 2) for k, v in b.items()} for p, b in data["builds"].items()})
    errors = [r for r in data["results"] if "error" in r]
    if errors:
        print("\nfailures:", [(r["project"], r["config"], r["error"]) for r in errors])

    kpath = os.path.join(HERE, "kernels.json")
    if os.path.exists(kpath):
        with open(kpath) as f:
            k = json.load(f)
        print("\nkernels: Anvil vs eager geomean", f"{geomean([r['torch'] / r['anvil'] for r in k['rows']]):.2f}×",
              " vs compile", f"{geomean([r['compile'] / r['anvil'] for r in k['rows']]):.2f}×",
              " vs MPS", f"{geomean([r['mps'] / r['anvil'] for r in k['rows'] if 'mps' in r]):.2f}×")
        for r in k["rows"]:
            extra = ""
            if "flops" in r:
                extra = f"   Anvil {r['flops'] / r['anvil'] / 1e9:.0f} GFLOP/s, eager {r['flops'] / r['torch'] / 1e9:.0f}, MPS {r['flops'] / r.get('mps', float('inf')) / 1e9:.0f}"
            if "bytes" in r:
                extra = f"   Anvil {r['bytes'] / r['anvil'] / 1e9:.0f} GB/s, eager {r['bytes'] / r['torch'] / 1e9:.0f}"
            print(f"  {r['name']:42s}{extra}")


if __name__ == "__main__":
    main()
