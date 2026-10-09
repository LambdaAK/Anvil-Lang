"""How good are the programs synth.anvil wrote for the test tasks?

    python3 examples/synth/evaluate.py

For each test task the model saw four examples. A program is right if it also gives the right
output on the task's four hidden examples (it found the function, not just something that fits).
With several tries, the program kept is the first that fits the four examples it was shown, which
can be checked without the hidden ones: write, run, keep what works.
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "synth")
sys.path.insert(0, HERE)
import pipes as P  # noqa: E402


def as_value(v):
    return v if isinstance(v, int) else list(v)


def fits(program, examples) -> bool:
    return program is not None and all(P.run(program, xs) == as_value(ys) for xs, ys in examples)


def main():
    best = np.load(os.path.join(DATA, "best.npy"))
    tries = np.load(os.path.join(DATA, "tries.npy"))
    n = len(best)
    k = len(tries) // n
    tasks = [json.loads(line) for line in open(os.path.join(DATA, "test.jsonl"))][:n]

    count = collections.Counter()
    by_len = collections.defaultdict(lambda: [0, 0])
    shown_examples = []
    for i, task in enumerate(tasks):
        shown, hidden = task["examples"][:P.N_SHOWN], task["examples"][P.N_SHOWN:]
        candidates = [P.parse(P.decode(best[i]))] + [P.parse(P.decode(tries[j * n + i])) for j in range(k)]
        first = candidates[0]
        count["valid"] += first is not None
        count["fits"] += fits(first, shown)
        count["right"] += fits(first, shown + hidden)
        count["same"] += first is not None and P.show(first) == task["program"]
        kept = next((c for c in candidates if fits(c, shown)), None)
        count["kept right"] += fits(kept, shown + hidden)
        count["any right"] += any(fits(c, shown + hidden) for c in candidates)
        count["valid tries"] += sum(c is not None for c in candidates[1:])
        stages = task["program"].count("|>")
        by_len[stages][0] += 1
        by_len[stages][1] += fits(kept, shown + hidden)
        if len(shown_examples) < 8 and i % 7 == 0:
            shown_examples.append((task, kept or first, fits(kept, shown + hidden)))

    pct = lambda c: f"{100 * c / n:5.1f}%"
    print(f"{n} test tasks, programs never seen in training\n")
    print("the likeliest program (one try):")
    print(f"  is a program                 {pct(count['valid'])}")
    print(f"  fits the 4 examples shown    {pct(count['fits'])}")
    print(f"  right on the 4 hidden ones   {pct(count['right'])}")
    print(f"  the very program of the task {pct(count['same'])}")
    print(f"\n{k + 1} tries, keeping the first that fits the examples shown:")
    print(f"  right on the 4 hidden ones   {pct(count['kept right'])}")
    print(f"  (one of the {k + 1} is right   {pct(count['any right'])};"
          f" {100 * count['valid tries'] / (n * k):.1f}% of the sampled tries are programs)")
    print("\nright, by the task's number of stages:")
    for stages in sorted(by_len):
        tot, ok = by_len[stages]
        print(f"  {stages}: {100 * ok / tot:5.1f}% of {tot}")
    print()
    for task, program, ok in shown_examples:
        ex = "   ".join(f"{P.show_value(a)} -> {P.show_value(b)}" for a, b in task["examples"][:2])
        print(f"  {ex}   ...")
        print(f"    task's program:  {task['program']}")
        print(f"    model's program: {P.show(program) if program else '(not a program)'}   {'✓' if ok else '✗'}")


def search(n: int, tries: int = 8, rounds: int = 8):
    """The same tasks, written by search.py: the likeliest program alone; the shortest of `tries`
    programs that fits the examples shown; with the exact search around the closest; and with `rounds`
    rounds of write, run, fix as well."""
    import time
    import search as S
    tasks = [json.loads(line) for line in open(os.path.join(DATA, "test.jsonl"))][:n]
    s = S.Search(batch=256)
    shown = [[(xs, ys) for xs, ys in t["examples"][:P.N_SHOWN]] for t in tasks]
    print(f"{n} test tasks, programs never seen in training; right = also right on the 4 hidden examples")
    for what, r, climbs in ((f"{tries} tries, the shortest that fits", 0, 0),
                            ("... and the exact search around the closest", 0, 4),
                            (f"... and {rounds} rounds of write, run, fix", rounds, 4)):
        start = time.time()
        pools = s.run_many(shown, tries=tries, rounds=r, climbs=climbs)
        took = (time.time() - start) / n
        if not r and not climbs:
            right = sum(fits(pool[0].program, t["examples"]) for pool, t in zip(pools, tasks))
            print(f"  {'the likeliest program':48s} {100 * right / n:5.1f}%")
        right = sum(fits(S.first_fit(pool, P.N_SHOWN).program, t["examples"]) for pool, t in zip(pools, tasks))
        print(f"  {what:48s} {100 * right / n:5.1f}%   ({took:.2f} s a task)")
    # the hybrid: programs of up to two stages by enumeration (instant), the model for the rest
    start = time.time()
    short = [enumerate_task(t["examples"][:P.N_SHOWN], 2)[0] for t in tasks]
    rest = [i for i, p in enumerate(short) if p is None]
    pools = s.run_many([shown[i] for i in rest], tries=tries, rounds=rounds, climbs=4) if rest else []
    chosen = dict(zip(rest, (S.first_fit(pool, P.N_SHOWN).program for pool in pools)))
    right = sum(fits(short[i] if short[i] is not None else chosen[i], t["examples"]) for i, t in enumerate(tasks))
    took = (time.time() - start) / n
    print(f"  {'up to 2 stages by enumeration, then all of that':48s} {100 * right / n:5.1f}%   ({took:.2f} s a task)")


def enumerate_task(shown, max_stages: int = 3, budget: float = 0.0):
    """No model: every program, the fewest stages first, until one fits the examples shown (the
    simplest explanation of them), as an enumerative synthesizer does. A partial program whose outputs
    on the examples are the same as a shorter one's is not extended (it cannot lead anywhere new).
    Returns (program or None, whether it ran out of time)."""
    import time
    t0 = time.time()
    lists, numbers = P.all_stages()
    want = [as_value(ys) for _, ys in shown]
    level = [([], [list(xs) for xs, _ in shown])]
    seen = {repr(level[0][1])}
    for depth in range(1, max_stages + 1):
        nxt = []
        for prog, vals in level:
            for st in lists + numbers:
                out = [P.apply(st, v) for v in vals]
                if any(o is None for o in out):
                    continue
                if out == want:
                    return prog + [st], False
                if depth < max_stages and st[0] not in P.NUMBER_STAGES:
                    key = repr(out)
                    if key not in seen:
                        seen.add(key)
                        nxt.append((prog + [st], out))
            if budget and time.time() - t0 > budget:
                return None, True
        level = nxt
    return None, False


def enumerate_baseline(n: int, max_stages: int = 3, budget: float = 0.0):
    import time
    tasks = [json.loads(line) for line in open(os.path.join(DATA, "test.jsonl"))][:n]
    start, right, found, timeouts = time.time(), 0, 0, 0
    for t in tasks:
        best, out_of_time = enumerate_task(t["examples"][:P.N_SHOWN], max_stages, budget)
        found += best is not None
        timeouts += out_of_time
        right += best is not None and fits(best, t["examples"])
    limit = f", at most {budget:g} s a task" if budget else ""
    print(f"every program of up to {max_stages} stages{limit}, the shortest that fits: right {100 * right / n:5.1f}%"
          f"   (one found for {100 * found / n:.1f}%, out of time for {100 * timeouts / n:.1f}%;"
          f" {(time.time() - start) / n:.2f} s a task)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--enumerate"]:
        args = sys.argv[2:] + [None] * 3
        enumerate_baseline(int(args[0] or 500), int(args[1] or 3), float(args[2] or 0))
        sys.exit()
    if sys.argv[1:2] == ["--search"]:
        search(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        main()
