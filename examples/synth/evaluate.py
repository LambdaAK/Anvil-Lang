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


if __name__ == "__main__":
    main()
