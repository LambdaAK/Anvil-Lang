"""Give examples; watch the diffusion model write a Pipes program for them.

    python3 examples/synth/write.py "[3, 1, 2] -> [1, 2, 3]" "[9, 4, 7, 5] -> [4, 5, 7, 9]"
    python3 examples/synth/write.py "[1, 2, 3, 4] -> 4" "[5, 6, 7] -> 6"
    python3 examples/synth/write.py --test 12          (the 12th test task)

Inputs are lists of 3 to 6 digits; outputs are lists of up to 6 numbers, or a number, between -99
and 99. The model always sees four examples (fewer are repeated). It writes 33 programs: the
likeliest, and 32 samples. The one shown being written is the first that fits every example.
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "examples", "data", "synth")
sys.path.insert(0, HERE)
import pipes as P  # noqa: E402

DIM, GREEN, RED, BOLD, OFF = "\033[2m", "\033[32m", "\033[31m", "\033[1m", "\033[0m"


def frames(steps) -> list[str]:
    """The program being written, step by step, laid out as it ends up: each word at its final place,
    shaded (░) until the model fixes it, so that words appear in the order the model chose them."""
    final = [int(t) for t in steps[-1]]
    words = [P.VOCAB[t] for t in final if t != P.PAD]
    gaps = ["" if i == 0 or w == ")" or words[i - 1] == "(" or w == "(" else " " for i, w in enumerate(words)]
    out = []
    for step in steps:
        line, k = "", 0
        for i, t in enumerate(final):
            if t == P.PAD:
                line += DIM + "░" + OFF if int(step[i]) == P.MASK else ""
                continue
            w = words[k]
            line += gaps[k] + (w if int(step[i]) != P.MASK else DIM + "░" * len(w) + OFF)
            k += 1
        out.append("x |> " + line)
    return out


def fits(program, examples) -> bool:
    return program is not None and all(P.run(program, xs) == ys for xs, ys in examples)


def main():
    args = sys.argv[1:]
    hidden = []
    if args[:1] == ["--test"]:
        task = [json.loads(line) for line in open(os.path.join(DATA, "test.jsonl"))][int(args[1])]
        examples = [(xs, ys) for xs, ys in task["examples"][:P.N_SHOWN]]
        hidden = [(xs, ys) for xs, ys in task["examples"][P.N_SHOWN:]]
        print(f"{DIM}test task {args[1]} (its program: {task['program']}){OFF}")
    else:
        examples = [P.parse_example(a) for a in args]
    if not examples:
        sys.exit(__doc__)
    for xs, ys in examples:
        if not (3 <= len(xs) <= P.LIST_LEN and all(0 <= a <= 9 for a in xs) and P.fits(ys)):
            sys.exit(f"{P.show_value(xs)} -> {P.show_value(ys)}: inputs are 3 to 6 digits, outputs up to 6 numbers in -99…99")
    for xs, ys in examples:
        print(f"  {P.show_value(xs)} -> {P.show_value(ys)}")
    spec = [examples[i % len(examples)] for i in range(P.N_SHOWN)]
    np.save(os.path.join(DATA, "ask.npy"), np.array([P.encode_spec(spec)], dtype=np.int32))
    subprocess.run([os.path.join(ROOT, "bin", "anvil"), "run", os.path.join(HERE, "write.anvil")], check=True)
    best = np.load(os.path.join(DATA, "ask_best.npy"))[:, 0]        # [step, token]
    tries = np.load(os.path.join(DATA, "ask_tries.npy"))            # [step, try, token]
    runs = [best] + [tries[:, k] for k in range(tries.shape[1])]
    programs = [P.parse(P.decode(r[-1])) for r in runs]
    good = [k for k, p in enumerate(programs) if fits(p, examples)]
    pick = good[0] if good else 0

    print()
    for line in frames(runs[pick]):
        sys.stdout.write("\r\033[K  " + line)
        sys.stdout.flush()
        time.sleep(0.12)
    ok = pick in good
    print(f"   {GREEN + '✓ fits every example' if ok else RED + '✗ does not fit the examples'}{OFF}")
    if hidden and ok:
        right = fits(programs[pick], hidden)
        print(f"  {GREEN if right else RED}{'✓' if right else '✗'} and the 4 examples it did not see{OFF}")
    others = sorted({P.show(programs[k]) for k in good} - {P.show(programs[pick])})
    if others:
        print(f"{DIM}  also fits: " + "\n             ".join(others) + OFF)
    if ok:
        xs = [random.randint(0, 9) for _ in range(random.randint(3, 6))]
        print(f"\n  on {P.show_value(xs)} it gives {P.show_value(P.run(programs[pick], xs))}")


if __name__ == "__main__":
    main()
