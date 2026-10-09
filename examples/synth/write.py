"""Give examples; watch the diffusion model write a Pipes program for them, run it, and fix it.

    python3 examples/synth/write.py "[3, 1, 2] -> [1, 2, 3]" "[9, 4, 7, 5] -> [4, 5, 7, 9]"
    python3 examples/synth/write.py "[1, 2, 3, 4] -> 12" "[6, 5, 8] -> 28" "[1, 3, 5] -> 0"
    python3 examples/synth/write.py --test 12          (the 12th test task, with 4 hidden examples)

Inputs are lists of 3 to 6 digits; outputs are lists of up to 6 numbers, or a number, between -99
and 99 (1 to 4 examples). The model writes 64 programs (search.py), runs them on the examples, and
edits those that get some wrong: it hides a part of the program again and writes that part anew.
Shown is how the simplest program that fits came about: each line a writing that got more examples
right, ░ the tokens being written.
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "synth")
sys.path.insert(0, HERE)
import pipes as P  # noqa: E402
import search as S  # noqa: E402

DIM, GREEN, RED, YELLOW, OFF = "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def layout(final, step) -> str:
    """A program being written, laid out as `final` ends up: each word at its final place, ░ until
    it is written (step[i] is MASK)."""
    words = [P.VOCAB[t] for t in final if t != P.PAD]
    gaps = ["" if i == 0 or w == ")" or words[i - 1] == "(" or w == "(" else " " for i, w in enumerate(words)]
    line, k = "", 0
    for i, t in enumerate(final):
        if t == P.PAD:
            line += DIM + "░" + OFF if step[i] == P.MASK else ""
            continue
        w = words[k]
        line += gaps[k] + (w if step[i] != P.MASK else DIM + "░" * len(w) + OFF)
        k += 1
    return "x |> " + line


def right(program, examples) -> int:
    return sum(program is not None and P.run(program, xs) == ys for xs, ys in examples)


def tag(program, examples) -> str:
    n = right(program, examples)
    if program is None:
        return f"{RED}✗ not a program{OFF}"
    color = GREEN if n == len(examples) else YELLOW if n else RED
    return f"{color}{'✓' if n == len(examples) else '✗'} {n} of {len(examples)}{OFF}"


def show(line: str, pause: float):
    sys.stdout.write("\r\033[K  " + line)
    sys.stdout.flush()
    time.sleep(pause)


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
    if not examples or len(examples) > P.N_SHOWN:
        sys.exit(__doc__)
    for xs, ys in examples:
        if not (3 <= len(xs) <= P.LIST_LEN and all(0 <= a <= 9 for a in xs) and P.fits(ys)):
            sys.exit(f"{P.show_value(xs)} -> {P.show_value(ys)}: inputs are 3 to 6 digits, outputs up to 6 numbers in -99…99")
    for xs, ys in examples:
        print(f"  {P.show_value(xs)} -> {P.show_value(ys)}")
    print()

    pool = S.Search(batch=64, seed=random.randrange(1 << 30)).run(examples, tries=64, rounds=12)
    best = S.first_fit(pool, len(examples))
    before, shown = None, []
    for entry in best.history:                      # the writings that got more examples right
        if not shown or right(P.parse(P.decode(entry[1][-1])), examples) > right(P.parse(P.decode(shown[-1][1][-1])), examples):
            shown.append(entry)
    if shown[-1] is not best.history[-1]:
        shown.append(best.history[-1])
    for hidden_ids, steps, _ in shown:
        final = [int(t) for t in steps[-1]]
        if before is not None:                      # the part that is written anew, hidden
            show(layout(before, [P.MASK if h == P.MASK else t for h, t in zip(hidden_ids, before)]), 0.5)
        for step in steps:
            show(layout(final, [int(t) for t in step]), 0.07)
        prog = P.parse(P.decode(final))
        print("   " + tag(prog, examples))
        before = final
    ok = right(best.program, examples) == len(examples)
    if hidden and ok:
        n = right(best.program, hidden)
        print(f"  {GREEN if n == len(hidden) else RED}and {n} of the {len(hidden)} examples it did not see{OFF}")
    others = sorted({P.show(c.program) for c in pool if c.program and right(c.program, examples) == len(examples)}
                    - {P.show(best.program) if best.program else ""})
    if others:
        print(f"{DIM}  also fits: " + "\n             ".join(others) + OFF)
    if ok:
        xs = [random.randint(0, 9) for _ in range(random.randint(3, 6))]
        print(f"\n  on {P.show_value(xs)} it gives {P.show_value(P.run(best.program, xs))}")


if __name__ == "__main__":
    main()
