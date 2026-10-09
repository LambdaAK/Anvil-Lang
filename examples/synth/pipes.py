"""Pipes: the little language that examples/synth.anvil writes programs in.

A program takes a list of digits x and sends it through up to four stages:

    x |> filter(odd) |> map(* 3) |> sort
    x |> drop(1) |> sum

Stages:
    map(+ k)  map(- k)  map(* k)        k = 1 … 4            every element
    filter(> k)  filter(< k)            k = 0 … 9            the elements that pass
    filter(even)  filter(odd)
    sort  reverse  cumsum                                     (cumsum: running totals)
    take(k)  drop(k)                    k = 1 … 4            the first k, or all but them
    sum  max  min  len  head  last                            a number (only as the last stage)

This file has the language (run, show), its tokens, and the tasks the model learns from: a random
program and eight lists run through it. The model sees four of the eight (input and output) and
writes a program; the other four check it.
"""
from __future__ import annotations

import random

LIST_STAGES = ["map", "filter", "sort", "reverse", "cumsum", "take", "drop"]
NUMBER_STAGES = ["sum", "max", "min", "len", "head", "last"]
MAX_STAGES = 4
LO, HI = -99, 99                     # every value a task shows lies in here

# ------------------------------------------------------------------ tokens
SPECIAL = ["[PAD]", "[MASK]", "[IN]", "[OUT]", "[NUM]"]
WORDS = ["|>", "map", "filter", "sort", "reverse", "cumsum", "take", "drop", "sum", "max", "min", "len",
         "head", "last", "(", ")", "+", "-", "*", ">", "<", "even", "odd"]
VOCAB = SPECIAL + WORDS + [str(v) for v in range(LO, HI + 1)]
ID = {t: i for i, t in enumerate(VOCAB)}
PAD, MASK, IN, OUT, NUM = (ID[t] for t in SPECIAL)

N_SHOWN, N_HIDDEN = 4, 4             # examples the model sees; examples that check its program
LIST_LEN = 6                         # the longest input (and output) list
EXAMPLE = 2 + 2 * LIST_LEN           # [IN] 6 values [OUT] 6 values   (or [NUM] and one value)
SPEC = N_SHOWN * EXAMPLE             # 56 tokens of examples
PROGRAM = 24                         # tokens of program (4 stages of up to 5 tokens, and 3 `|>`)
SEQ = SPEC + PROGRAM


def run(program: list, xs: list):
    """The program's result on the list xs: a list, a number, or None if a stage fails (max of
    nothing)."""
    v = list(xs)
    for name, arg in program:
        if name == "map":
            op, k = arg
            v = [a + k if op == "+" else a - k if op == "-" else a * k for a in v]
        elif name == "filter":
            v = [a for a in v if (a % 2 == 0 if arg == "even" else a % 2 == 1 if arg == "odd"
                                  else a > arg[1] if arg[0] == ">" else a < arg[1])]
        elif name == "sort":
            v = sorted(v)
        elif name == "reverse":
            v = v[::-1]
        elif name == "cumsum":
            out, t = [], 0
            for a in v:
                t += a
                out.append(t)
            v = out
        elif name == "take":
            v = v[:arg]
        elif name == "drop":
            v = v[arg:]
        else:                                           # a number
            if name == "len":
                return len(v)
            if not v:
                return None
            return {"sum": sum, "max": max, "min": min, "head": lambda a: a[0], "last": lambda a: a[-1]}[name](v)
    return v


def show(program: list) -> str:
    parts = ["x"]
    for name, arg in program:
        if arg is None:
            parts.append(name)
        elif isinstance(arg, tuple):
            parts.append(f"{name}({arg[0]} {arg[1]})")
        else:
            parts.append(f"{name}({arg})")
    return " |> ".join(parts)


def tokens(program: list) -> list[str]:
    out = []
    for i, (name, arg) in enumerate(program):
        if i:
            out.append("|>")
        out.append(name)
        if isinstance(arg, tuple):
            out += ["(", arg[0], str(arg[1]), ")"]
        elif arg is not None:
            out += ["(", str(arg), ")"]
    return out


def parse(toks: list[str]):
    """Tokens (without padding) back to a program; None if they are not one."""
    program, i = [], 0
    while i < len(toks):
        if program:
            if toks[i] != "|>":
                return None
            i += 1
        if i >= len(toks):
            return None
        name = toks[i]
        i += 1
        if program and program[-1][0] in NUMBER_STAGES:
            return None                                 # nothing can follow a number
        if name in ("sort", "reverse", "cumsum") or name in NUMBER_STAGES:
            program.append((name, None))
            continue
        if name not in ("map", "filter", "take", "drop") or i >= len(toks) or toks[i] != "(":
            return None
        i += 1
        try:
            if name == "map" and toks[i] in "+-*" and 1 <= int(toks[i + 1]) <= 4 and toks[i + 2] == ")":
                program.append((name, (toks[i], int(toks[i + 1]))))
                i += 3
            elif name == "filter" and toks[i] in ("even", "odd") and toks[i + 1] == ")":
                program.append((name, toks[i]))
                i += 2
            elif name == "filter" and toks[i] in "<>" and 0 <= int(toks[i + 1]) <= 9 and toks[i + 2] == ")":
                program.append((name, (toks[i], int(toks[i + 1]))))
                i += 3
            elif name in ("take", "drop") and 1 <= int(toks[i]) <= 4 and toks[i + 1] == ")":
                program.append((name, int(toks[i])))
                i += 2
            else:
                return None
        except (IndexError, ValueError):
            return None
    return program if 1 <= len(program) <= MAX_STAGES else None


def decode(ids) -> list[str]:
    """Program token ids to tokens, up to the first padding."""
    out = []
    for i in ids:
        if int(i) == PAD:
            break
        out.append(VOCAB[int(i)])
    return out


# ------------------------------------------------------------------ random programs and tasks
def random_stage(rng: random.Random, last: bool):
    if last and rng.random() < 0.35:
        return (rng.choice(NUMBER_STAGES), None)
    name = rng.choice(LIST_STAGES)
    if name == "map":
        return (name, (rng.choice("+-*"), rng.randint(1, 4)))
    if name == "filter":
        r = rng.random()
        return (name, rng.choice(["even", "odd"])) if r < 0.3 else (name, (rng.choice("<>"), rng.randint(0, 9)))
    if name in ("take", "drop"):
        return (name, rng.randint(1, 4))
    return (name, None)


def random_program(rng: random.Random) -> list:
    n = rng.choice([1, 2, 2, 3, 3, 4])
    return [random_stage(rng, k == n - 1) for k in range(n)]


def random_list(rng: random.Random) -> list:
    return [rng.randint(0, 9) for _ in range(rng.randint(3, LIST_LEN))]


def fits(v) -> bool:
    if v is None:
        return False
    if isinstance(v, int):
        return LO <= v <= HI
    return len(v) <= LIST_LEN and all(LO <= a <= HI for a in v)


def random_task(rng: random.Random):
    """(program, [(input, output)] * 8), with every output defined and in range, and not all the same."""
    while True:
        program = random_program(rng)
        pairs = []
        for _ in range(N_SHOWN + N_HIDDEN):
            xs = random_list(rng)
            ys = run(program, xs)
            if not fits(ys):
                break
            pairs.append((xs, ys))
        else:
            if len({str(y) for _, y in pairs}) > 1:
                return program, pairs


def encode_example(xs, ys) -> list[int]:
    out = [IN] + [ID[str(a)] for a in xs] + [PAD] * (LIST_LEN - len(xs))
    if isinstance(ys, int):
        return out + [NUM, ID[str(ys)]] + [PAD] * (LIST_LEN - 1)
    return out + [OUT] + [ID[str(a)] for a in ys] + [PAD] * (LIST_LEN - len(ys))


def encode_spec(pairs) -> list[int]:
    return [t for xs, ys in pairs for t in encode_example(xs, ys)]


def encode_program(program) -> list[int]:
    ids = [ID[t] for t in tokens(program)]
    return ids + [PAD] * (PROGRAM - len(ids))


def parse_example(text: str):
    """'[3, 1, 2] -> [1, 2, 3]' or '[3, 1, 2] -> 6' as (input, output)."""
    left, right = text.split("->")
    xs = [int(a) for a in left.strip().strip("[]").split(",") if a.strip()]
    right = right.strip()
    ys = [int(a) for a in right.strip("[]").split(",") if a.strip()] if right.startswith("[") else int(right)
    return xs, ys


def show_value(v) -> str:
    return "[" + ", ".join(str(a) for a in v) + "]" if isinstance(v, list) else str(v)
