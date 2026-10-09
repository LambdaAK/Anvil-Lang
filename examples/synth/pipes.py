"""Pipes: the little language that examples/synth.anvil writes programs in.

A program takes a list of digits x and sends it through up to four stages:

    x |> filter(x % 2 == 1) |> map(3 * x + 1) |> sum
    x |> cumsum |> diff |> rotate(2)

Stages (a program that ends in a number stage gives a number, otherwise a list):

    map(a * x + b)      a = -3 … 5, b = -9 … 9     every element (shown simply: map(x + 3), map(-x))
    map(x % k)          k = 2 … 9
    map(x // k)         k = 2 … 5                  (rounding down)
    map(x * x)
    filter(x > k)       k = -5 … 30                the elements that pass
    filter(x < k)       k = -5 … 30
    filter(x != k)      k = 0 … 9
    filter(x % m == r)  m = 2 … 5, r < m
    sort  reverse  unique  cumsum  cummax  diff     unique: first copies; cummax: running maximum;
                                                    diff: each element minus the one before it
    take(k)  drop(k)    k = 1 … 5                  the first k, or all but them
    rotate(k)           k = 1 … 3                  the first k moved to the end
    sum  max  min  len  head  last                 a number (only as the last stage)

There are about 280 stages, so 22 million programs of up to three stages and 6 billion of up to
four: too many to try them all. (The first version of Pipes had 45 stages, and trying every
program of up to three, about 100,000, found the right one more often than the model did.)

This file has the language (run, show, parse), its tokens, and the tasks the model learns from: a
random program and eight lists run through it. The model sees four of the eight (input and output)
and writes a program; the other four check it.
"""
from __future__ import annotations

import random

LIST_NAMES = ["map", "filter", "sort", "reverse", "unique", "cumsum", "cummax", "diff", "take", "drop", "rotate"]
NUMBER_STAGES = ["sum", "max", "min", "len", "head", "last"]
PLAIN = ["sort", "reverse", "unique", "cumsum", "cummax", "diff"]          # stages without an argument
MAX_STAGES = 4
LO, HI = -99, 99                     # every value a task shows lies in here
SCALES = [1, 2, 3, 4, 5, -1, -2, -3]

# ------------------------------------------------------------------ tokens
SPECIAL = ["[PAD]", "[MASK]", "[IN]", "[OUT]", "[NUM]"]
WORDS = ["|>", "x", "(", ")", "+", "-", "*", "%", "//", ">", "<", "==", "!="] + LIST_NAMES + NUMBER_STAGES
VOCAB = SPECIAL + WORDS + [str(v) for v in range(LO, HI + 1)]
ID = {t: i for i, t in enumerate(VOCAB)}
PAD, MASK, IN, OUT, NUM = (ID[t] for t in SPECIAL)

N_SHOWN, N_HIDDEN = 4, 4             # examples the model sees; examples that check its program
LIST_LEN = 6                         # the longest input (and output) list
EXAMPLE = 2 + 2 * LIST_LEN           # [IN] 6 values [OUT] 6 values   (or [NUM] and one value)
SPEC = N_SHOWN * EXAMPLE             # 56 tokens of examples
PROGRAM = 36                         # tokens of program: 4 stages of up to 8 tokens, and 3 `|>`
SEQ = SPEC + PROGRAM


def apply(stage, v: list):
    """One stage on a list: a list, a number, or None (max of nothing)."""
    name, arg = stage
    if name == "map":
        kind = arg[0]
        if kind == "affine":
            return [arg[1] * a + arg[2] for a in v]
        if kind == "mod":
            return [a % arg[1] for a in v]
        if kind == "div":
            return [a // arg[1] for a in v]
        return [a * a for a in v]
    if name == "filter":
        kind = arg[0]
        if kind == ">":
            return [a for a in v if a > arg[1]]
        if kind == "<":
            return [a for a in v if a < arg[1]]
        if kind == "!=":
            return [a for a in v if a != arg[1]]
        return [a for a in v if a % arg[1] == arg[2]]
    if name == "sort":
        return sorted(v)
    if name == "reverse":
        return v[::-1]
    if name == "unique":
        out = []
        for a in v:
            if a not in out:
                out.append(a)
        return out
    if name in ("cumsum", "cummax"):
        out = []
        for a in v:
            out.append(a if not out else out[-1] + a if name == "cumsum" else max(out[-1], a))
        return out
    if name == "diff":
        return [v[i + 1] - v[i] for i in range(len(v) - 1)]
    if name == "take":
        return v[:arg]
    if name == "drop":
        return v[arg:]
    if name == "rotate":
        k = arg % len(v) if v else 0
        return v[k:] + v[:k]
    if name == "len":
        return len(v)
    if not v:
        return None
    return {"sum": sum, "max": max, "min": min, "head": lambda a: a[0], "last": lambda a: a[-1]}[name](v)


def run(program: list, xs: list):
    """The program's result on the list xs: a list, a number, or None if a stage fails."""
    v = list(xs)
    for stage in program:
        if not isinstance(v, list):
            return None
        v = apply(stage, v)
        if v is None:
            return None
    return v


def stage_tokens(stage) -> list[str]:
    name, arg = stage
    if arg is None:
        return [name]
    if name in ("take", "drop", "rotate"):
        return [name, "(", str(arg), ")"]
    if name == "map":
        kind = arg[0]
        if kind == "affine":
            a, b = arg[1], arg[2]
            body = ["x"] if a == 1 else ["-", "x"] if a == -1 else [str(a), "*", "x"]
            body += ["+", str(b)] if b > 0 else ["-", str(-b)] if b < 0 else []
        elif kind == "square":
            body = ["x", "*", "x"]
        else:
            body = ["x", "%" if kind == "mod" else "//", str(arg[1])]
        return ["map", "("] + body + [")"]
    kind = arg[0]
    body = ["x", "%", str(arg[1]), "==", str(arg[2])] if kind == "mod" else ["x", kind, str(arg[1])]
    return ["filter", "("] + body + [")"]


def tokens(program: list) -> list[str]:
    out = []
    for i, stage in enumerate(program):
        if i:
            out.append("|>")
        out += stage_tokens(stage)
    return out


def show_stage(stage) -> str:
    toks = stage_tokens(stage)
    if len(toks) == 1:
        return toks[0]
    inner = " ".join(toks[2:-1]).replace("- x", "-x")
    return f"{toks[0]}({inner})"


def show(program: list) -> str:
    return " |> ".join(["x"] + [show_stage(s) for s in program])


def all_stages() -> tuple[list, list]:
    """Every list stage, and every number stage."""
    lists = [("map", ("affine", a, b)) for a in SCALES for b in range(-9, 10) if (a, b) != (1, 0)]
    lists += [("map", ("mod", k)) for k in range(2, 10)] + [("map", ("div", k)) for k in range(2, 6)]
    lists += [("map", ("square",))]
    lists += [("filter", (op, k)) for op in "><" for k in range(-5, 31)] + [("filter", ("!=", k)) for k in range(10)]
    lists += [("filter", ("mod", m, r)) for m in range(2, 6) for r in range(m)]
    lists += [(n, None) for n in PLAIN] + [(n, k) for n in ("take", "drop") for k in range(1, 6)]
    lists += [("rotate", k) for k in range(1, 4)]
    return lists, [(n, None) for n in NUMBER_STAGES]


STAGES = {tuple(stage_tokens(s)): s for group in all_stages() for s in group}


def parse(toks: list[str]):
    """Tokens (without padding) back to a program; None if they are not one."""
    program, cur = [], []
    for t in toks + ["|>"]:
        if t != "|>":
            cur.append(t)
            continue
        stage = STAGES.get(tuple(cur))
        if stage is None or (program and program[-1][0] in NUMBER_STAGES):
            return None
        program.append(stage)
        cur = []
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
    if last and rng.random() < 0.3:
        return (rng.choice(NUMBER_STAGES), None)
    r = rng.random()
    if r < 0.35:
        s = rng.random()
        if s < 0.7:
            a, b = 1, 0
            while (a, b) == (1, 0):
                a, b = rng.choice(SCALES), rng.randint(-9, 9)
            return ("map", ("affine", a, b))
        if s < 0.85:
            return ("map", ("mod", rng.randint(2, 9)))
        if s < 0.95:
            return ("map", ("div", rng.randint(2, 5)))
        return ("map", ("square",))
    if r < 0.6:
        s = rng.random()
        if s < 0.7:
            return ("filter", (rng.choice("><"), rng.randint(-5, 30)))
        if s < 0.8:
            return ("filter", ("!=", rng.randint(0, 9)))
        m = rng.randint(2, 5)
        return ("filter", ("mod", m, rng.randrange(m)))
    if r < 0.85:
        return (rng.choice(PLAIN), None)
    name = rng.choice(["take", "drop", "rotate"])
    return (name, rng.randint(1, 3 if name == "rotate" else 5))


def random_program(rng: random.Random) -> list:
    n = rng.choices([1, 2, 3, 4], weights=[15, 30, 30, 25])[0]
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
    """(program, [(input, output)] * 8): every output defined and in range, not all the same, and no
    stage that could be left out without changing an output (no `sort |> sort`, no filter that keeps
    everything), so that a program is about as short as what it does."""
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
            if len({str(y) for _, y in pairs}) == 1:
                continue
            if all(any(run(program[:i] + program[i + 1:], xs) != ys for xs, ys in pairs)
                   for i in range(len(program))):
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
