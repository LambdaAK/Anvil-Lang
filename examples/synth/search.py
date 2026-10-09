"""Write, run, fix: the diffusion model as an editor, in a search for a program that fits the examples.

The model writes a few programs. Each is run on the examples (pipes.py). A program that fits only some
of them is edited: a part of it is hidden again and the model writes that part anew, seeing the rest:

    one stage           x |> filter(odd) |> ░░░(░ ░) |> sum       (the same length)
    the numbers         x |> filter(odd) |> map(░ ░) |> sum
    the rest            x |> filter(odd) |> ░░░░░░░░░░░░░░░░░░    (from a stage on, any length)
    all of it           (now and then: a fresh start)

An edit is kept if the program fits at least as many examples as before. Masked diffusion can do this
because it fills in hidden tokens anywhere, with the tokens on both sides in view; a left-to-right
model can only rewrite the end.

The model runs through anvil.function (compiled once, with the trained weights baked in, and called on
NumPy arrays). Search(...).run(examples) returns the result and the history of the edits.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import pipes as P  # noqa: E402

WEIGHTS = os.environ.get("SYNTH_WEIGHTS", os.path.join(ROOT, "examples", "synth.weights"))
SOURCE = f'''use "{os.path.join(ROOT, "examples", "synth_model.anvil")}"
net = Writer()
found = load(net, "{WEIGHTS}")
fn sample(seq, noise) = fill(net, seq, noise, 1.0)
fn likeliest(seq, noise) = fill(net, seq, noise, 0.0, true)
'''


def closeness(got, want) -> float:
    """1 for the right output; less for one that is nearly right (a number off by a little, a list with
    some elements right), so that the search can tell a better wrong program from a worse one."""
    want = want if isinstance(want, int) else list(want)
    if got == want:
        return 1.0
    if got is None or isinstance(got, int) != isinstance(want, int):
        return 0.0
    if isinstance(want, int):
        return 0.5 / (1 + abs(got - want))
    n = max(len(got), len(want), 1)
    return 0.6 * sum(a == b for a, b in zip(got, want)) / n + 0.2 * (len(got) == len(want))


def score(program, examples) -> float:
    """The program's closeness on the examples: as many as it has right, plus partial credit (-1: not a
    program)."""
    if program is None:
        return -1.0
    return sum(closeness(P.run(program, xs), ys) for xs, ys in examples)


@dataclass
class Candidate:
    ids: list                                       # PROGRAM token ids
    program: object
    score: int
    history: list = field(default_factory=list)     # (what was hidden, steps of the fill, score after)


class Search:
    def __init__(self, batch: int = 64, seed: int = 0):
        import anvil
        self.batch = batch                          # every call is this many programs (one compilation)
        self.rng = np.random.default_rng(seed)
        self.sample_fn = anvil.function(SOURCE, name="sample")
        self.likeliest_fn = anvil.function(SOURCE, name="likeliest")

    def fill(self, seqs: np.ndarray, likeliest: bool = False):
        """Fill the masks of many sequences, `batch` at a time: (programs, steps[n][step][token])."""
        progs, steps = [], []
        for i in range(0, len(seqs), self.batch):
            part = seqs[i:i + self.batch]
            pad = np.repeat(part[-1:], self.batch - len(part), axis=0)
            full = np.concatenate([part, pad]).astype(np.int32)
            noise = self.rng.uniform(1e-6, 1.0, (self.batch, P.PROGRAM, len(P.VOCAB))).astype(np.float32)
            p, s = (self.likeliest_fn if likeliest else self.sample_fn)(full, noise)
            progs.append(p[:len(part)])
            steps.append(np.transpose(s, (1, 0, 2))[:len(part)])
        return np.concatenate(progs), np.concatenate(steps)

    def edit(self, ids: list) -> list:
        return edit(ids, self.rng)

    def run_many(self, tasks: list, tries: int = 8, rounds: int = 8) -> list:
        """Search for a program for each task's examples at once; returns each task's candidates."""
        specs = [P.encode_spec([ex[i % len(ex)] for i in range(P.N_SHOWN)]) for ex in tasks]
        # first, the likeliest program, then tries - 1 samples
        first = np.array([s + [P.MASK] * P.PROGRAM for s in specs], dtype=np.int32)
        best, best_steps = self.fill(first, likeliest=True)
        many = np.repeat(first, tries - 1, axis=0)
        samples, sample_steps = self.fill(many)
        pools = []
        for t, ex in enumerate(tasks):
            pool = []
            rows = [(best[t], best_steps[t])] + [(samples[t * (tries - 1) + j], sample_steps[t * (tries - 1) + j])
                                                 for j in range(tries - 1)]
            for ids, steps in rows:
                prog = P.parse(P.decode(ids))
                c = Candidate(list(map(int, ids)), prog, score(prog, ex))
                c.history.append(([P.MASK] * P.PROGRAM, steps, c.score))
                pool.append(c)
            pools.append(pool)
        for _ in range(rounds):
            todo = [(t, c) for t, pool in enumerate(pools) if max(c.score for c in pool) < len(tasks[t])
                    for c in pool]
            if not todo:
                break
            edits = [self.edit(c.ids) for _, c in todo]
            seqs = np.array([specs[t] + e for (t, _), e in zip(todo, edits)], dtype=np.int32)
            progs, steps = self.fill(seqs)
            for (t, c), hidden, ids, st in zip(todo, edits, progs, steps):
                prog = P.parse(P.decode(ids))
                s = score(prog, tasks[t])
                ids = list(map(int, ids))
                if s >= c.score and ids != c.ids:
                    c.ids, c.program, c.score = ids, prog, s
                    c.history.append((hidden, st, s))
        return pools

    def run(self, examples: list, tries: int = 8, rounds: int = 12) -> list:
        return self.run_many([examples], tries, rounds)[0]


def edit(ids: list, rng) -> list:
    """A copy of a program with a part hidden again (see the module docstring)."""
    ids = list(ids)
    end = ids.index(P.PAD) if P.PAD in ids else len(ids)
    starts = [0] + [i + 1 for i in range(end) if ids[i] == P.ID["|>"]]
    k = int(rng.integers(len(starts)))
    lo = starts[k]
    hi = (starts[k + 1] - 1) if k + 1 < len(starts) else end           # the stage's tokens
    r = rng.random()
    if r < 0.1:                                                         # all of it: a fresh start
        return [P.MASK] * len(ids)
    if r < 0.45:                                                        # the rest, from this stage on
        for i in range(lo, len(ids)):
            ids[i] = P.MASK
    elif r < 0.75:                                                      # this stage
        for i in range(lo, hi):
            ids[i] = P.MASK
    else:                                                               # its numbers and operators
        hidden = [i for i in range(lo, hi) if P.VOCAB[ids[i]] not in ("(", ")") and i > lo]
        for i in hidden or range(lo, hi):
            ids[i] = P.MASK
    return ids


def first_fit(pool: list, n: int):
    """The shortest candidate that fits all n examples (the simplest explanation is likeliest to be
    the function meant), then the one with the fewest edits; else the closest."""
    fits = [c for c in pool if c.score >= n]
    if fits:
        return min(fits, key=lambda c: (len(P.decode(c.ids)), len(c.history)))
    return max(pool, key=lambda c: c.score)
