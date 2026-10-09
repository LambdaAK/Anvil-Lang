"""examples/synth: the Pipes language, its tokens, and the diffusion model that writes it."""
import os
import random
import sys

from util import ROOT, compile_text

sys.path.insert(0, os.path.join(ROOT, "examples", "synth"))
import pipes as P  # noqa: E402


def test_programs_survive_tokens_and_text():
    rng = random.Random(0)
    for _ in range(2000):
        program, pairs = P.random_task(rng)
        ids = P.encode_program(program)
        assert len(ids) == P.PROGRAM and P.parse(P.decode(ids)) == program
        assert len(P.encode_spec(pairs[:P.N_SHOWN])) == P.SPEC
        assert all(P.run(program, xs) == ys for xs, ys in pairs)


def test_the_language():
    prog = P.parse("filter ( x % 2 == 1 ) |> map ( 3 * x + 1 ) |> sum".split())
    assert P.show(prog) == "x |> filter(x % 2 == 1) |> map(3 * x + 1) |> sum"
    assert P.run(prog, [1, 2, 3, 4, 5]) == 4 + 10 + 16
    assert P.run(P.parse("cumsum |> diff".split()), [1, 2, 3]) == [2, 3]
    assert P.run(P.parse("map ( - x ) |> cummax".split()), [3, 1, 2]) == [-3, -1, -1]
    assert P.run(P.parse("unique |> rotate ( 1 )".split()), [4, 4, 5, 6]) == [5, 6, 4]
    assert P.run(P.parse(["max"]), []) is None
    for bad in ["sum |> sort", "map ( 9 * x )", "take ( 0 )", "|> sort", "sort |>", "filter ( x > )", "map ( x )", ""]:
        assert P.parse(bad.split()) is None, bad
    lists, numbers = P.all_stages()
    assert len(set(lists)) == len(lists) == 279 and len(numbers) == 6
    assert all(P.parse(P.stage_tokens(st)) == [st] for st in lists + numbers)
    assert P.parse_example("[3, 1, 2] -> [1, 2, 3]") == ([3, 1, 2], [1, 2, 3])
    assert P.parse_example("[3, 1] -> 4") == ([3, 1], 4)


def test_the_model_compiles():
    """The model and its sampler, on a made-up task (the trained weights are not needed)."""
    src = f'''use "{os.path.join(ROOT, "examples", "synth_model.anvil")}"
net = Writer()
spec: i32[2, SPEC] = PAD
progs, steps = write(net, spec, 1.0)
'''
    c = compile_text(src, path=os.path.join(ROOT, "examples", "synth", "t.anvil"))
    assert len(P.VOCAB) == c.elab.globals.vars["VOCAB"].val.value


def test_the_search_scores_edits_and_climbs():
    """search.py: partial credit for nearly right outputs, edits that hide part of a program, and an
    exact search that repairs a program one change away from fitting."""
    import numpy as np
    import search as S
    ex = [([1, 2, 3], [2, 4, 6]), ([5, 0, 4], [10, 0, 8])]
    assert S.score(P.parse("map ( 2 * x )".split()), ex) == 2.0
    wrong_length = S.score(P.parse("take ( 2 )".split()), ex)               # [1, 2], [5, 0]: nothing right
    right_length = S.score(P.parse("map ( 3 * x )".split()), ex)            # [3, 6, 9]: the right length
    half_right = S.score(P.parse("map ( 2 * x ) |> take ( 2 )".split()), ex)  # [2, 4]: the first two right
    assert 0 <= wrong_length < right_length < half_right < 2
    assert S.score(None, ex) == -1.0
    assert S.closeness(7, 6) > S.closeness(9, 6) > S.closeness([6], 6) == 0.0
    rng = np.random.default_rng(0)
    ids = P.encode_program(P.parse("filter ( x % 2 == 1 ) |> map ( 3 * x + 1 ) |> sum".split()))
    for _ in range(200):
        e = S.edit(ids, rng)
        assert len(e) == len(ids) and P.MASK in e
        assert all(a == b for a, b in zip(e, ids) if a != P.MASK)        # only hides, never changes
    rng = random.Random(3)
    for _ in range(40):                                                   # one stage wrong: repaired
        program, pairs = P.random_task(rng)
        wrong = list(program)
        i = rng.randrange(len(wrong))
        wrong[i] = ("map", ("affine", 5, 9)) if wrong[i] != ("map", ("affine", 5, 9)) else ("sort", None)
        if program[-1][0] in P.NUMBER_STAGES and i == len(wrong) - 1:
            wrong[i] = ("len", None) if program[-1][0] != "len" else ("sum", None)
        c = S.Candidate(P.encode_program(wrong), wrong, S.score(wrong, pairs[:4]))
        S.climb(c, pairs[:4])
        assert c.score >= 4, (P.show(program), P.show(wrong), P.show(c.program))
