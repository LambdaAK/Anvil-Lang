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
    prog = P.parse("filter ( odd ) |> map ( * 3 ) |> sum".split())
    assert P.show(prog) == "x |> filter(odd) |> map(* 3) |> sum"
    assert P.run(prog, [1, 2, 3, 4, 5]) == 27
    assert P.run(P.parse("cumsum |> drop ( 1 )".split()), [1, 2, 3]) == [3, 6]
    assert P.run(P.parse(["max"]), []) is None
    for bad in ["sum |> sort", "map ( * 5 )", "take ( 0 )", "|> sort", "sort |>", "filter ( > )", ""]:
        assert P.parse(bad.split()) is None, bad
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


def test_the_search_scores_and_edits():
    """search.py: partial credit for nearly right outputs, and edits that hide part of a program."""
    import numpy as np
    import search as S
    ex = [([1, 2, 3], [2, 4, 6]), ([5, 0, 4], [10, 0, 8])]
    assert S.score(P.parse("map ( * 2 )".split()), ex) == 2.0
    wrong_length = S.score(P.parse("take ( 2 )".split()), ex)               # [1, 2], [5, 0]: nothing right
    right_length = S.score(P.parse("map ( * 3 )".split()), ex)              # [3, 6, 9]: the right length
    half_right = S.score(P.parse("map ( * 2 ) |> take ( 2 )".split()), ex)  # [2, 4]: the first two right
    assert 0 <= wrong_length < right_length < half_right < 2
    assert S.score(None, ex) == -1.0
    assert S.closeness(7, 6) > S.closeness(9, 6) > S.closeness([6], 6) == 0.0
    rng = np.random.default_rng(0)
    ids = P.encode_program(P.parse("filter ( odd ) |> map ( * 3 ) |> sum".split()))
    for _ in range(200):
        e = S.edit(ids, rng)
        assert len(e) == len(ids) and P.MASK in e
        assert all(a == b for a, b in zip(e, ids) if a != P.MASK)        # only hides, never changes
