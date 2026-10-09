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
