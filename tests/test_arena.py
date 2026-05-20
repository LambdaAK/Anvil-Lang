"""Temporaries share memory (anvil/backend/arena.py): the output must not change, and memory goes down."""
import os
import subprocess
import tempfile

from util import ROOT, check_native, compile_text

LIFETIMES = """
x: [64] ~ normal(0, 1)
a = exp(x * 0.5)                  # made before the loop, read in every iteration
total = 0.0
for i in range(6):
    b = a * (i + 1)               # temporaries that die within an iteration
    c = sum(b * b)
    if c > 400:
        d = sqrt(b)
        total = total + sum(d)
    else:
        total = total - mean(b)
    e = tanh(b)
    while sum(e) > 4:             # a loop inside the loop
        e = e * 0.5
    total = total + sum(e)
    if total > 1000:
        break
f[k] = sum a[j + k] * x[j + k] where k < 32, j < 32
print(total, sum(a), sum(f))
"""


def run(text, share):
    from anvil.backend.aarch64 import generate
    from anvil.backend.toolchain import assemble_and_link
    asm, gen = generate(compile_text(text).program, share=share)
    with tempfile.TemporaryDirectory() as d:
        s, exe = os.path.join(d, "p.s"), os.path.join(d, "p")
        with open(s, "w") as f:
            f.write(asm)
        assemble_and_link(s, exe)
        r = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout, gen


def test_sharing_keeps_the_output():
    shared, gen = run(LIFETIMES, True)
    alone, _ = run(LIFETIMES, False)
    assert shared == alone
    assert gen.arena.offsets and gen.arena.size < gen.arena.unshared
    check_native(LIFETIMES)


def test_checkers_temporaries_shrink():
    from anvil.backend.aarch64 import generate
    from anvil.driver import compile_file
    c = compile_file(os.path.join(ROOT, "examples", "checkers.anvil"))
    _, gen = generate(c.program)
    assert gen.arena.size * 4 < gen.arena.unshared, (gen.arena.size, gen.arena.unshared)


def test_lifetimes_never_overlap_in_memory():
    """Independent check of the plan: two temporaries that share bytes are never both alive."""
    from anvil.backend.aarch64 import generate
    from anvil.backend.arena import flatten, lifetime
    from anvil.driver import compile_file
    for name in ("checkers", "attention", "snake"):
        prog = compile_file(os.path.join(ROOT, "examples", f"{name}.anvil")).program
        _, gen = generate(prog)
        uses, loops = flatten(prog)
        spans = {b.id: (lifetime(us, loops), b.nbytes) for b, us in uses.items() if b.id in gen.arena.offsets}
        items = [(gen.arena.offsets[i], gen.arena.offsets[i] + n, span) for i, (span, n) in spans.items()]
        for k, (s1, e1, (lo1, hi1)) in enumerate(items):
            for s2, e2, (lo2, hi2) in items[k + 1:]:
                if s1 < e2 and s2 < e1:
                    assert hi1 < lo2 or hi2 < lo1, name
