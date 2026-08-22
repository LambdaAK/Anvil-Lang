"""`anvil cost`: arithmetic and memory traffic counted from the static shapes and loop counts."""
import os
import subprocess
import sys

from util import ROOT, compile_text

from anvil.cost import program_cost


def test_flops_of_a_matmul_in_a_loop():
    c = program_cost(compile_text("""
A: [8, 16] ~ normal(0, 1)
B: [16, 4] ~ normal(0, 1)
s = 0.0
for i in range(10):
    s = s + sum(A @ B)
print(s)
""").program)
    # the matmul: 8·4 outputs × 16 multiply-adds × 2, ten times; the sums and the samples are small
    assert 10 * 8 * 4 * 16 * 2 <= c.flops <= 10 * 8 * 4 * 16 * 2 + 4000
    assert not c.unbounded


def test_run_time_loops_are_lower_bounds():
    c = program_cost(compile_text("x = 1.0\nwhile x < 100:\n    x = x * 2\nprint(x)\n").program)
    assert c.unbounded


def test_cli_report():
    r = subprocess.run([sys.executable, "-m", "anvil", "cost", "mnist.anvil"], capture_output=True, text=True,
                       cwd=os.path.join(ROOT, "examples"), env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    assert "101,770 parameters" in r.stdout and "GFLOP" in r.stdout
    assert "minimize loss with sgd(lr=0.1)" in r.stdout and "4,685" in r.stdout      # 937 steps × 5 epochs
