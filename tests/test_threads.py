"""Multithreaded kernels give exactly the same results for any number of threads."""
from util import run_native

PROGRAM = """
x: [512, 96] ~ normal(0, 1)
labels = i32(argmax(x @ eye(96)[:, 0:7]))
model Net:
    l1 = Linear(96, 192)
    l2 = Linear(192, 7)
    fn forward(x) = x |> l1 |> relu |> l2
net = Net()
for epoch in range(3):
    for xb, yb in batches(x, labels, size=128, shuffle=true):
        loss = cross_entropy(net(xb), yb)
        minimize loss with adam(lr=0.01)
print(loss, accuracy(net(x), labels))
print(net.l2.b)
big[i, j] = sum x[i, k] * x[j, k]
print(sum(big), max(big))
"""


def test_thread_count_does_not_change_results():
    outs = []
    for n in ("1", "3", "8", "16"):
        out, err, code, c = run_native(PROGRAM, env={"ANVIL_THREADS": n})
        assert code == 0, err
        outs.append(out)
    assert all(o == outs[0] for o in outs), outs


def test_some_kernels_are_parallel():
    from anvil.backend.aarch64 import generate
    from util import compile_text
    asm, _ = generate(compile_text(PROGRAM).program)
    assert "bl _anvil_rt_parallel" in asm
