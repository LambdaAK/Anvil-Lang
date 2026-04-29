# Anvil

**A small language for differentiable tensor programs.**

*Anvil: Autodiff, Native, Vectorized, Index Language.*

You write a model the way it is written on paper (tensors, index notation, a loss to minimize)
and Anvil differentiates and runs it.

```python
param w: [3]
x: [100, 3] ~ normal(0, 1)
y[i] = sum x[i, j] * [1.0, -2.0, 0.5][j]
for step in range(200):
    loss = mean((x @ w - y) ** 2)
    minimize loss with sgd(lr=0.1)
print(w)
```

## Quick start

```bash
bin/anvil run examples/hello.anvil
bin/anvil run --interp examples/linear_regression.anvil
```

Status: the parser, shape checker, automatic differentiation and a NumPy reference interpreter
work. A native ARM64 backend is next ([docs/PLAN.md](docs/PLAN.md)).
