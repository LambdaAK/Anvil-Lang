# Anvil

**A small language for differentiable tensor programs.**

*Anvil: Autodiff, Native, Vectorized, Index Language.*

You write a model the way it is written on paper (tensors, index notation, a loss to minimize)
and Anvil differentiates and runs it. The goal is a compiler that turns the whole training
program into native code for Apple Silicon.

Status: early. See [docs/PLAN.md](docs/PLAN.md) for the design.
