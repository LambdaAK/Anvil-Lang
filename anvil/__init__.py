"""Anvil: a small language for differentiable tensor programs that compiles to AArch64 assembly.

From Python, `anvil.function(source)` compiles an Anvil function and calls it on NumPy arrays."""
__version__ = "0.2.0"


def function(source: str, name: str | None = None, seed: int = 0):
    """An Anvil function callable on NumPy arrays (see anvil.pyapi)."""
    from .pyapi import function as make
    return make(source, name=name, seed=seed)


def load_ipython_extension(ip):
    """`%load_ext anvil` in Jupyter: `%%anvil` cells (see anvil.jupyter)."""
    from .jupyter import load_ipython_extension as load
    load(ip)
