"""Anvil in Jupyter notebooks: `%load_ext anvil`, then `%%anvil` cells.

A cell of declarations defines its functions in the notebook, as Python functions on NumPy
arrays (`anvil.function`):

    %%anvil
    fn attention(q: [t, d], k: [t, d], v: [t, d]) = softmax(q @ k.T / sqrt(d)) @ v

    attention(q, k, v)            # in a Python cell

Any other cell is a program: it is compiled to native code and run, and what it prints appears as
it runs (a training loop's progress, say). Options on the %%anvil line:

    --interp            run on the reference interpreter
    --set NAME=VALUE    give a `const` another value (repeatable)
    --check             stop at the first nan, and show the line that made it
    --profile           time every kernel
    --cost              show what the program costs instead of running it

The cell's file is the notebook's directory, so `idx("data/…")` and `use "model.anvil"` work as they
do next to a .anvil file. A function's cell runs once, when its function is first called; to call a
trained network, train in one cell, `save` it, and `load` it in the function's cell.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys

DECLARATIONS = ("FnDecl", "ModelDecl", "OptimizerDecl", "ConstDecl", "Use", "ParamDecl", "Assign", "AnnAssign")


def run_cell(line: str, cell: str, namespace: dict, out=None) -> None:
    """What `%%anvil <line>` does with a cell: defines its functions in namespace, or runs it."""
    from . import ast as A
    from .cli import parse_set
    from .diagnostics import AnvilError
    from .parser import parse
    from .source import SourceFile
    out = out or sys.stdout
    args = shlex.split(line)
    opts = {"interp": False, "check": None, "profile": False, "cost": False}
    consts = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--set" and i + 1 < len(args):
            k, v = parse_set(args[i + 1])
            consts[k] = v
            i += 2
            continue
        if a.startswith("--check"):
            opts["check"] = a.split("=", 1)[1] if "=" in a else "nan"
        elif a.lstrip("-") in opts:
            opts[a.lstrip("-")] = True
        else:
            print(f"%%anvil: unknown option {a}", file=out)
            return
        i += 1
    if not cell.endswith("\n"):
        cell += "\n"
    path = os.path.join(os.getcwd(), "<cell>.anvil")
    try:
        tree = parse(SourceFile(path, cell))
    except AnvilError as e:
        print(e.render(False), file=out)
        return
    fns = [s.name.id for s in tree.body if isinstance(s, A.FnDecl)]
    program = any(type(s).__name__ not in DECLARATIONS for s in tree.body) or not fns
    if not program:
        from .pyapi import Function
        for name in fns:
            namespace[name] = Function(cell, name=name, consts=consts)
        print(f"defined {', '.join(fns)}", file=out)
        return
    run_program(cell, path, opts, consts, out)


def run_program(cell: str, path: str, opts: dict, consts: dict, out) -> None:
    from .cli import build_native
    from .diagnostics import AnvilError
    from .driver import compile_text
    try:
        compiled = compile_text(cell, path=path, consts=consts, optimize=not opts["check"])
    except AnvilError as e:
        print(e.render(False), file=out)
        return
    for w in compiled.warnings:
        print(w.render(False), file=out)
    if opts["cost"]:
        from .cost import report
        print(report(compiled, "this cell"), file=out)
        return
    if opts["interp"]:
        from .interp import AnvilRuntimeError, Interpreter, ProgramExit
        try:
            Interpreter(compiled.program, out=out, check=opts["check"]).run()
        except (AnvilRuntimeError, ProgramExit) as e:
            print(f"runtime error: {e}", file=out)
        return
    exe, _, _, _ = build_native(compiled, 0, None, profile=opts["profile"], check=opts["check"])
    p = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for text in p.stdout:                      # as it runs
        out.write(text)
        if hasattr(out, "flush"):
            out.flush()
    p.wait()


def load_ipython_extension(ip):
    from IPython.core.magic import Magics, cell_magic, magics_class

    @magics_class
    class AnvilMagics(Magics):
        @cell_magic
        def anvil(self, line, cell):
            """%%anvil [--interp] [--set NAME=VALUE] [--check] [--profile] [--cost]: an Anvil cell."""
            run_cell(line, cell, self.shell.user_ns)

    ip.register_magics(AnvilMagics)
