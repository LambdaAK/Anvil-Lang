"""Compilation pipeline: source -> AST -> kernel IR -> (optimized) IR."""
from __future__ import annotations

import os

from . import ir
from .elaborate import Elaborator
from .parser import parse
from .source import SourceFile

PRELUDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelude.anvil")
_prelude_cache = {}


def load_prelude():
    mtime = os.path.getmtime(PRELUDE_PATH)
    hit = _prelude_cache.get("p")
    if hit is None or hit[0] != mtime:
        src = SourceFile.read(PRELUDE_PATH)
        hit = (mtime, parse(src))
        _prelude_cache["p"] = hit
    return hit[1]


class Compiled:
    def __init__(self, program: ir.Program, elab: Elaborator, source: SourceFile):
        self.program = program
        self.elab = elab
        self.source = source
        self.warnings = elab.warnings


def compile_source(src: SourceFile, optimize: bool = True, seed: int = 0, consts: dict | None = None) -> Compiled:
    """consts: values for the program's `const`s, overriding their definitions (`--set`)."""
    tree = parse(src)
    elab = Elaborator(source=src, seed=seed, consts=consts)
    program = elab.run(tree, load_prelude())
    if optimize:
        from .optimize import optimize_program
        optimize_program(program)
    return Compiled(program, elab, src)


def compile_file(path: str, **kw) -> Compiled:
    return compile_source(SourceFile.read(path), **kw)


def compile_text(text: str, path: str = "<input>.anvil", **kw) -> Compiled:
    return compile_source(SourceFile(path, text), **kw)
