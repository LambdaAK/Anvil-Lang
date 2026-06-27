"""`anvil repl`: try Anvil a line at a time.

Each input is added to the session's program, which is compiled again and run on the reference
interpreter; only the output that is new is shown. Programs are deterministic (random numbers
come from a fixed seed), so everything printed before comes out the same and is skipped. A bare
expression is printed. An input that does not compile is dropped, and its errors are shown."""
from __future__ import annotations

import io
import os
import sys

from . import ast as A
from .diagnostics import AnvilError, use_color
from .driver import compile_text
from .interp import AnvilRuntimeError, EndOfInput, Interpreter, ProgramExit
from .parser import parse
from .source import SourceFile

BANNER = """anvil repl — type Anvil; a bare expression is printed. A line ending in `:` starts a block,
which ends at an empty line. :program shows the session so far, :reset starts again, :quit (or
ctrl-D) leaves. Every input re-runs the whole session on the reference interpreter."""

QUIET_CALLS = ("print", "show", "sleep", "save", "seed")


class Session:
    def __init__(self, base_dir: str = "."):
        self.lines: list[str] = []
        self.shown = ""
        self.path = os.path.join(os.path.abspath(base_dir), "<repl>.anvil")

    def as_statement(self, text: str) -> str:
        """`x + 1` -> `print(x + 1)`; anything else stays as it is."""
        try:
            tree = parse(SourceFile("<input>", text + "\n"))
        except AnvilError:
            return text
        if len(tree.body) == 1 and isinstance(tree.body[0], A.ExprStmt):
            e = tree.body[0].expr
            if not (isinstance(e, A.Call) and isinstance(e.func, A.Name) and e.func.id in QUIET_CALLS):
                return f"print({text.strip()})"
        return text

    def run(self, text: str) -> tuple[str, str]:
        """Add text to the session; returns (new output, error message or "")."""
        candidate = self.lines + self.as_statement(text).splitlines()
        source = "\n".join(candidate) + "\n"
        try:
            compiled = compile_text(source, path=self.path)
        except AnvilError as e:
            return "", e.render(use_color(sys.stderr))
        out = io.StringIO()
        error = ""
        try:
            Interpreter(compiled.program, out=out, inp=io.StringIO("")).run()
        except (AnvilRuntimeError, ProgramExit, EndOfInput) as e:
            error = f"runtime error: {e}"
        full = out.getvalue()
        new = full[len(self.shown):] if full.startswith(self.shown) else full
        if error:
            return new, error                     # the input stays out of the session
        self.lines = candidate
        self.shown = full
        return new, ""


def main(base_dir: str = ".") -> int:
    try:
        import readline  # noqa: F401  (history and line editing)
    except ImportError:
        pass
    print(BANNER)
    s = Session(base_dir)
    while True:
        try:
            line = input("anvil> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if not line.strip():
            continue
        if line.strip() in (":quit", ":q", ":exit"):
            return 0
        if line.strip() == ":reset":
            s = Session(base_dir)
            continue
        if line.strip() == ":program":
            print("\n".join(s.lines) if s.lines else "(empty)")
            continue
        block = [line]
        if line.rstrip().endswith(":"):
            while True:
                try:
                    more = input("...  ")
                except EOFError:
                    break
                if not more.strip():
                    break
                block.append(more)
        out, err = s.run("\n".join(block))
        if out:
            sys.stdout.write(out)
        if err:
            print(err, file=sys.stderr)
