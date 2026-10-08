"""Compiler errors and their pretty rendering.

    error: shape mismatch in `@`
      ┌─ examples/mnist.anvil:14:12
      │
    14│     return h @ W1 + b2
      │            ━━━━━━ cannot multiply [64, 128] by [784, 128]
      │
      = note: inner dimensions must agree (128 ≠ 784)
"""
from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass, field

from .source import Span


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.enabled else s

    def red(self, s): return self._wrap("1;31", s)
    def yellow(self, s): return self._wrap("1;33", s)
    def blue(self, s): return self._wrap("34", s)
    def cyan(self, s): return self._wrap("36", s)
    def green(self, s): return self._wrap("32", s)
    def bold(self, s): return self._wrap("1", s)
    def dim(self, s): return self._wrap("2", s)


def use_color(stream=None) -> bool:
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


@dataclass
class Label:
    span: Span
    message: str = ""
    primary: bool = False


@dataclass
class AnvilError(Exception):
    message: str
    span: Span | None = None
    label: str = ""
    labels: list[Label] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    help: str | None = None
    kind: str = "error"

    def __post_init__(self):
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        loc = f"{self.span.location()}: " if self.span else ""
        return f"{loc}{self.message}"

    def note(self, text: str) -> "AnvilError":
        self.notes.append(text)
        return self

    def render(self, color: bool = False) -> str:
        st = Style(color)
        head = st.red(self.kind) if self.kind == "error" else st.yellow(self.kind)
        out = [f"{head}{st.bold(': ' + self.message)}"]
        labels = []
        if self.span is not None:
            labels.append(Label(self.span, self.label, primary=True))
        labels += self.labels
        if labels:
            primary = labels[0]
            gutter_w = max(len(str(l.span.line)) for l in labels)
            pad = " " * gutter_w
            out.append(f"{pad} {st.blue('┌─')} {primary.span.location()}")
            out.append(f"{pad} {st.blue('│')}")
            # group labels by line, in order of appearance
            by_line: dict[tuple, list[Label]] = {}
            for l in labels:
                by_line.setdefault((l.span.file.path, l.span.line), []).append(l)
            prev_line = None
            prev_path = primary.span.file.path
            for (path, line), ls in sorted(by_line.items(), key=lambda kv: (kv[0][0] != primary.span.file.path, kv[0][1])):
                if path != prev_path:                     # labels in another file (a call site)
                    out.append(f"{pad} {st.blue('│')}")
                    out.append(f"{pad} {st.blue('┌─')} {ls[0].span.location()}")
                    out.append(f"{pad} {st.blue('│')}")
                    prev_path, prev_line = path, None
                elif prev_line is not None and line > prev_line + 1:
                    out.append(f"{pad} {st.blue('·')}")
                prev_line = line
                src = ls[0].span.file.line_text(line).rstrip("\n")
                out.append(f"{st.blue(str(line).rjust(gutter_w))}{st.blue('│')} {src}")
                for l in sorted(ls, key=lambda l: l.span.start):
                    line_start = l.span.file.line_starts[line - 1]
                    col0 = l.span.start - line_start
                    end = min(l.span.end, line_start + len(src))
                    width = max(1, end - l.span.start)
                    # account for wide/tab characters crudely: keep 1 col per char
                    mark_ch = "━" if l.primary else "─"
                    marks = mark_ch * width
                    marks = st.red(marks) if l.primary else st.blue(marks)
                    msg = (" " + (st.red(l.message) if l.primary else st.blue(l.message))) if l.message else ""
                    out.append(f"{pad} {st.blue('│')} {' ' * col0}{marks}{msg}")
            out.append(f"{pad} {st.blue('│')}")
        else:
            pad = ""
        for n in self.notes:
            out.append(f"{pad} {st.blue('=')} {st.bold('note')}: {n}")
        if self.help:
            out.append(f"{pad} {st.blue('=')} {st.green('help')}: {self.help}")
        return "\n".join(out)


class AnvilErrors(AnvilError):
    """Several independent errors, reported together (the first one is the message)."""

    def __init__(self, errors: list):
        first = errors[0]
        super().__init__(first.message, first.span, first.label, list(first.labels), list(first.notes), first.help)
        self.errors = errors

    def render(self, color: bool = False) -> str:
        st = Style(color)
        parts = [e.render(color) for e in self.errors]
        more = "" if len(self.errors) < MAX_ERRORS else " (stopped after that many)"
        return "\n\n".join(parts) + f"\n\n{st.red('error')}{st.bold(f': {len(self.errors)} errors{more}')}"


class PoisonError(AnvilError):
    """A use of a name whose definition already failed: not reported again."""


MAX_ERRORS = 20


def suggest(name: str, candidates) -> str | None:
    """'did you mean' helper."""
    matches = difflib.get_close_matches(name, [c for c in candidates if not c.startswith("_")], n=1, cutoff=0.6)
    return matches[0] if matches else None


def fmt_shape(shape) -> str:
    """[BATCH=16, T=64, 4]: a named dimension with its size, others as numbers (see dims.py)."""
    from .dims import Dim, show
    return "[" + ", ".join(f"{show(d)}={int(d)}" if isinstance(d, Dim) else str(d) for d in shape) + "]"
