"""Source files and spans."""
from __future__ import annotations

import bisect
import os
from dataclasses import dataclass


class SourceFile:
    def __init__(self, path: str, text: str):
        self.path = path
        self.text = text
        self.line_starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.line_starts.append(i + 1)

    @classmethod
    def read(cls, path: str) -> "SourceFile":
        with open(path, encoding="utf-8") as f:
            return cls(path, f.read())

    @property
    def dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    def line_col(self, offset: int) -> tuple[int, int]:
        """1-based (line, column) of a character offset."""
        line = bisect.bisect_right(self.line_starts, offset) - 1
        return line + 1, offset - self.line_starts[line] + 1

    def line_text(self, line: int) -> str:
        start = self.line_starts[line - 1]
        end = self.line_starts[line] - 1 if line < len(self.line_starts) else len(self.text)
        return self.text[start:end]

    def display_path(self) -> str:
        try:
            rel = os.path.relpath(self.path)
            return rel if not rel.startswith("../..") else self.path
        except ValueError:
            return self.path


@dataclass(frozen=True)
class Span:
    file: SourceFile
    start: int
    end: int

    def to(self, other: "Span | None") -> "Span":
        if other is None:
            return self
        return Span(self.file, min(self.start, other.start), max(self.end, other.end))

    @property
    def line(self) -> int:
        return self.file.line_col(self.start)[0]

    @property
    def col(self) -> int:
        return self.file.line_col(self.start)[1]

    @property
    def text(self) -> str:
        return self.file.text[self.start:self.end]

    def location(self) -> str:
        line, col = self.file.line_col(self.start)
        return f"{self.file.display_path()}:{line}:{col}"

    def __repr__(self) -> str:  # keep dataclass reprs of AST nodes short
        return f"<{self.location()}>"
