"""Tokenizer with Python-style significant indentation."""
from __future__ import annotations

from dataclasses import dataclass

from .diagnostics import AnvilError
from .source import SourceFile, Span

KEYWORDS = {
    "const", "param", "fn", "model", "optimizer", "return", "if", "elif", "else",
    "for", "in", "while", "break", "continue", "pass", "minimize", "maximize",
    "and", "or", "not", "true", "false", "assert",
}

# longest first
OPERATORS = [
    "...", "**=", "//=",
    "->", "|>", "**", "//", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "@=",
    "+", "-", "*", "/", "%", "@", "<", ">", "=", "(", ")", "[", "]", "{", "}",
    ",", ":", ".", "~", ";",
]

# Unicode sugar -> (kind, value)
ALIASES = {
    "Σ": ("NAME", "sum"), "∑": ("NAME", "sum"), "∏": ("NAME", "prod"),
    "∇": ("NAME", "grad"), "∞": ("NAME", "inf"),
    "√": ("OP", "√"), "→": ("OP", "->"), "≤": ("OP", "<="), "≥": ("OP", ">="),
    "≠": ("OP", "!="), "⋅": ("OP", "*"), "·": ("OP", "*"),
}

OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {")", "]", "}"}


@dataclass
class Token:
    kind: str     # NAME KEYWORD INT FLOAT STRING OP NEWLINE INDENT DEDENT EOF
    value: object
    span: Span

    def __repr__(self) -> str:
        return f"{self.kind}({self.value!r})"

    def is_op(self, *ops) -> bool:
        return self.kind == "OP" and self.value in ops

    def is_kw(self, *kws) -> bool:
        return self.kind == "KEYWORD" and self.value in kws

    def is_name(self, *names) -> bool:
        return self.kind == "NAME" and (not names or self.value in names)


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def _is_ident_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def tokenize(src: SourceFile, base: int = 0, text: str | None = None, interactive_indent: bool = True) -> list[Token]:
    """Tokenize `text` (default: whole file). `base` is the offset of text within the file
    (used for interpolated expressions inside strings)."""
    text = src.text if text is None else text
    toks: list[Token] = []
    n = len(text)
    i = 0
    indents = [0]
    depth: list[tuple[str, int]] = []   # open brackets
    at_line_start = True

    def span(a, b):
        return Span(src, base + a, base + b)

    def err(msg, a, b=None, **kw):
        raise AnvilError(msg, span(a, b if b is not None else a + 1), **kw)

    while i < n:
        ch = text[i]

        if at_line_start and not depth and interactive_indent:
            # measure indentation
            j = i
            col = 0
            while j < n and text[j] in " \t\f":
                if text[j] == "\t":
                    err("tabs are not allowed for indentation", j, help="indent with spaces (4 per level is conventional)")
                col += 1
                j += 1
            if j >= n:
                i = j
                break
            if text[j] == "\n" or text[j] == "#" or text[j] == "\r":
                # blank or comment-only line: skip entirely
                while j < n and text[j] != "\n":
                    j += 1
                i = j + 1
                continue
            at_line_start = False
            if col > indents[-1]:
                indents.append(col)
                toks.append(Token("INDENT", col, span(j, j)))
            elif col < indents[-1]:
                while col < indents[-1]:
                    indents.pop()
                    toks.append(Token("DEDENT", col, span(j, j)))
                if col != indents[-1]:
                    err("this line's indentation does not match any enclosing block", i, j,
                        help="make it line up with the start of an earlier line")
            i = j
            continue

        if ch == "\n":
            if not depth and interactive_indent:
                if toks and toks[-1].kind not in ("NEWLINE", "INDENT", "DEDENT"):
                    toks.append(Token("NEWLINE", None, span(i, i + 1)))
                at_line_start = True
            i += 1
            continue
        if ch in " \t\r\f":
            i += 1
            continue
        if ch == "\\" and i + 1 < n and text[i + 1] == "\n":
            i += 2
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue

        # numbers
        if ch.isdigit():
            j = i
            is_float = False
            while j < n and (text[j].isdigit() or text[j] == "_"):
                j += 1
            if j < n and text[j] == "." and j + 1 < n and text[j + 1].isdigit():
                is_float = True
                j += 1
                while j < n and (text[j].isdigit() or text[j] == "_"):
                    j += 1
            elif j < n and text[j] == "." and not (j + 1 < n and (text[j + 1] == "." or _is_ident_start(text[j + 1]))):
                # "1." is a float
                is_float = True
                j += 1
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    is_float = True
                    j = k
                    while j < n and text[j].isdigit():
                        j += 1
            raw = text[i:j].replace("_", "")
            if j < n and _is_ident_start(text[j]):
                err(f"invalid number literal `{text[i:j + 1]}`", i, j + 1)
            toks.append(Token("FLOAT" if is_float else "INT", float(raw) if is_float else int(raw), span(i, j)))
            i = j
            continue

        # strings (an `f` prefix is accepted out of Python habit: every string interpolates)
        if ch == "f" and i + 1 < n and text[i + 1] in "\"'":
            i += 1
            ch = text[i]
        if ch == '"' or ch == "'":
            q = ch
            j = i + 1
            while j < n and text[j] != q:
                if text[j] == "\\":
                    j += 1
                if j < n and text[j] == "\n":
                    err("unterminated string", i, j)
                j += 1
            if j >= n:
                err("unterminated string", i, j)
            toks.append(Token("STRING", text[i + 1:j], span(i, j + 1)))
            i = j + 1
            continue

        # unicode aliases
        if ch in ALIASES:
            kind, val = ALIASES[ch]
            toks.append(Token(kind, val, span(i, i + 1)))
            i += 1
            continue

        # identifiers / keywords
        if _is_ident_start(ch):
            j = i + 1
            while j < n and _is_ident_char(text[j]) and text[j] not in ALIASES:
                j += 1
            word = text[i:j]
            toks.append(Token("KEYWORD" if word in KEYWORDS else "NAME", word, span(i, j)))
            i = j
            continue

        # operators
        for op in OPERATORS:
            if text.startswith(op, i):
                if op in OPEN:
                    depth.append((op, i))
                elif op in CLOSE:
                    if not depth:
                        err(f"unmatched `{op}`", i)
                    o, oi = depth.pop()
                    if OPEN[o] != op:
                        raise AnvilError(f"`{op}` does not match the `{o}` opened here", span(i, i + 1),
                                       label="closing bracket", labels=[])
                toks.append(Token("OP", op, span(i, i + len(op))))
                i += len(op)
                break
        else:
            err(f"unexpected character `{ch}`", i)

    if depth:
        o, oi = depth[-1]
        err(f"`{o}` was never closed", oi)
    end = len(text)
    if toks and toks[-1].kind not in ("NEWLINE", "INDENT", "DEDENT") and interactive_indent:
        toks.append(Token("NEWLINE", None, span(end, end)))
    while len(indents) > 1:
        indents.pop()
        toks.append(Token("DEDENT", 0, span(end, end)))
    toks.append(Token("EOF", None, span(end, end)))
    return toks
