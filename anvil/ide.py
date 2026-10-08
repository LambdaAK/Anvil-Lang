"""What an editor needs from the compiler: `anvil check --json file.anvil`.

    {"diagnostics": [{"range": [line, col, end_line, end_col], "severity": "error", "message": …,
                      "help": …, "notes": […], "related": [{"range": …, "message": …}]}],
     "hovers":      [{"range": […], "text": "f32[64, 128]", "definition": true, "inlay": "f32[64, 128]"}]}

Lines and columns count from 0, as editors do. The hovers come from the elaborator, which records
what every name in the file holds as the program is compiled. Shapes are static, so the editor can
show the shape of any tensor under the mouse, and after every definition (`h = relu(x @ W)` gets
`: f32[64, 128]` after `h`). A name inside a function that is called with different shapes shows
all of them. Hovers collected before a compile error are still reported.
"""
from __future__ import annotations

import os

from . import ast as A
from .dims import formula, has_names, show, show_shape
from .diagnostics import AnvilError, AnvilErrors, fmt_shape
from .source import SourceFile, Span

MAX_SHAPES = 4          # different values shown for one name (a function called with several shapes)


class CallSig:
    """How a function was called at a call site: its arguments' shapes and its result's."""
    kind = "call"

    def __init__(self, text: str):
        self.text = text


def short(val) -> str:
    """A value's type in a signature: f32[BATCH, T, D] (names only; the hover of each name has the numbers)."""
    from .values import CVal, TVal
    if isinstance(val, TVal):
        return f"{val.dtype}{show_shape(val.shape)}" if val.shape else f"{val.dtype}"
    if isinstance(val, CVal):
        return repr(val.value)
    return describe(val).split("\n")[0]


class Recorder:
    """Values seen at spans of one source file."""

    def __init__(self, path: str):
        self.path = path
        self.seen: dict[tuple, dict] = {}         # (start, end) -> {"vals": [...], "definition": bool, ...}

    def note(self, span: Span | None, val, definition: bool = False, annotated: bool = False):
        if span is None or span.file.path != self.path:
            return
        entry = self.seen.setdefault((span.start, span.end), {"span": span, "vals": [], "definition": False,
                                                              "annotated": False})
        entry["definition"] |= definition
        entry["annotated"] |= annotated
        if len(entry["vals"]) < 32:
            entry["vals"].append(val)


def describe(val, name: str = "") -> str:
    """How a value looks in a hover."""
    from .values import (AffVal, BatchesVal, BuiltinVal, CVal, DistVal, EVal, FnVal, FStrVal, IdxVal,
                         ModelDefVal, ModelInstVal, OptDefVal, OptSpecVal, PickVal, RangeVal, SVal, TextVal,
                         TupleVal, TVal)
    if isinstance(val, TVal):
        kind = "param " if val.buf.kind == "param" else ""
        if not val.shape:
            return f"{kind}{val.dtype} (scalar)"
        if has_names(val.shape):                      # f32[BATCH, T, D] = [16, 64, 64]
            return f"{kind}{val.dtype}{show_shape(val.shape)} = [{', '.join(str(int(d)) for d in val.shape)}]"
        return f"{kind}{val.dtype}{show_shape(val.shape)}"
    if isinstance(val, CVal):
        v = val.value
        if isinstance(v, bool):
            return f"{'true' if v else 'false'} (compile-time constant)"
        f = formula(v)                                # `16 = D/HEADS` for `const DH = D // HEADS`
        return f"{v!r}{f' = {f}' if f else ''} (compile-time constant, {'i32' if isinstance(v, int) else 'f32'})"
    if isinstance(val, SVal):
        return f'"{val.value}" (string constant)'
    if isinstance(val, IdxVal):
        vs = val.affine.vars()
        if len(vs) == 1 and val.affine.coef(vs[0]) == 1 and val.affine.const == 0:
            v = vs[0]
            if v.extent is None:
                return f"index {v.name}"
            return f"index {v.name} < {show(v.dim)} ({v.extent})" if v.dim is not None else f"index {v.name} < {v.extent}"
        return "index expression"
    if isinstance(val, AffVal):
        return "i32 (run-time integer: a loop counter, or computed from one)"
    if isinstance(val, EVal):
        return f"{val.dtype} element (index notation)"
    if isinstance(val, TupleVal):
        return "(" + ", ".join(describe(x) for x in val.items) + ")"
    if isinstance(val, FnVal):
        return declaration_line(val.decl)
    if isinstance(val, BuiltinVal):
        return f"builtin `{val.name}`"
    if isinstance(val, ModelDefVal):
        return declaration_line(val.decl)
    if isinstance(val, ModelInstVal):
        return model_summary(val)
    if isinstance(val, (OptDefVal,)):
        return declaration_line(val.decl)
    if isinstance(val, OptSpecVal):
        return f"optimizer {val.opt.decl.name.id}"
    if isinstance(val, DistVal):
        return f"{val.dist} distribution"
    if isinstance(val, BatchesVal):
        return f"batches of {val.size}"
    if isinstance(val, RangeVal):
        return "range"
    if isinstance(val, (PickVal, TextVal, FStrVal)):
        return "string"
    return val.kind


def declaration_line(decl) -> str:
    span = getattr(decl, "span", None)
    if span is None:
        return getattr(decl, "kind", "declaration")
    text = span.file.line_text(span.line).strip()
    where = os.path.basename(span.file.path)
    return f"{text}\n\n({where}:{span.line})" if where == "prelude.anvil" else text


def model_summary(inst) -> str:
    """The model's parameters with their shapes, and how many numbers they hold."""
    from .values import ModelInstVal, TVal
    rows, total = [], 0

    def walk(m, prefix):
        nonlocal total
        for name, b in m.scope.vars.items():
            if b.what == "param" and isinstance(b.val, TVal):
                rows.append(f"  {prefix}{name}: {describe(b.val).removeprefix('param ')}")
                total += int(b.val.numel)
            elif isinstance(b.val, ModelInstVal):
                walk(b.val, f"{prefix}{name}.")
    walk(inst, "")
    head = f"{inst.decl.name.id} (model, {total:,} parameters)"
    return "\n".join([head] + rows[:24] + (["  …"] if len(rows) > 24 else []))


def to_range(span: Span) -> list[int]:
    l1, c1 = span.file.line_col(span.start)
    l2, c2 = span.file.line_col(max(span.start, span.end))
    return [l1 - 1, c1 - 1, l2 - 1, c2 - 1]


def diagnostic(e: AnvilError, path: str, severity: str) -> dict:
    """An error at a span in this file. An error inside the standard library points at the
    first of its labels in this file, or at the top of the file."""
    span = e.span
    related = [lab for lab in e.labels if lab.span.file.path == path]
    message = e.message
    if span is None or span.file.path != path:
        if span is not None:
            message += f"  (at {span.location()})"
        span = related[0].span if related else None
    out = {"severity": severity, "message": message, "range": to_range(span) if span else [0, 0, 0, 0],
           "help": e.help, "notes": list(e.notes)}
    if e.label:
        out["label"] = e.label
    out["related"] = [{"range": to_range(lab.span), "message": lab.message} for lab in related if lab.span != span]
    return out


def analyze(path: str, text: str | None = None) -> dict:
    """Diagnostics and hovers for one file (text: its unsaved contents, if any)."""
    from .driver import load_prelude
    from .elaborate import Elaborator
    from .parser import parse
    path = os.path.abspath(path)
    if text is None:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    src = SourceFile(path, text)
    diagnostics = []
    rec = Recorder(path)
    try:
        tree = parse(src)
    except AnvilError as e:
        return {"diagnostics": [diagnostic(x, path, "error") for x in getattr(e, "errors", [e])], "hovers": []}
    elab = Elaborator(source=src)
    elab.recorder = rec
    try:
        elab.run(tree, load_prelude())
    except AnvilErrors as e:
        diagnostics += [diagnostic(x, path, "error") for x in e.errors]
    except AnvilError as e:
        diagnostics.append(diagnostic(e, path, "error"))
    except RecursionError:
        diagnostics.append({"severity": "error", "message": "program is nested too deeply", "range": [0, 0, 0, 0],
                            "help": None, "notes": [], "related": []})
    for w in getattr(elab, "warnings", []):
        diagnostics.append(diagnostic(w, path, "warning"))
    return {"diagnostics": diagnostics, "hovers": hovers(rec)}


def hovers(rec: Recorder) -> list[dict]:
    from .values import PoisonVal, TVal
    out = []
    for entry in rec.seen.values():
        sigs = list(dict.fromkeys(v.text for v in entry["vals"] if isinstance(v, CallSig)))
        vals = [v for v in entry["vals"] if not isinstance(v, (PoisonVal, CallSig))]
        if not vals and not sigs:
            continue
        texts = list(dict.fromkeys(describe(v) for v in vals))
        text = texts[0] if len(texts) == 1 else " | ".join(texts[:MAX_SHAPES]) + (" | …" if len(texts) > MAX_SHAPES else "")
        if sigs:                                       # at a call: the shapes in and out, with their names
            calls = "\n".join(sigs[:MAX_SHAPES]) + ("\n…" if len(sigs) > MAX_SHAPES else "")
            text = (text + "\n\n" if text else "") + "called here as\n" + calls
        h = {"range": to_range(entry["span"]), "text": text, "definition": entry["definition"]}
        # a shape after each definition of a tensor (not when the line already states it)
        if entry["definition"] and not entry["annotated"] and all(isinstance(v, TVal) and v.shape for v in vals):
            shapes = list(dict.fromkeys(f"{v.dtype}{show_shape(v.shape)}" for v in vals))
            h["inlay"] = " | ".join(shapes[:MAX_SHAPES])
        out.append(h)
    out.sort(key=lambda h: h["range"])
    return out


def shape_sheet(path: str) -> str:
    """`anvil shapes FILE`: the program's named dimensions, then every tensor it defines, line by line,
    with its shape in names and in numbers."""
    result = analyze(path)
    lines = open(path, encoding="utf-8").read().splitlines()
    out = []
    names, rows = [], []
    for h in result["hovers"]:
        if not h["definition"]:
            continue
        l0, c0, l1, c1 = h["range"]
        name = lines[l0][c0:c1] if l0 == l1 else lines[l0][c0:]
        text = h["text"].split("\n")[0]
        if text.endswith("(compile-time constant, i32)"):
            v = text.removesuffix(" (compile-time constant, i32)")
            names.append((name, v.split(" = ")[1] + " = " + v.split(" = ")[0] if " = " in v else v))
        elif text.startswith(("f32", "i32", "param ")):
            rows.append((l0 + 1, name, text))
    dims = [f"{n} = {v}" for n, v in names]
    if dims:
        out.append("dimensions: " + ", ".join(dims))
        out.append("")
    width = max((len(n) for _, n, _ in rows), default=4)
    seen = set()
    for line, name, text in rows:
        if (line, name) in seen:
            continue
        seen.add((line, name))
        out.append(f"{line:5d}  {name:<{width}}  {text}")
    for d in result["diagnostics"]:
        out.append(f"{d['severity']} at line {d['range'][0] + 1}: {d['message'].splitlines()[0]}")
    return "\n".join(out)
