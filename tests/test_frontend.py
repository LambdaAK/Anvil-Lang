"""Lexer and parser."""
import pytest

from anvil import ast as A
from anvil.diagnostics import AnvilError
from anvil.lexer import tokenize
from anvil.parser import parse_text
from anvil.source import SourceFile


def kinds(text):
    return [(t.kind, t.value) for t in tokenize(SourceFile("t.anvil", text))]


def test_indentation_tokens():
    toks = kinds("if x:\n    y = 1\nz = 2\n")
    assert ("INDENT", 4) in toks and ("DEDENT", 0) in toks


def test_unicode_aliases():
    toks = kinds("y = Σ x[i] → √z ≤ 1\n")
    assert ("NAME", "sum") in toks and ("OP", "->") in toks and ("OP", "√") in toks and ("OP", "<=") in toks


def test_numbers():
    toks = kinds("a = 1_000 + 1e-3 + 2.5\n")
    assert ("INT", 1000) in toks and ("FLOAT", 0.001) in toks and ("FLOAT", 2.5) in toks


def test_fstring_prefix_accepted():
    prog = parse_text('print(f"loss {x:.3f}")\n')
    call = prog.body[0].expr
    assert isinstance(call.args[0], A.Str) and call.args[0].parts[1].spec == ".3f"


def test_reduction_precedence():
    # sum binds a product, not the following `+`: (Σ a*b) + c
    e = parse_text("y = sum a[i] * b[i] + c\n").body[0].value
    assert isinstance(e, A.Binary) and e.op == "+" and isinstance(e.left, A.Reduce)


def test_spaced_reduction_is_prefix():
    e = parse_text("v = mean (x[j] - m) ** 2\n").body[0].value
    assert isinstance(e, A.Reduce)          # mean of the square, not square of the mean
    e = parse_text("v = mean(x) ** 2\n").body[0].value
    assert isinstance(e, A.Binary) and e.op == "**"


def test_pipe_desugars_to_calls():
    e = parse_text("y = x |> f |> g(2)\n").body[0].value
    assert isinstance(e, A.Call) and e.func.id == "g" and isinstance(e.args[0], A.Call)


def test_index_assignment_with_where():
    s = parse_text("p[i, j] = max x[2*i + u, 2*j + v] where u < 2, v < 2\n").body[0]
    assert isinstance(s, A.IndexAssign) and [w[0].id for w in s.where] == ["u", "v"]


def test_declarations():
    prog = parse_text("""
param W: [784, 128] ~ normal(0, 0.1)
fn f(x: [n, 784]) -> [n, 10] = x @ W
model M(a, b):
    param p: [a]
    fn forward(x) = x + p
optimizer o(lr = 0.1):
    state m
    step(w, g):
        w -= lr * g
minimize loss over M with o(lr=0.5)
""")
    kinds_ = [type(s).__name__ for s in prog.body]
    assert kinds_ == ["ParamDecl", "FnDecl", "ModelDecl", "OptimizerDecl", "Minimize"]


@pytest.mark.parametrize("src,msg", [
    ("for x in range(3)\n    print(x)\n", "expected `:`"),
    ("y = (1 + 2\n", "never closed"),
    ("minimize loss\n", "with <optimizer>"),
    ("x = 1 <= 2 <= 3\n", "chained comparisons"),
    ("x[1] ~ normal(0, 1)\n", "sampling needs a type"),
    ("print(\"a {x\")\n", "unclosed `{`"),
])
def test_parse_errors(src, msg):
    with pytest.raises(AnvilError) as e:
        parse_text(src)
    assert msg in e.value.message


def test_item_assignment_forms():
    assert isinstance(parse_text("x[ptr + 1] = v\n").body[0], A.SetItem)
    assert isinstance(parse_text("x[2:5] += 1\n").body[0], A.SetItem)
    assert isinstance(parse_text("y[i, j] = 0\n").body[0], A.IndexAssign)   # decided by the elaborator


def test_string_escapes():
    s = parse_text('print("\\e[1m \\x41 \\u00e9")\n').body[0].expr.args[0]
    assert s.plain == "\x1b[1m A é"
