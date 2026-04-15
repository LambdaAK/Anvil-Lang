"""Recursive-descent parser: tokens -> AST."""
from __future__ import annotations

from . import ast as A
from .diagnostics import AnvilError
from .lexer import Token, tokenize
from .source import SourceFile, Span

REDUCTIONS = {"sum", "mean", "max", "min", "prod", "argmax", "argmin"}
AUG_OPS = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "//=": "//", "**=": "**", "@=": "@"}
COMPARE = {"==", "!=", "<", "<=", ">", ">="}


def _describe(tok: Token) -> str:
    if tok.kind == "EOF":
        return "end of file"
    if tok.kind == "NEWLINE":
        return "end of line"
    if tok.kind == "INDENT":
        return "an indented block"
    if tok.kind == "DEDENT":
        return "end of block"
    if tok.kind == "STRING":
        return "a string"
    if tok.kind in ("INT", "FLOAT"):
        return f"number `{tok.value}`"
    return f"`{tok.value}`"


class Parser:
    def __init__(self, src: SourceFile, tokens: list[Token]):
        self.src = src
        self.toks = tokens
        self.i = 0

    # ------------------------------------------------------------------ helpers
    @property
    def tok(self) -> Token:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Token:
        t = self.toks[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def error(self, msg: str, tok: Token | None = None, **kw) -> AnvilError:
        tok = tok or self.tok
        return AnvilError(msg, tok.span, **kw)

    def expect_op(self, op: str, context: str = "") -> Token:
        if not self.tok.is_op(op):
            ctx = f" {context}" if context else ""
            raise self.error(f"expected `{op}`{ctx}, found {_describe(self.tok)}")
        return self.next()

    def expect_kw(self, kw: str) -> Token:
        if not self.tok.is_kw(kw):
            raise self.error(f"expected `{kw}`, found {_describe(self.tok)}")
        return self.next()

    def expect_name(self, what: str = "a name") -> A.Name:
        t = self.tok
        if t.kind != "NAME":
            if t.kind == "KEYWORD":
                raise self.error(f"expected {what}, found keyword `{t.value}`",
                                 help=f"`{t.value}` is reserved; pick another name")
            raise self.error(f"expected {what}, found {_describe(t)}")
        self.next()
        return A.Name(t.span, t.value)

    def expect_newline(self):
        if self.tok.kind == "NEWLINE":
            self.next()
        elif self.tok.kind in ("EOF", "DEDENT"):
            pass
        elif self.tok.is_op(";"):
            self.next()
        else:
            raise self.error(f"expected end of line, found {_describe(self.tok)}")

    def span_from(self, start: Span) -> Span:
        prev = self.toks[self.i - 1] if self.i > 0 else self.tok
        return start.to(prev.span)

    # ------------------------------------------------------------------ program / blocks
    def parse_program(self) -> A.Program:
        start = self.tok.span
        body = []
        while self.tok.kind != "EOF":
            if self.tok.kind == "NEWLINE":
                self.next()
                continue
            if self.tok.kind == "INDENT":
                raise self.error("unexpected indentation")
            body.append(self.parse_stmt())
        return A.Program(start, body)

    def parse_block(self, context: str) -> list[A.Stmt]:
        self.expect_op(":", f"to start the body of {context}")
        if self.tok.kind == "NEWLINE":
            self.next()
            if self.tok.kind != "INDENT":
                raise self.error(f"expected an indented block for {context}")
            self.next()
            stmts = []
            while self.tok.kind not in ("DEDENT", "EOF"):
                if self.tok.kind == "NEWLINE":
                    self.next()
                    continue
                stmts.append(self.parse_stmt())
            if self.tok.kind == "DEDENT":
                self.next()
            return stmts
        # single-line block
        stmt = self.parse_simple_stmt()
        stmts = [stmt]
        while self.tok.is_op(";"):
            self.next()
            if self.tok.kind == "NEWLINE":
                break
            stmts.append(self.parse_simple_stmt())
        self.expect_newline()
        return stmts

    # ------------------------------------------------------------------ statements
    def parse_stmt(self) -> A.Stmt:
        t = self.tok
        if t.kind == "KEYWORD":
            kw = t.value
            if kw == "const":
                return self.parse_const()
            if kw == "param":
                return self.parse_param()
            if kw == "fn":
                return self.parse_fn()
            if kw == "model":
                return self.parse_model()
            if kw == "optimizer":
                return self.parse_optimizer()
            if kw == "if":
                return self.parse_if()
            if kw == "for":
                return self.parse_for()
            if kw == "while":
                return self.parse_while()
            if kw in ("elif", "else"):
                raise self.error(f"`{kw}` without a matching `if`")
        if t.kind == "NAME" and t.value == "use" and self.peek().kind == "STRING":
            self.next()                              # `use "file.anvil"`: a soft keyword
            path = self.parse_expr()
            if not (isinstance(path, A.Str) and path.is_plain):
                raise AnvilError("`use` takes a file name in a plain string", path.span)
            self.expect_newline()
            return A.Use(self.span_from(t.span), path.plain)
        if t.kind == "NAME" and t.value == "static" and self.peek().is_kw("for"):
            self.next()                              # `static for`: a soft keyword
            loop = self.parse_for()
            loop.static = True
            loop.span = t.span.to(loop.span)
            return loop
        s = self.parse_simple_stmt()
        self.expect_newline()
        return s

    def parse_simple_stmt(self) -> A.Stmt:
        t = self.tok
        if t.is_kw("return"):
            self.next()
            if self.tok.kind in ("NEWLINE", "EOF", "DEDENT") or self.tok.is_op(";"):
                return A.Return(t.span, None)
            val = self.parse_exprlist()
            return A.Return(self.span_from(t.span), val)
        if t.is_kw("break"):
            self.next()
            return A.Break(t.span)
        if t.is_kw("continue"):
            self.next()
            return A.Continue(t.span)
        if t.is_kw("pass"):
            self.next()
            return A.Pass(t.span)
        if t.is_kw("assert"):
            self.next()
            cond = self.parse_expr()
            message = None
            if self.tok.is_op(","):
                self.next()
                message = self.parse_expr()
            return A.Assert(self.span_from(t.span), cond, message)
        if t.is_kw("minimize", "maximize"):
            return self.parse_minimize()
        if t.kind == "KEYWORD" and t.value in ("const", "param", "fn", "model", "optimizer", "if", "for", "while"):
            raise self.error(f"`{t.value}` must start its own line")

        first = self.parse_expr()
        targets = [first]
        while self.tok.is_op(","):
            self.next()
            targets.append(self.parse_expr())

        tok = self.tok
        if tok.is_op("="):
            self.next()
            if len(targets) == 1 and isinstance(first, A.Subscript) and isinstance(first.value, A.Name):
                return self.finish_index_assign(first, accumulate=False)
            for tg in targets:
                if not isinstance(tg, A.Name):
                    raise AnvilError("cannot assign to this expression", tg.span,
                                   help="assign to a name (`x = ...`) or define a tensor by its indices (`y[i] = ...`)")
            value = self.parse_exprlist()
            return A.Assign(self.span_from(first.span), targets, value)
        if tok.is_op(":") and len(targets) == 1 and isinstance(first, A.Name):
            self.next()
            ty = self.parse_type()
            value = sample = None
            if self.tok.is_op("="):
                self.next()
                value = self.parse_expr()
            elif self.tok.is_op("~"):
                self.next()
                sample = self.parse_expr()
            else:
                raise self.error(f"expected `=` or `~` after the type of `{first.id}`")
            return A.AnnAssign(self.span_from(first.span), first, ty, value, sample)
        if tok.kind == "OP" and tok.value in AUG_OPS:
            self.next()
            if len(targets) != 1:
                raise AnvilError("augmented assignment needs a single target", tok.span)
            if isinstance(first, A.Subscript) and isinstance(first.value, A.Name):
                value = self.parse_expr()
                where = self.parse_where()
                return A.SetItem(self.span_from(first.span), first.value, first.items, value, AUG_OPS[tok.value],
                                 where)
            if not isinstance(first, A.Name):
                raise AnvilError("cannot assign to this expression", first.span)
            value = self.parse_expr()
            return A.AugAssign(self.span_from(first.span), first, AUG_OPS[tok.value], value)
        if tok.is_op("~"):
            raise self.error("sampling needs a type: write `x: [shape] ~ distribution`")
        if len(targets) > 1:
            raise self.error(f"expected `=`, found {_describe(tok)}")
        return A.ExprStmt(first.span, first)

    def finish_index_assign(self, target: A.Subscript, accumulate: bool):
        if not all(isinstance(it, (A.Name, A.EllipsisIdx)) for it in target.items):
            # `x[ptr + 1] = v`, `x[2:5] = 0`: assignment into part of an existing tensor
            value = self.parse_expr()
            return A.SetItem(self.span_from(target.span), target.value, target.items, value, None)
        indices = []
        seen = set()
        for it in target.items:
            if isinstance(it, A.EllipsisIdx):
                indices.append(it)
            elif isinstance(it, A.Name):
                if it.id in seen:
                    raise AnvilError(f"index `{it.id}` appears twice on the left-hand side", it.span)
                seen.add(it.id)
                indices.append(it)
        value = self.parse_expr()
        return A.IndexAssign(self.span_from(target.span), target.value, indices, value, self.parse_where(),
                             accumulate)

    def parse_where(self) -> list:
        where = []
        if self.tok.is_name("where"):
            self.next()
            while True:
                nm = self.expect_name("an index name")
                self.expect_op("<", "in a `where` clause (write `u < 3`)")
                bound = self.parse_arith()
                where.append((nm, bound))
                if not self.tok.is_op(","):
                    break
                self.next()
        return where

    def parse_minimize(self) -> A.Minimize:
        t = self.next()
        objective = self.parse_expr()
        over = None
        if self.tok.is_name("over"):
            self.next()
            over = [self.parse_expr()]
            while self.tok.is_op(","):
                self.next()
                over.append(self.parse_expr())
        if not self.tok.is_name("with"):
            raise self.error(f"expected `with <optimizer>` after `{t.value} ...`",
                             help=f"e.g. `{t.value} loss with sgd(lr=0.1)`")
        self.next()
        opt = self.parse_expr()
        return A.Minimize(self.span_from(t.span), objective, over, opt, maximize=(t.value == "maximize"))

    def parse_const(self) -> A.ConstDecl:
        t = self.next()
        name = self.expect_name()
        self.expect_op("=", "in a const declaration")
        value = self.parse_expr()
        self.expect_newline()
        return A.ConstDecl(self.span_from(t.span), name, value)

    def parse_param(self) -> A.ParamDecl:
        t = self.next()
        name = self.expect_name("a parameter name")
        self.expect_op(":", f"(parameters need a type, e.g. `param {name.id}: [784, 128]`)")
        ty = self.parse_type()
        init = sample = None
        if self.tok.is_op("="):
            self.next()
            init = self.parse_expr()
        elif self.tok.is_op("~"):
            self.next()
            sample = self.parse_expr()
        self.expect_newline()
        return A.ParamDecl(self.span_from(t.span), name, ty, init, sample)

    def parse_params(self) -> list[A.Param]:
        self.expect_op("(")
        params = []
        while not self.tok.is_op(")"):
            nm = self.expect_name("a parameter name")
            ty = default = None
            if self.tok.is_op(":"):
                self.next()
                ty = self.parse_type()
            if self.tok.is_op("="):
                self.next()
                default = self.parse_expr()
            params.append(A.Param(self.span_from(nm.span), nm.id, ty, default))
            if not self.tok.is_op(","):
                break
            self.next()
        self.expect_op(")", "to close the parameter list")
        return params

    def parse_fn(self) -> A.FnDecl:
        t = self.next()
        name = self.expect_name("a function name")
        params = self.parse_params()
        ret = None
        if self.tok.is_op("->"):
            self.next()
            ret = self.parse_type()
        if self.tok.is_op("="):
            self.next()
            body = self.parse_expr()
            self.expect_newline()
            return A.FnDecl(self.span_from(t.span), name, params, ret, body)
        if not self.tok.is_op(":"):
            raise self.error(f"expected `:` or `=` after the signature of `{name.id}`",
                             help=f"`fn {name.id}(x) = expr` or a block after `:`")
        body = self.parse_block(f"function `{name.id}`")
        return A.FnDecl(t.span.to(name.span), name, params, ret, body)

    def parse_model(self) -> A.ModelDecl:
        t = self.next()
        name = self.expect_name("a model name")
        params = self.parse_params() if self.tok.is_op("(") else []
        body = self.parse_block(f"model `{name.id}`")
        return A.ModelDecl(t.span.to(name.span), name, params, body)

    def parse_optimizer(self) -> A.OptimizerDecl:
        t = self.next()
        name = self.expect_name("an optimizer name")
        params = self.parse_params() if self.tok.is_op("(") else []
        self.expect_op(":", f"to start the body of optimizer `{name.id}`")
        if self.tok.kind != "NEWLINE":
            raise self.error("an optimizer body must be an indented block")
        self.next()
        if self.tok.kind != "INDENT":
            raise self.error("expected an indented block")
        self.next()
        state: list[A.Name] = []
        step_params = None
        step_body = None
        while self.tok.kind not in ("DEDENT", "EOF"):
            if self.tok.kind == "NEWLINE":
                self.next()
                continue
            if self.tok.is_name("state"):
                self.next()
                state.append(self.expect_name("a state name"))
                while self.tok.is_op(","):
                    self.next()
                    state.append(self.expect_name("a state name"))
                self.expect_newline()
            elif self.tok.is_name("step"):
                st = self.next()
                self.expect_op("(")
                step_params = []
                while not self.tok.is_op(")"):
                    step_params.append(self.expect_name())
                    if not self.tok.is_op(","):
                        break
                    self.next()
                self.expect_op(")")
                if not 2 <= len(step_params) <= 3:
                    raise AnvilError("`step` takes (w, g) or (w, g, t)", st.span)
                step_body = self.parse_block("the optimizer step")
            else:
                raise self.error("an optimizer body contains `state ...` and `step(w, g, t): ...`")
        if self.tok.kind == "DEDENT":
            self.next()
        if step_body is None:
            raise AnvilError(f"optimizer `{name.id}` has no `step(w, g): ...` rule", name.span)
        return A.OptimizerDecl(t.span.to(name.span), name, params, state, step_params, step_body)

    def parse_if(self) -> A.If:
        t = self.next()
        cond = self.parse_expr()
        body = self.parse_block("`if`")
        orelse: list[A.Stmt] = []
        if self.tok.is_kw("elif"):
            orelse = [self.parse_if()]
        elif self.tok.is_kw("else"):
            self.next()
            orelse = self.parse_block("`else`")
        return A.If(t.span.to(cond.span), cond, body, orelse)

    def parse_for(self) -> A.For:
        t = self.next()
        targets = [self.expect_name("a loop variable")]
        while self.tok.is_op(","):
            self.next()
            targets.append(self.expect_name("a loop variable"))
        if not self.tok.is_kw("in"):
            raise self.error(f"expected `in` after the loop variable{'s' if len(targets) > 1 else ''}")
        self.next()
        it = self.parse_expr()
        body = self.parse_block("the `for` loop")
        return A.For(t.span.to(it.span), targets, it, body)

    def parse_while(self) -> A.While:
        t = self.next()
        cond = self.parse_expr()
        body = self.parse_block("the `while` loop")
        return A.While(t.span.to(cond.span), cond, body)

    # ------------------------------------------------------------------ types
    def parse_type(self) -> A.TypeExpr:
        t = self.tok
        dtype = None
        if t.is_name("f32", "i32"):
            dtype = t.value
            self.next()
            if not self.tok.is_op("["):
                return A.TypeExpr(t.span, dtype, None)
        elif t.is_name("f64", "f16", "bf16", "i64", "u8", "bool"):
            raise self.error(f"element type `{t.value}` is not supported yet", help="use f32 or i32")
        if not self.tok.is_op("["):
            raise self.error(f"expected a type like `[64, 784]` or `i32[n]`, found {_describe(self.tok)}")
        self.next()
        dims = []
        while not self.tok.is_op("]"):
            dims.append(self.parse_arith())
            if not self.tok.is_op(","):
                break
            self.next()
        self.expect_op("]", "to close the shape")
        return A.TypeExpr(self.span_from(t.span), dtype, dims)

    # ------------------------------------------------------------------ expressions
    def parse_exprlist(self) -> A.Expr:
        first = self.parse_expr()
        if not self.tok.is_op(","):
            return first
        items = [first]
        while self.tok.is_op(","):
            self.next()
            if self.tok.kind in ("NEWLINE", "EOF"):
                break
            items.append(self.parse_expr())
        return A.TupleLit(self.span_from(first.span), items)

    def parse_expr(self) -> A.Expr:
        e = self.parse_pipe()
        if self.tok.is_kw("if"):
            self.next()
            cond = self.parse_pipe()
            if not self.tok.is_kw("else"):
                raise self.error("expected `else` in conditional expression", help="`a if cond else b`")
            self.next()
            other = self.parse_expr()
            return A.Cond(self.span_from(e.span), cond, e, other)
        return e

    def parse_pipe(self) -> A.Expr:
        e = self.parse_or()
        while self.tok.is_op("|>"):
            self.next()
            f = self.parse_or()
            if isinstance(f, A.Call):
                e = A.Call(e.span.to(f.span), f.func, [e] + f.args, f.kwargs)
            else:
                e = A.Call(e.span.to(f.span), f, [e], [])
        return e

    def parse_or(self) -> A.Expr:
        e = self.parse_and()
        while self.tok.is_kw("or"):
            self.next()
            r = self.parse_and()
            e = A.Binary(e.span.to(r.span), "or", e, r)
        return e

    def parse_and(self) -> A.Expr:
        e = self.parse_not()
        while self.tok.is_kw("and"):
            self.next()
            r = self.parse_not()
            e = A.Binary(e.span.to(r.span), "and", e, r)
        return e

    def parse_not(self) -> A.Expr:
        if self.tok.is_kw("not"):
            t = self.next()
            e = self.parse_not()
            return A.Unary(t.span.to(e.span), "not", e)
        return self.parse_comparison()

    def parse_comparison(self) -> A.Expr:
        e = self.parse_arith()
        if self.tok.kind == "OP" and self.tok.value in COMPARE:
            op = self.next().value
            r = self.parse_arith()
            e = A.Binary(e.span.to(r.span), op, e, r)
            if self.tok.kind == "OP" and self.tok.value in COMPARE:
                raise self.error("chained comparisons are not supported", help="combine them with `and`")
        return e

    def parse_arith(self) -> A.Expr:
        e = self.parse_term()
        while self.tok.is_op("+", "-"):
            op = self.next().value
            r = self.parse_term()
            e = A.Binary(e.span.to(r.span), op, e, r)
        return e

    def parse_term(self) -> A.Expr:
        e = self.parse_factor()
        while self.tok.is_op("*", "/", "//", "%", "@"):
            op = self.next().value
            r = self.parse_factor()
            e = A.Binary(e.span.to(r.span), op, e, r)
        return e

    def _starts_operand(self, tok: Token) -> bool:
        return (tok.kind in ("NAME", "INT", "FLOAT")
                or tok.is_op("-", "√")
                or tok.is_kw("true", "false"))

    def parse_factor(self) -> A.Expr:
        t = self.tok
        if t.is_op("-", "+", "√"):
            self.next()
            e = self.parse_factor()
            if t.value == "+":
                return e
            return A.Unary(t.span.to(e.span), t.value, e)
        nxt = self.peek()
        spaced_paren = nxt.is_op("(") and nxt.span.start > t.span.end   # `sum (a - b) ** 2` reads as math
        if t.kind == "NAME" and t.value in REDUCTIONS and (self._starts_operand(nxt) or spaced_paren):
            self.next()
            operand = self.parse_term()
            return A.Reduce(t.span.to(operand.span), t.value, operand)
        return self.parse_power()

    def parse_power(self) -> A.Expr:
        e = self.parse_postfix()
        if self.tok.is_op("**"):
            self.next()
            r = self.parse_factor()
            return A.Binary(e.span.to(r.span), "**", e, r)
        return e

    def parse_postfix(self) -> A.Expr:
        e = self.parse_atom()
        while True:
            t = self.tok
            if t.is_op("("):
                e = self.finish_call(e)
            elif t.is_op("["):
                e = self.finish_subscript(e)
            elif t.is_op("."):
                self.next()
                nm = self.expect_name("an attribute name")
                e = A.Attr(e.span.to(nm.span), e, nm.id)
            else:
                return e

    def finish_call(self, func: A.Expr) -> A.Call:
        self.expect_op("(")
        args: list[A.Expr] = []
        kwargs: list[A.Keyword] = []
        while not self.tok.is_op(")"):
            if self.tok.kind == "NAME" and self.peek().is_op("="):
                nm = self.next()
                self.next()
                val = self.parse_expr()
                if any(k.name == nm.value for k in kwargs):
                    raise AnvilError(f"keyword argument `{nm.value}` given twice", nm.span)
                kwargs.append(A.Keyword(nm.span.to(val.span), nm.value, val))
            else:
                if kwargs:
                    raise self.error("positional argument after keyword argument")
                args.append(self.parse_expr())
            if not self.tok.is_op(","):
                break
            self.next()
        end = self.expect_op(")", "to close the argument list")
        return A.Call(func.span.to(end.span), func, args, kwargs)

    def finish_subscript(self, value: A.Expr) -> A.Subscript:
        self.expect_op("[")
        items = []
        while not self.tok.is_op("]"):
            items.append(self.parse_subscript_item())
            if not self.tok.is_op(","):
                break
            self.next()
        end = self.expect_op("]", "to close the subscript")
        if not items:
            raise AnvilError("empty subscript", value.span.to(end.span))
        return A.Subscript(value.span.to(end.span), value, items)

    def parse_subscript_item(self):
        t = self.tok
        if t.is_op("..."):
            self.next()
            return A.EllipsisIdx(t.span)
        start = stop = step = None
        if not t.is_op(":"):
            start = self.parse_expr()
            if not self.tok.is_op(":"):
                return start
        # slice
        self.expect_op(":")
        if not (self.tok.is_op(",", "]", ":")):
            stop = self.parse_expr()
        if self.tok.is_op(":"):
            self.next()
            if not self.tok.is_op(",", "]"):
                step = self.parse_expr()
        return A.Slice(self.span_from(t.span), start, stop, step)

    def parse_atom(self) -> A.Expr:
        t = self.tok
        if t.kind == "NAME":
            self.next()
            return A.Name(t.span, t.value)
        if t.kind in ("INT", "FLOAT"):
            self.next()
            return A.Num(t.span, t.value)
        if t.kind == "STRING":
            self.next()
            return self.parse_string(t)
        if t.is_kw("true", "false"):
            self.next()
            return A.Bool(t.span, t.value == "true")
        if t.is_op("("):
            self.next()
            if self.tok.is_op(")"):
                end = self.next()
                return A.TupleLit(t.span.to(end.span), [])
            e = self.parse_expr()
            if self.tok.is_op(","):
                items = [e]
                while self.tok.is_op(","):
                    self.next()
                    if self.tok.is_op(")"):
                        break
                    items.append(self.parse_expr())
                end = self.expect_op(")")
                return A.TupleLit(t.span.to(end.span), items)
            self.expect_op(")", "to close the parenthesis")
            return e
        if t.is_op("["):
            self.next()
            items = []
            while not self.tok.is_op("]"):
                items.append(self.parse_expr())
                if not self.tok.is_op(","):
                    break
                self.next()
            end = self.expect_op("]", "to close the list")
            return A.ListLit(t.span.to(end.span), items)
        if t.is_op("..."):
            raise self.error("`...` can only be used inside an index list, e.g. `x[..., j]`")
        if t.kind == "KEYWORD":
            raise self.error(f"expected an expression, found keyword `{t.value}`")
        raise self.error(f"expected an expression, found {_describe(t)}")

    # ------------------------------------------------------------------ strings
    def parse_string(self, tok: Token) -> A.Str:
        raw: str = tok.value
        base = tok.span.start + 1           # offset of raw[0] in the file
        parts: list = []
        buf: list[str] = []
        i = 0
        n = len(raw)
        escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "0": "\0", "r": "\r", "e": "\x1b"}
        while i < n:
            ch = raw[i]
            if ch == "\\" and i + 1 < n and raw[i + 1] in "xu":
                width = 2 if raw[i + 1] == "x" else 4
                digits = raw[i + 2:i + 2 + width]
                if len(digits) != width or any(c not in "0123456789abcdefABCDEF" for c in digits):
                    raise AnvilError(f"bad escape: expected {width} hex digits after \\{raw[i + 1]}",
                                   Span(self.src, base + i, base + i + 2))
                buf.append(chr(int(digits, 16)))
                i += 2 + width
            elif ch == "\\" and i + 1 < n:
                buf.append(escapes.get(raw[i + 1], "\\" + raw[i + 1]))
                i += 2
            elif ch == "{" and i + 1 < n and raw[i + 1] == "{":
                buf.append("{")
                i += 2
            elif ch == "}" and i + 1 < n and raw[i + 1] == "}":
                buf.append("}")
                i += 2
            elif ch == "{":
                # find the matching close brace and the top-level ':' (format spec)
                depth = 0
                j = i + 1
                colon = None
                while j < n:
                    c = raw[j]
                    if c in "([{":
                        depth += 1
                    elif c in ")]":
                        depth -= 1
                    elif c == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif c == ":" and depth == 0 and colon is None:
                        colon = j
                    elif c in "\"'":
                        raise AnvilError("string literals are not allowed inside `{...}` interpolation",
                                       Span(self.src, base + j, base + j + 1))
                    j += 1
                if j >= n:
                    raise AnvilError("unclosed `{` in string", Span(self.src, base + i, base + i + 1),
                                   help="write `{{` for a literal brace")
                expr_end = colon if colon is not None else j
                expr_text = raw[i + 1:expr_end]
                if not expr_text.strip():
                    raise AnvilError("empty `{}` in string", Span(self.src, base + i, base + j + 1),
                                   help="write `{{}}` for literal braces")
                spec = raw[colon + 1:j].strip() if colon is not None else ""
                if buf:
                    parts.append("".join(buf))
                    buf = []
                sub = tokenize(self.src, base=base + i + 1, text=expr_text, interactive_indent=False)
                p = Parser(self.src, sub)
                e = p.parse_expr()
                if p.tok.kind != "EOF":
                    raise p.error(f"unexpected {_describe(p.tok)} in interpolated expression")
                parts.append(A.Interp(Span(self.src, base + i, base + j + 1), e, spec))
                i = j + 1
            elif ch == "}":
                raise AnvilError("single `}` in string", Span(self.src, base + i, base + i + 1),
                               help="write `}}` for a literal brace")
            else:
                buf.append(ch)
                i += 1
        if buf or not parts:
            parts.append("".join(buf))
        return A.Str(tok.span, parts)


def parse(src: SourceFile) -> A.Program:
    return Parser(src, tokenize(src)).parse_program()


def parse_text(text: str, path: str = "<input>") -> A.Program:
    return parse(SourceFile(path, text))
