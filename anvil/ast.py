"""Abstract syntax tree."""
from __future__ import annotations

from dataclasses import dataclass, field

from .source import Span


@dataclass(repr=False)
class Node:
    span: Span

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if k != "span")
        return f"{type(self).__name__}({fields})"


# ----------------------------------------------------------------------------- expressions

class Expr(Node):
    pass


@dataclass(repr=False)
class Name(Expr):
    id: str


@dataclass(repr=False)
class Num(Expr):
    value: int | float


@dataclass(repr=False)
class Bool(Expr):
    value: bool


@dataclass(repr=False)
class Interp(Node):
    expr: Expr
    spec: str          # format spec after ':' ('' if none)


@dataclass(repr=False)
class Str(Expr):
    parts: list        # list[str | Interp]

    @property
    def is_plain(self) -> bool:
        return all(isinstance(p, str) for p in self.parts)

    @property
    def plain(self) -> str:
        return "".join(p for p in self.parts if isinstance(p, str))


@dataclass(repr=False)
class ListLit(Expr):
    items: list[Expr]


@dataclass(repr=False)
class ListComp(Expr):
    """[elt for var in iter]: a list built at compile time (a tuple of values, or of models)."""
    elt: Expr
    var: Name
    iter: Expr


@dataclass(repr=False)
class TupleLit(Expr):
    items: list[Expr]


@dataclass(repr=False)
class Unary(Expr):
    op: str            # - + not √
    operand: Expr


@dataclass(repr=False)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(repr=False)
class Cond(Expr):
    cond: Expr
    then: Expr
    orelse: Expr


@dataclass(repr=False)
class Keyword(Node):
    name: str
    value: Expr


@dataclass(repr=False)
class Call(Expr):
    func: Expr
    args: list[Expr]
    kwargs: list[Keyword] = field(default_factory=list)


@dataclass(repr=False)
class Reduce(Expr):
    """Prefix reduction: `sum x[i] * y[i]`."""
    op: str
    operand: Expr


@dataclass(repr=False)
class Slice(Node):
    start: Expr | None
    stop: Expr | None
    step: Expr | None


@dataclass(repr=False)
class EllipsisIdx(Node):
    pass


@dataclass(repr=False)
class Subscript(Expr):
    value: Expr
    items: list        # list[Expr | Slice | EllipsisIdx]


@dataclass(repr=False)
class Attr(Expr):
    value: Expr
    name: str


# ----------------------------------------------------------------------------- types

@dataclass(repr=False)
class TypeExpr(Node):
    dtype: str | None          # 'f32' | 'i32' | None (default f32)
    dims: list[Expr] | None    # None => scalar of dtype


# ----------------------------------------------------------------------------- statements

class Stmt(Node):
    pass


@dataclass(repr=False)
class Assign(Stmt):
    targets: list[Name]        # several => tuple unpacking
    value: Expr


@dataclass(repr=False)
class AnnAssign(Stmt):
    target: Name
    type: TypeExpr
    value: Expr | None
    sample: Expr | None        # `x: T ~ dist`


@dataclass(repr=False)
class AugAssign(Stmt):
    target: Name
    op: str                    # + - * / // ** @
    value: Expr


@dataclass(repr=False)
class IndexAssign(Stmt):
    """Comprehension: `y[i, j] = expr [where u < 2, ...]`."""
    target: Name
    indices: list              # list[Name | EllipsisIdx]
    value: Expr
    where: list[tuple]         # list[(Name, Expr)]
    accumulate: bool = False


@dataclass(repr=False)
class SetItem(Stmt):
    """Item / slice assignment: `x[ptr] = v`, `board[y, x] = 3`, `x[2:5] += 1`."""
    target: Name
    items: list                # subscript items, as in an expression
    value: Expr
    op: str | None = None      # for augmented forms: + - * / ...
    where: list = field(default_factory=list)      # `x[2*k] += 1 where k < 3`: list[(Name, Expr)]


@dataclass(repr=False)
class ExprStmt(Stmt):
    expr: Expr


@dataclass(repr=False)
class Use(Stmt):
    """`use "model.anvil"`: the declarations of another file (functions, models, optimizers,
    constants), as if they were written here."""
    path: str


@dataclass(repr=False)
class Return(Stmt):
    value: Expr | None


@dataclass(repr=False)
class If(Stmt):
    cond: Expr
    body: list[Stmt]
    orelse: list[Stmt]


@dataclass(repr=False)
class For(Stmt):
    targets: list[Name]
    iter: Expr
    body: list[Stmt]
    static: bool = False       # `static for`: unrolled at compile time


@dataclass(repr=False)
class While(Stmt):
    cond: Expr
    body: list[Stmt]


@dataclass(repr=False)
class Assert(Stmt):
    cond: Expr
    message: Expr | None


@dataclass(repr=False)
class Break(Stmt):
    pass


@dataclass(repr=False)
class Continue(Stmt):
    pass


@dataclass(repr=False)
class Pass(Stmt):
    pass


@dataclass(repr=False)
class Minimize(Stmt):
    objective: Expr
    over: list[Expr] | None
    optimizer: Expr
    maximize: bool = False


@dataclass(repr=False)
class ConstDecl(Stmt):
    name: Name
    value: Expr


@dataclass(repr=False)
class ParamDecl(Stmt):
    name: Name
    type: TypeExpr
    init: Expr | None
    sample: Expr | None


@dataclass(repr=False)
class Param(Node):
    name: str
    type: TypeExpr | None
    default: Expr | None


@dataclass(repr=False)
class FnDecl(Stmt):
    name: Name
    params: list[Param]
    ret: TypeExpr | None
    body: list[Stmt] | Expr    # Expr for `fn f(x) = expr`


@dataclass(repr=False)
class ModelDecl(Stmt):
    name: Name
    params: list[Param]
    body: list[Stmt]


@dataclass(repr=False)
class OptimizerDecl(Stmt):
    name: Name
    params: list[Param]
    state: list[Name]
    step_params: list[Name]
    step_body: list[Stmt]


@dataclass(repr=False)
class Program(Node):
    body: list[Stmt]


def walk_assigned_names(stmts) -> list[str]:
    """Names assigned anywhere in a statement list (not descending into nested declarations)."""
    out: list[str] = []

    def visit(ss):
        for s in ss:
            if isinstance(s, Assign):
                out.extend(t.id for t in s.targets)
            elif isinstance(s, (AnnAssign, AugAssign)):
                out.append(s.target.id)
            elif isinstance(s, (IndexAssign, SetItem)):
                out.append(s.target.id)
            elif isinstance(s, ConstDecl):
                out.append(s.name.id)
            elif isinstance(s, If):
                visit(s.body)
                visit(s.orelse)
            elif isinstance(s, For):
                out.extend(t.id for t in s.targets)
                visit(s.body)
            elif isinstance(s, While):
                visit(s.body)
    visit(stmts)
    return out
