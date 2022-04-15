from pycparser import c_ast, c_parser


def c_int_type_decl(declname: str) -> c_ast.TypeDecl:
    return c_ast.TypeDecl(declname=declname, quals=[], align=[],
                          type=c_ast.IdentifierType(['int']))


def c_int_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return c_ast.Decl(name=name, quals=[], align=[], storage=[],
                      funcspec=[], type=c_int_type_decl(name),
                      init=init, bitsize=None, coord=None)


def c_id(name: str) -> c_ast.ID:
    return c_ast.ID(name=name)


def c_func_call(name: str, args: list[c_ast.Node] = None) -> c_ast.FuncCall:
    return c_ast.FuncCall(name=c_ast.ID(name=name), args=c_ast.ExprList(exprs=args or []))


def c_neg(arg: c_ast.Node) -> c_ast.Node:
    return c_ast.UnaryOp('!', arg)


def parse_c_expr(c_expr: str) -> c_ast.Node:
    c_expr = "int x = (" + c_expr + ");"
    return c_parser.CParser().parse(c_expr).ext[0].init
