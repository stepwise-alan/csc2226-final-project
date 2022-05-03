from typing import Union

from pycparser import c_ast, c_parser


def c_int_type_decl(declname: str) -> c_ast.TypeDecl:
    return c_ast.TypeDecl(declname=declname, quals=[], align=[],
                          type=c_ast.IdentifierType(['int']))


def c_int_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return c_ast.Decl(name=name, quals=[], align=[], storage=[],
                      funcspec=[], type=c_int_type_decl(name),
                      init=init, bitsize=None, coord=None)


def c_id(name: str, neg: bool = False) -> Union[c_ast.ID, c_ast.UnaryOp]:
    if neg:
        return c_neg(c_ast.ID(name=name))
    else:
        return c_ast.ID(name=name)


def c_func_call(name: str, args: list[c_ast.Node] = None) -> c_ast.FuncCall:
    return c_ast.FuncCall(name=c_ast.ID(name=name), args=c_ast.ExprList(exprs=args or []))


def c_neg(arg: c_ast.Node) -> c_ast.UnaryOp:
    return c_ast.UnaryOp('!', arg)


def c_equal(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('=', left, right)


def c_not_equal(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('!=', left, right)


def c_and(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('&&', left, right)


def parse_c_expr(c_expr: str) -> c_ast.Node:
    c_expr = "int x = (" + c_expr + ");"
    return c_parser.CParser().parse(c_expr).ext[0].init
