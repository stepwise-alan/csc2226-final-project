from typing import Union

from pycparser import c_ast, c_parser

ASSUME_FUNCTION_NAME: str = 'assume'
ND_INT_FUNCTION_NAME: str = 'nd_int'
ND_BOOL_FUNCTION_NAME: str = 'nd_bool'


def c_int(value: int) -> c_ast.Constant:
    return c_ast.Constant(type='int', value=value)


def c_int_type_decl(declname: str) -> c_ast.TypeDecl:
    return c_ast.TypeDecl(declname=declname, quals=[], align=[],
                          type=c_ast.IdentifierType(['int']))


def c_bool_type_decl(declname: str) -> c_ast.TypeDecl:
    return c_ast.TypeDecl(declname=declname, quals=[], align=[],
                          type=c_ast.IdentifierType(['_Bool']))


def c_int_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return c_ast.Decl(name=name, quals=[], align=[], storage=[],
                      funcspec=[], type=c_int_type_decl(name),
                      init=init, bitsize=None, coord=None)


def c_bool_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return c_ast.Decl(name=name, quals=[], align=[], storage=[],
                      funcspec=[], type=c_bool_type_decl(name),
                      init=init, bitsize=None, coord=None)


def c_id(name: str, neg: bool = False) -> Union[c_ast.ID, c_ast.UnaryOp]:
    if neg:
        return c_neg(c_ast.ID(name=name))
    else:
        return c_ast.ID(name=name)


def c_nd_int() -> c_ast.FuncCall:
    return c_func_call(ND_INT_FUNCTION_NAME)


def c_nd_bool() -> c_ast.FuncCall:
    return c_func_call(ND_BOOL_FUNCTION_NAME)


def c_assume(arg: c_ast.Node) -> c_ast.FuncCall:
    return c_func_call(ASSUME_FUNCTION_NAME, arg)


def c_func_call(name: str, *args: c_ast.Node) -> c_ast.FuncCall:
    return c_ast.FuncCall(name=c_ast.ID(name=name),
                          args=c_ast.ExprList(exprs=args or []))


def c_neg(arg: c_ast.Node) -> c_ast.UnaryOp:
    return c_ast.UnaryOp('!', arg)


def c_equal(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('==', left, right)


def c_not_equal(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('!=', left, right)


def c_and(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('&&', left, right)


def c_or(left: c_ast.Node, right: c_ast.Node) -> c_ast.BinaryOp:
    return c_ast.BinaryOp('||', left, right)


def parse_c_expr(c_expr: str) -> c_ast.Node:
    c_expr = "int x = (" + c_expr + ");"
    return c_parser.CParser().parse(c_expr).ext[0].init
