from typing import Iterable

from pysmt.smtlib.commands import DECLARE_FUN, ASSERT, DECLARE_CONST

SMT2_BOOL: str = 'Bool'
SMT2_INT: str = 'Int'
SMT2_TRUE: str = 'true'
SMT2_FORALL: str = 'forall'
SMT2_IMPLIES: str = '=>'
SMT2_AND: str = 'and'


def smt2_declare_fun(name: str, param_types: Iterable[str],
                     return_type: str) -> str:
    return f'({DECLARE_FUN} {name} ' \
           f'({" ".join(param_types)}) {return_type})'


def smt2_declare_const(name: str, type_: str) -> str:
    return f'({DECLARE_CONST} {name} {type_})'


def smt2_fun_call(name: str, exprs: Iterable[str]) -> str:
    return f'({name} {" ".join(exprs)})'


def smt2_assert(expr: str) -> str:
    return f'({ASSERT} {expr})'


def smt2_forall(params: Iterable[tuple[str, str]], expr: str) -> str:
    return f'({SMT2_FORALL} ' \
           f'({" ".join(f"({n} {t})" for n, t in params)}) {expr})'


def smt2_implies(l_expr: str, r_expr: str) -> str:
    return f'({SMT2_IMPLIES} {l_expr} {r_expr})'


def smt2_and(exprs: list[str]) -> str:
    return f'({SMT2_AND} {" ".join(exprs)})'
