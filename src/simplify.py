import io
from typing import Optional, Iterable

from pysmt.fnode import FNode
from pysmt.shortcuts import *
from pysmt.smtlib.parser import get_formula

from smt2_utils import smt2_assert, SMT2_INT, smt2_declare_const

replace_rules: list[tuple[str, str]] = [
    (" = ", " == "),
    (" and ", " && "),
    (" or ", " || "),
    ("#f", "0"),
    ("#t", "1"),
    (" & ", " && "),
    (" | ", " || "),
]


def _c_expr_clean_up(c_expr: str):
    for old, new in replace_rules:
        c_expr = c_expr.replace(old, new)
    return c_expr


def formula_to_c_expr(formula: FNode) -> str:
    return _c_expr_clean_up(serialize(formula))


def get_simplified_formula(s_expr: str, variables: Iterable[str]) -> FNode:
    reset_env()
    script: str = ''
    for v in variables:
        script += smt2_declare_const(v, SMT2_INT) + '\n'
    script += smt2_assert(s_expr)
    formula: FNode = get_formula(io.StringIO(script))
    return self_subsume(simplify(formula))


self_subsume_cache: dict[FNode, FNode] = {}


def self_subsume(formula: FNode) -> FNode:
    result: Optional[FNode] = self_subsume_cache.get(formula)
    if result:
        return result

    result = _self_subsume(formula)
    self_subsume_cache[formula] = result
    self_subsume_cache[result] = result
    return result


def _self_subsume(formula: FNode) -> FNode:
    if formula.is_and() or formula.is_or():
        args: list[FNode] = [self_subsume(arg) for arg in formula.args()]
        formula_is_and: bool = formula.is_and()

        while True:
            for i, arg in enumerate(args):
                others: list[FNode] = args[:i] + args[i + 1:]
                implies: Implies
                if formula_is_and:
                    implies = Implies(And(others), arg)
                else:
                    implies = Implies(arg, Or(others))
                if is_valid(implies):
                    args = others
                    break
            else:
                break

        if formula_is_and:
            return And(args)
        else:
            return Or(args)

    return formula
