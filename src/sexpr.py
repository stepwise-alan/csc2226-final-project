import re
from typing import Union


def to_c_expr(n: Union['Node', str]) -> str:
    match n:
        case Node():
            return n.c_expr()
        case '#f' | 'false':
            return '0'
        case '#t' | 'true':
            return '1'
        case _:
            return n


class Node:
    def __init__(self, children: list[Union['Node', str]]):
        self.children = children

    def __str__(self) -> str:
        return '(' + ' '.join(str(c) for c in self.children) + ')'

    @staticmethod
    def _c_op(op: str) -> str:
        match op:
            case 'and':
                return '&&'
            case 'or':
                return '||'
            case _:
                return op

    def c_expr(self) -> str:
        match self.children:
            case [arg] | ['and' | 'or', arg]:
                return to_c_expr(arg)
            case ['not', arg]:
                return '!' + to_c_expr(arg)
            case [op, arg]:
                return op + to_c_expr(arg)
            case ['=', arg0, arg1]:
                return f'({to_c_expr(arg0)} == {to_c_expr(arg1)})'
            case ['xor', arg0, arg1]:
                return f'(!{to_c_expr(arg0)} != !{to_c_expr(arg1)})'
            case ['ite', arg0, arg1, arg2]:
                return f'({to_c_expr(arg0)} ? {to_c_expr(arg1)} : {to_c_expr(arg2)})'
            case [op, *args]:
                return '(' + to_c_expr(op).join(to_c_expr(arg) for arg in args) + ')'


def parse_s_expr(s_expr: str) -> Union[Node, str]:
    nodes: list[Union[Node, str]] = parse_s_exprs(s_expr)
    assert len(nodes) == 1, f"expected 1 s expr, but got {len(nodes)}"
    return nodes[0]


def parse_s_exprs(s_exprs: str) -> list[Union[Node, str]]:
    stack: list[list[Union[Node, str]]] = []
    current: list[Union[Node, str]] = []
    for match in re.finditer(r'''(?mx)\s*(?:
            (?P<left_brace>\()|
            (?P<right_brace>\))|
            (?P<true>true)|
            (?P<false>false)|
            (?P<integer>-?\d+)|
            (?P<float>-?\d+\.\d+)|
            (?P<quoted_string>"[^"]*")|
            (?P<string>[^(^)\s]+))''', s_exprs):
        kind, value = [(t, v) for t, v in match.groupdict().items() if v][0]
        match kind:
            case 'left_brace':
                stack.append(current)
                current = []
            case 'right_brace':
                assert stack, "unmatched right brace"
                node: Node = Node(current)
                current = stack.pop()
                current.append(node)
            case 'true' | 'false':
                current.append(value)
            case 'integer':
                current.append(value)
            case 'float':
                current.append(value)
            case 'quoted_string':
                current.append(value)
            case 'string':
                current.append(value)
            case _:
                assert False, f"unknown value {value} has unknown kind {kind}"
    assert not stack, "unmatched left brace"
    assert current, "no s exprs found"
    return current
