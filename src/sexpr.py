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
    line_num: int = 1
    line_start: int = 0
    for match in re.finditer(rf'''(?x)(?:
            (?P<LEFT_BRACE>\()|
            (?P<RIGHT_BRACE>\))|
            (?P<BOOLEAN>true|false)|
            (?P<INTEGER>-?\d+)|
            (?P<FLOAT>-?\d+\.\d+)|
            (?P<QUOTED_STRING>"[^"]*")|
            (?P<STRING>[^()\s]+)|
            (?P<NEWLINE>\n)|
            (?P<SKIP>\s+)|
            (?P<MISMATCH>.)
            )''', s_exprs):
        kind: str = match.lastgroup
        value: str = match.group()
        column: int = match.start() - line_start
        match kind:
            case 'LEFT_BRACE':
                stack.append(current)
                current = []
            case 'RIGHT_BRACE':
                assert stack, "unmatched right brace"
                node: Node = Node(current)
                current = stack.pop()
                current.append(node)
            case 'BOOLEAN':
                current.append(value)
            case 'INTEGER':
                current.append(value)
            case 'FLOAT':
                current.append(value)
            case 'QUOTED_STRING':
                current.append(value)
            case 'STRING':
                current.append(value)
            case 'NEWLINE':
                line_start = match.end()
                line_num += 1
                continue
            case 'SKIP':
                continue
            case 'MISMATCH':
                raise RuntimeError(f'{value!r} unexpected '
                                   f'on line {line_num} column {column}')
            case _:
                raise RuntimeError(f"{value!r} has unknown kind {kind!r} "
                                   f"on line {line_num} column{column}")
    assert not stack, "unmatched left brace"
    assert current, "no s-expressions found"
    return current
