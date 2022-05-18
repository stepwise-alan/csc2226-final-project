import io
import itertools
import os
import subprocess
import sys
import tempfile
from typing import Final, Iterable, Container, Optional, Sequence

import antlr4
import llvmlite.binding as ll_ast
import pycparser
import pycparser.c_generator
import pycparser_fake_libc
import pysmt.shortcuts
import pysmt.smtlib.commands
import pysmt.smtlib.parser
import pysmt.typing
from pycparser import c_ast
from pysmt import fnode

from .SMTLIBv2Lexer import SMTLIBv2Lexer
from .SMTLIBv2Parser import SMTLIBv2Parser

DEFAULT_CPP_PATH: Final[str] = 'cpp'
DEFAULT_SEAHORN_PATH: Final[str] = 'sea'
DEFAULT_Z3_PATH: Final[str] = 'z3'
DEFAULT_TIMEOUT: Final[int] = 500
DEFAULT_USE_NORMALIZED_PATH: Final[bool] = False

_MAIN_FUNCTION_NAME: Final[str] = 'main'
_ASSUME_FUNCTION_NAME: Final[str] = 'assume'
_ASSERT_FUNCTION_NAME: Final[str] = 'assert'
_P_INIT_FUNCTION_NAME: Final[str] = 'p_init'
_TARGET_FUNCTION_NAME: Final[str] = 'target'
_ND_FUNCTION_PREFIX: Final[str] = '__nd_'

_C_KEYWORD_INT: Final[str] = 'int'
_C_KEYWORD_BOOL: Final[str] = '_Bool'
_C_KEYWORD_EXTERN: Final[str] = 'extern'
_C_KEYWORD_VOID: Final[str] = 'void'

_SMT2_KEYWORD_AND: Final[str] = 'and'
_SMT2_KEYWORD_INT: Final[str] = 'Int'
_SMT2_KEYWORD_BOOL: Final[str] = 'Bool'
_SMT2_KEYWORD_IMPLIES: Final[str] = '=>'
_SMT2_KEYWORD_TRUE: Final[str] = 'true'
_SMT2_KEYWORD_FORALL: Final[str] = 'forall'

_SEAHORN_SAT: Final[str] = 'sat'
_SEAHORN_UNSAT: Final[str] = 'unsat'
_SEAHORN_INCLUDE: Final[str] = f'''
extern void __VERIFIER_error (void);
extern void __VERIFIER_assume (int);
#define {_ASSUME_FUNCTION_NAME} __VERIFIER_assume
#define {_ASSERT_FUNCTION_NAME}(X) (void)((X) || (__VERIFIER_error (), 0))
'''

_Z3_LOG_PRECONDITION: Final[str] = f'** expand-pob: {_P_INIT_FUNCTION_NAME}'


class FunctionNotDefinedException(Exception):
    pass


class UnexpectedSeaHornResultException(Exception):
    pass


def _c_id(name: str) -> c_ast.ID:
    return c_ast.ID(name=name)


def _c_identifier_type(type_name: str) -> c_ast.IdentifierType:
    return c_ast.IdentifierType(names=[type_name])


def _c_type(type_name: str, name: Optional[str] = None) -> c_ast.TypeDecl:
    return c_ast.TypeDecl(declname=name, quals=[], align=[],
                          type=_c_identifier_type(type_name))


def _c_decl(name: str, type_name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return c_ast.Decl(name=name, quals=[], align=[], storage=[],
                      funcspec=[], type=_c_type(type_name, name),
                      init=init, bitsize=None, coord=None)


def _c_func_call(name: str,
                 args: Optional[list[c_ast.Node]] = None) -> c_ast.FuncCall:
    return c_ast.FuncCall(name=c_ast.ID(name=name),
                          args=c_ast.ExprList(exprs=args or []))


def _c_func_decl(return_type: c_ast.Node, name: str,
                 args: Optional[list[c_ast.Node]] = None,
                 extern: bool = False) -> c_ast.Decl:
    if args is None:
        args = [c_ast.Typename(name=None, quals=[], align=None,
                               type=_c_type(_C_KEYWORD_VOID))]
    return c_ast.Decl(
        name=name, quals=[], align=[],
        storage=[_C_KEYWORD_EXTERN] if extern else [],
        funcspec=[],
        type=c_ast.FuncDecl(args=c_ast.ParamList(params=args or []),
                            type=return_type),
        init=None, bitsize=None)


def _c_assume(arg: c_ast.Node) -> c_ast.FuncCall:
    return _c_func_call(_ASSUME_FUNCTION_NAME, [arg])


def _c_neg(arg: c_ast.Node) -> c_ast.UnaryOp:
    return c_ast.UnaryOp('!', arg)


def _get_nd_function_name(name: str) -> str:
    return f'{_ND_FUNCTION_PREFIX}{name}'


def _get_nd_variable_name(name: str) -> Optional[str]:
    if name.startswith(_ND_FUNCTION_PREFIX):
        return name.removeprefix(_ND_FUNCTION_PREFIX)
    return None


def _get_smt2_variable_name(name: str) -> str:
    return f'main@%{name}_0'


def _get_p_init_variable_name(i: int) -> str:
    return f'{_P_INIT_FUNCTION_NAME}_{i}_n'


def _c_int_type(declname: str) -> c_ast.TypeDecl:
    return _c_type(_C_KEYWORD_INT, declname)


def _c_bool_type(declname: str) -> c_ast.TypeDecl:
    return _c_type(_C_KEYWORD_BOOL, declname)


def _c_int_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return _c_decl(name, _C_KEYWORD_INT, init)


def _c_bool_decl(name: str, init: c_ast.Node = None) -> c_ast.Decl:
    return _c_decl(name, _C_KEYWORD_BOOL, init)


def _c_nd_int_decl(name: str) -> c_ast.Decl:
    return _c_int_decl(name, _c_nd_call(name))


def _c_nd_bool_decl(name: str) -> c_ast.Decl:
    return _c_bool_decl(name, _c_nd_call(name))


def _c_nd_func_decl(name: str, c_type: c_ast.Node) -> c_ast.Decl:
    return _c_func_decl(c_ast.TypeDecl(
        declname=_get_nd_function_name(name), quals=[], align=[], type=c_type),
        name, extern=True)


def _c_nd_int_func_decl(variable_name: str) -> c_ast.Decl:
    return _c_nd_func_decl(variable_name, _c_int_type(variable_name))


def _c_nd_bool_func_decl(variable_name: str) -> c_ast.Decl:
    return _c_nd_func_decl(variable_name, _c_bool_type(variable_name))


def _c_nd_call(name: str) -> c_ast.FuncCall:
    return _c_func_call(_get_nd_function_name(name))


def _get_function(file_ast: c_ast.FileAST,
                  function_name: str) -> c_ast.FuncDef:
    for n in file_ast:
        if isinstance(n, c_ast.FuncDef) and n.decl.name == function_name:
            return n
    raise FunctionNotDefinedException


def _get_function_names(file_ast: c_ast.FileAST,
                        exceptions: Container[str] = ()) -> set[str]:
    function_names: set[str] = set()
    for n in file_ast:
        if isinstance(n, c_ast.FuncDef) and n.decl.name not in exceptions:
            function_names.add(n.decl.name)
    return function_names


def _add_features_to_function_arguments(file_ast: c_ast.FileAST,
                                        features: Iterable[str]) -> None:
    class Visitor(c_ast.NodeVisitor):
        def __init__(self):
            super().__init__()

        def visit_FuncDecl(self, node: c_ast.FuncDecl) -> None:
            if node.type.declname in function_names:
                if node.args:
                    node.args.params.extend(decls)
                else:
                    node.args = c_ast.ParamList(params=decls)
            self.generic_visit(node)

        def visit_FuncCall(self, node: c_ast.FuncCall) -> None:
            if node.name.name in function_names:
                node.args.exprs.extend(ids)
            self.generic_visit(node)

    ids: list[c_ast.ID] = list(map(_c_id, features))
    decls: list[c_ast.Decl] = list(map(_c_bool_decl, features))

    function_names: set[str] = _get_function_names(
        file_ast, exceptions=(_MAIN_FUNCTION_NAME,))
    Visitor().visit(file_ast)


def _add_feature_decls(main_function: c_ast.FuncDef,
                       features: Iterable[str]) -> None:
    decls: list[c_ast.Decl] = list(map(_c_nd_bool_decl, features))
    main_function.body.block_items = decls + main_function.body.block_items


def _add_nd_func_decls(file_ast: c_ast.FileAST,
                       features: Iterable[str],
                       inputs: Iterable[str]) -> None:
    feature_decls: list[c_ast.Decl] = list(map(_c_nd_bool_func_decl, features))
    input_decls: list[c_ast.Decl] = list(map(_c_nd_int_func_decl, inputs))
    decls: list[c_ast.Decl] = []
    for n in file_ast:
        if isinstance(n, c_ast.FuncDef):
            decls.append(n.decl)
    file_ast.ext = feature_decls + input_decls + decls + file_ast.ext


def _get_inputs(main_function: c_ast.FuncDef) -> list[str]:
    class Visitor(c_ast.NodeVisitor):
        def __init__(self):
            self.inputs: list[str] = []

        def visit_Decl(self, node: c_ast.Decl) -> None:
            if not node.init:
                node.init = _c_nd_call(node.name)
                self.inputs.append(node.name)

    visitor: Visitor = Visitor()
    visitor.visit(main_function.body)
    return visitor.inputs


def _generate_c_code(node: c_ast.Node) -> str:
    return pycparser.c_generator.CGenerator().visit(node)


def _smt2_func_call(name: str, args: Iterable[str]) -> str:
    return f'({name} {" ".join(args)})'


def _smt2_func_decl(return_type_name: str, name: str,
                    arg_type_names: Iterable[str]) -> str:
    return f'({pysmt.smtlib.commands.DECLARE_FUN} {name} ' \
           f'({" ".join(arg_type_names)}) {return_type_name})'


def _smt2_declare_const(name: str, type_name: str) -> str:
    return f'({pysmt.smtlib.commands.DECLARE_CONST} {name} {type_name})'


def _smt2_assert(s_expr: str) -> str:
    return f'({pysmt.smtlib.commands.ASSERT} {s_expr})'


def _smt2_implies(left_s_expr: str, right_s_expr: str) -> str:
    return f'({_SMT2_KEYWORD_IMPLIES} {left_s_expr} {right_s_expr})'


def _smt2_forall(args: Iterable[tuple[str, str]], s_expr: str) -> str:
    return f'({_SMT2_KEYWORD_FORALL} ' \
           f'({" ".join(f"({n} {t})" for n, t in args)}) {s_expr})'


def _smt2_type(type_name: str) -> pysmt.typing.PySMTType:
    if type_name == _SMT2_KEYWORD_INT:
        return pysmt.shortcuts.INT
    elif type_name == _SMT2_KEYWORD_BOOL:
        return pysmt.shortcuts.BOOL
    assert False


def _smt2_symbol(name: str, smt2_type: pysmt.typing.PySMTType) -> fnode.FNode:
    return pysmt.shortcuts.Symbol(name, smt2_type)


def _is_unsat(process: subprocess.CompletedProcess) -> bool:
    result: str = process.stdout.rstrip().split('\n')[-1]
    if _SEAHORN_UNSAT in result:
        return True
    elif _SEAHORN_SAT in result:
        return False
    raise UnexpectedSeaHornResultException


def _get_variable_mapping(ll_filepath: str) -> dict[str, str]:
    with open(ll_filepath) as file:
        module_ref: ll_ast.ModuleRef = ll_ast.parse_assembly(file.read())

    mapping: dict[str, str] = {}
    for block in module_ref.get_function('main').blocks:
        assert block.is_block
        for instruction in block.instructions:
            assert instruction.is_instruction
            if instruction.opcode == 'call':
                for operand in instruction.operands:
                    assert operand.is_operand
                    name: Optional[str] = _get_nd_variable_name(operand.name)
                    if name:
                        mapping[_get_smt2_variable_name(instruction.name)] \
                            = name
    return mapping


def _generate_new_c_file(new_c_filepath: str, file_ast: c_ast.FileAST) -> None:
    with open(new_c_filepath, 'w+') as file:
        file.write(_SEAHORN_INCLUDE + '\n' * 2 + _generate_c_code(file_ast))


def _get_separator(nodes: Sequence[antlr4.ParserRuleContext],
                   file_stream: antlr4.FileStream, default: str) -> str:
    if len(nodes) > 1:
        return file_stream.getText(
            nodes[0].stop.stop + 1,
            nodes[1].start.start - 1)
    else:
        return default


def _get_forall_terms(commands: Iterable[SMTLIBv2Parser.CommandContext]
                      ) -> list[SMTLIBv2Parser.TermContext]:
    forall_terms: list[SMTLIBv2Parser.TermContext] = []
    for command in commands:
        command: SMTLIBv2Parser.CommandContext
        if command.cmd_assert():
            term: SMTLIBv2Parser.TermContext = command.term(0)
            if term.GRW_Forall():
                forall_terms.append(term)
    return forall_terms


def _get_smt2_variable_names_and_type_names(
        forall_terms: list[SMTLIBv2Parser.TermContext], mapping: dict[str, str]
) -> tuple[list[str], list[str]]:
    smt2_var_names: list[str] = []
    smt2_type_names: list[str] = []
    for forall in forall_terms:
        for sorted_var in forall.sorted_var():
            sorted_var: SMTLIBv2Parser.Sorted_varContext
            variable_name: str = sorted_var.symbol().getText()
            if variable_name in mapping:
                smt2_var_names.append(variable_name)
                smt2_type_names.append(sorted_var.sort().getText())
    return smt2_var_names, smt2_type_names


def _generate_new_smt2_file(smt2_filepath: str, new_smt2_filepath: str,
                            mapping: dict[str, str]
                            ) -> tuple[list[str], list[str]]:
    file_stream: antlr4.FileStream = antlr4.FileStream(smt2_filepath)
    lexer: SMTLIBv2Lexer = SMTLIBv2Lexer(file_stream)
    parser: SMTLIBv2Parser = SMTLIBv2Parser(antlr4.CommonTokenStream(lexer))
    commands: list[SMTLIBv2Parser.CommandContext] \
        = parser.start().script().command()

    forall_terms: list[SMTLIBv2Parser.TermContext] = _get_forall_terms(commands)

    smt2_var_names, smt2_type_names = \
        _get_smt2_variable_names_and_type_names(forall_terms, mapping)

    target_call: str = _smt2_func_call(_TARGET_FUNCTION_NAME, smt2_var_names)
    p_init_call: str = _smt2_func_call(_P_INIT_FUNCTION_NAME, smt2_var_names)

    position: int = 0
    new_position: int
    with open(new_smt2_filepath, 'w+') as file:
        for command in commands:
            if not command.cmd_assert():
                continue

            new_position = command.start.start
            file.write(file_stream.getText(position, new_position - 1))
            position = new_position

            separator: str = _get_separator(commands, file_stream, '\n')

            file.write(_smt2_func_decl(
                _SMT2_KEYWORD_BOOL, _P_INIT_FUNCTION_NAME, smt2_type_names))
            file.write(separator)
            file.write(_smt2_func_decl(
                _SMT2_KEYWORD_BOOL, _TARGET_FUNCTION_NAME, smt2_type_names))
            file.write(separator)
            file.write(_smt2_assert(_smt2_forall(
                zip(smt2_var_names, smt2_type_names),
                _smt2_implies(_SMT2_KEYWORD_TRUE, target_call))))
            file.write(separator)
            file.write(_smt2_assert(_smt2_forall(
                zip(smt2_var_names, smt2_type_names),
                _smt2_implies(target_call, p_init_call))))
            break

        for forall in forall_terms:
            term0: SMTLIBv2Parser.TermContext = forall.term(0)
            if term0.GRW_Exclamation():
                term1: SMTLIBv2Parser.TermContext = term0.term(0)
                assert term1.GRW_Let()

            for var_binding in term1.var_binding():
                var_binding: SMTLIBv2Parser.Var_bindingContext
                term2: SMTLIBv2Parser.TermContext = var_binding.term()

                q: Optional[SMTLIBv2Parser.Qual_identifierContext] \
                    = term2.qual_identifier()
                assert q and q.getText() == _SMT2_KEYWORD_AND

                terms: list[SMTLIBv2Parser.TermContext] = term2.term()

                new_position = terms[0].stop.stop
                file.write(file_stream.getText(position, new_position))
                position = new_position + 1

                file.write(_get_separator(terms, file_stream, ' '))
                file.write(p_init_call)

        file.write(file_stream.getText(position, file_stream.size))

    return smt2_var_names, smt2_type_names


def _get_s_exprs(log_filepath: str) -> list[str]:
    _s_exprs: list[str] = []
    started: bool = False
    with open(log_filepath) as file:
        for line in file.readlines():
            if line.startswith(_Z3_LOG_PRECONDITION):
                started = True
            elif started:
                line = line.strip()
                if line == '':
                    break
                _s_exprs.append(line)
    return _s_exprs


def _get_formula(s_exprs: list[str],
                 smt2_var_names: list[str],
                 smt2_type_names: list[str],
                 mapping: dict[str, str]) -> fnode.FNode:
    pysmt.shortcuts.reset_env()
    p_init_var_names: list[str] = list(map(
        _get_p_init_variable_name, range(len(smt2_var_names))))
    declares: map = map(_smt2_declare_const, p_init_var_names, smt2_type_names)
    asserts: map = map(_smt2_assert, s_exprs)
    formula: fnode.FNode = pysmt.smtlib.parser.get_formula(
        io.StringIO('\n'.join(itertools.chain(declares, asserts))))
    return formula.substitute({
        _smt2_symbol(p, t): _smt2_symbol(mapping[v], t)
        for p, v, t in zip(
            p_init_var_names,
            smt2_var_names,
            map(_smt2_type, smt2_type_names)
        )
    })


def _split_formula(formula: fnode.FNode, features: Iterable[str]
                   ) -> tuple[fnode.FNode, fnode.FNode]:
    feature_conjuncts: list[fnode.FNode] = []
    input_conjuncts: list[fnode.FNode] = []
    for conjunct in formula.args():
        free_variables: list[fnode.FNode] = conjunct.get_free_variables()
        for v in free_variables:
            if str(v) not in features:
                input_conjuncts.append(conjunct)
                break
        else:
            feature_conjuncts.append(conjunct)

    feature_formula: fnode.FNode = pysmt.shortcuts.And(feature_conjuncts)
    input_formula: fnode.FNode = pysmt.shortcuts.And(input_conjuncts)
    return feature_formula, input_formula


def _formula_to_c_expr(formula: fnode.FNode) -> str:
    c_expr: str = pysmt.shortcuts.serialize(formula)
    for old, new in (
            (" = ", " == "),
            (" and ", " && "),
            (" or ", " || "),
            ("#f", "0"),
            ("#t", "1"),
            (" & ", " && "),
            (" | ", " || "),
    ):
        c_expr = c_expr.replace(old, new)
    return c_expr


def _parse_c_expr(c_expr: str) -> c_ast.Node:
    text: str = f"int x = ({c_expr});"
    return pycparser.c_parser.CParser().parse(text).ext[0].init


def _add_assumes(assumes: list[c_ast.Node],
                 main_function: c_ast.FuncDef) -> None:
    body: c_ast.Compound = main_function.body
    for i, node in enumerate(body.block_items):
        if isinstance(node, c_ast.FuncCall):
            # if node.name.name == 'assert':
            body.block_items = body.block_items[:i] + assumes \
                               + body.block_items[i:]
            return


class VariabilityAwareModelChecker:
    def __init__(
            self,
            cpp_path: str = DEFAULT_CPP_PATH,
            seahorn_path: str = DEFAULT_SEAHORN_PATH,
            z3_path: str = DEFAULT_Z3_PATH,
            use_normalized_path: bool = DEFAULT_USE_NORMALIZED_PATH,
            timeout: int = DEFAULT_TIMEOUT,
    ):
        self._cpp_path: Final[str] = cpp_path
        self._seahorn_path: Final[str] = seahorn_path
        self._z3_path: Final[str] = z3_path
        self._use_normalized_path: Final[bool] = use_normalized_path
        self._timeout: Final[int] = timeout

    def _normalize_path(self, path: str) -> str:
        if self._use_normalized_path:
            return os.path.relpath(path).replace('\\', '/')
        return path

    def _parse_c_file(self, c_filepath: str) -> c_ast.FileAST:
        return pycparser.parse_file(
            filename=self._normalize_path(c_filepath),
            use_cpp=True, cpp_path=self._cpp_path,
            cpp_args=['-I', pycparser_fake_libc.directory])

    def _run(self, command: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command, capture_output=True, universal_newlines=True,
                timeout=self._timeout, check=True)
        except subprocess.CalledProcessError as e:
            print(command)
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            raise e

    def _run_seahorn(self, new_c_filepath: str, smt2_filepath: str,
                     ll_filepath: str) -> subprocess.CompletedProcess:
        return self._run(
            f'{self._seahorn_path} smt {new_c_filepath} --solve '
            f'--horn-format=pure-smt2 -o {smt2_filepath} --oll={ll_filepath}')

    def _run_z3(self, new_smt2_filepath: str, log_filepath: str
                ) -> subprocess.CompletedProcess:
        return self._run(
            f"{self._z3_path} "
            f"proof=true fp.engine=spacer fp.spacer.order_children=2 "
            f"fp.xform.subsumption_checker=false fp.xform.inline_eager=false "
            f"fp.xform.inline_linear=false "
            f"fp.spacer.trace_file={log_filepath} "
            f"-v:2 {new_smt2_filepath}")

    def _get_out_prefix(self, c_filepath: str, out_dir_path: str) -> str:
        head, tail = os.path.split(c_filepath)
        return self._normalize_path(os.path.join(
            out_dir_path, os.path.splitext(tail or os.path.basename(head))[0]))

    def _check(self, c_filepath: str, features: Iterable[str],
               out_dir_path: str) -> Iterable[tuple[fnode.FNode, fnode.FNode]]:
        file_ast: c_ast.FileAST = self._parse_c_file(c_filepath)

        main_function: c_ast.FuncDef = _get_function(
            file_ast, _MAIN_FUNCTION_NAME)

        _add_features_to_function_arguments(file_ast, features)
        _add_feature_decls(main_function, features)
        inputs: list[str] = _get_inputs(main_function)
        _add_nd_func_decls(file_ast, features, inputs)

        out_prefix: str = self._get_out_prefix(c_filepath, out_dir_path)

        i: int = 0
        while True:
            new_c_filepath: str = f'{out_prefix}.{i}.c'
            ll_filepath: str = f'{out_prefix}.{i}.ll'
            smt2_filepath: str = f'{out_prefix}.{i}.0.smt2'
            new_smt2_filepath: str = f'{out_prefix}.{i}.1.smt2'
            log_filepath: str = f'{out_prefix}.{i}.log'

            _generate_new_c_file(new_c_filepath, file_ast)

            if _is_unsat(self._run_seahorn(
                    new_c_filepath, smt2_filepath, ll_filepath)):
                break

            mapping: dict[str, str] = _get_variable_mapping(ll_filepath)
            smt2_var_names, smt2_type_names = _generate_new_smt2_file(
                smt2_filepath, new_smt2_filepath, mapping)

            self._run_z3(new_smt2_filepath, log_filepath)
            s_exprs: list[str] = _get_s_exprs(log_filepath)

            formula: fnode.FNode = _get_formula(
                s_exprs, smt2_var_names, smt2_type_names, mapping)
            feature_formula, input_formula = _split_formula(formula, features)

            c_expr: str = _formula_to_c_expr(feature_formula)
            feature_c_ast: c_ast.Node = _parse_c_expr(c_expr)

            _add_assumes([_c_assume(_c_neg(feature_c_ast))], main_function)
            yield feature_formula, input_formula

            i += 1

    def check(self, c_filepath: str, features: Iterable[str],
              out_dir_path: Optional[str]
              ) -> Iterable[tuple[fnode.FNode, fnode.FNode]]:
        if out_dir_path is not None:
            os.makedirs(out_dir_path, exist_ok=True)
            return self._check(c_filepath, features, out_dir_path)
        else:
            with tempfile.TemporaryDirectory() as out_dir_path:
                return self._check(c_filepath, features, out_dir_path)
