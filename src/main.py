#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
from re import Match
from typing import Optional, AnyStr, Container, Iterable, Iterator

import pycparser
import pycparser_fake_libc
import pysmt.shortcuts
from pycparser import c_ast, c_generator
from pysmt.fnode import FNode
from pysmt.typing import INT

from c_utils import c_int_decl, c_id, c_func_call, parse_c_expr, \
    c_neg, c_equal, c_not_equal, c_and
from simplify import get_simplified_formula, formula_to_c_expr
from smt2_utils import smt2_and, smt2_fun_call, smt2_declare_fun, \
    smt2_forall, smt2_assert, smt2_implies, SMT2_BOOL, SMT2_TRUE

cpp_path: str = 'cpp'
seahorn_path: str = 'sea'
z3_path: str = 'z3'
timeout: int = 500
wsl: bool = False

SEAHORN_SAT: str = 'sat'
SEAHORN_UNSAT: str = 'unsat'

SEAHORN_INCLUDE: str = '''extern void __VERIFIER_assume (int);
extern void __VERIFIER_error ();
__attribute__((__noreturn__)) extern void __VERIFIER_error (void);
#define assert(X) if(!(X)){__VERIFIER_error ();}
#define assume __VERIFIER_assume
extern int nd();
'''

P_INIT_FUNCTION_NAME: str = 'p_init'
TARGET_FUNCTION_NAME: str = 'target'
ND_FUNCTION_NAME: str = 'nd'


def parse_file(path: str) -> c_ast.FileAST:
    return pycparser.parse_file(
        path, use_cpp=True, cpp_path=cpp_path,
        cpp_args=['-I', pycparser_fake_libc.directory])


class InputInitializer(c_ast.NodeVisitor):
    def __init__(self):
        self.inputs: list[str] = []

    def visit_Decl(self, node: c_ast.Decl):
        if not node.init:
            node.init = c_func_call(ND_FUNCTION_NAME)
            self.inputs.append(node.name)


class FeatureAdder(c_ast.NodeVisitor):
    """A visitor that adds all features to target function definitions,
    declarations and calls."""

    def __init__(self, features: Iterable[str], targets: Container[str]):
        super().__init__()
        self.targets: Container[str] = targets
        self.ids: list[c_ast.ID] = [c_id(f) for f in features]
        self.decls: list[c_ast.Decl] = [c_int_decl(f) for f in features]

    def visit_FuncDecl(self, node: c_ast.FuncDecl) -> None:
        if node.type.declname in self.targets:
            if node.args:
                node.args.params.extend(self.decls)
            else:
                node.args = c_ast.ParamList(params=self.decls)
        self.generic_visit(node)

    def visit_FuncCall(self, node: c_ast.FuncCall) -> None:
        if node.name.name in self.targets:
            node.args.exprs.extend(self.ids)
        self.generic_visit(node)


def replace_extension(path: str, suffix: str) -> str:
    return os.path.splitext(path)[0] + suffix


def insert_lines_before(text: str, before: str, insert_lines: list[str],
                        prefix: str, count: int = -1) -> str:
    return text.replace(prefix + before,
                        prefix + prefix.join(insert_lines) + prefix + before,
                        count)


def run(command: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            shlex.split(command), capture_output=True,
            universal_newlines=True, timeout=timeout, check=True)
    except subprocess.CalledProcessError as e:
        print("COMMAND FAILED ".ljust(80, '='))
        print(command)
        print("STDOUT ".ljust(80, '='))
        print(e.stdout)
        print("STDERR ".ljust(80, '='))
        print(e.stderr)
        exit(e.returncode)


def get_precondition_in_s_expr(log_filepath: str) -> str:
    preconditions: list[str] = []
    started: bool = False
    with open(log_filepath) as file:
        for line in file.readlines():
            if line.startswith(f'** expand-pob: {P_INIT_FUNCTION_NAME}'):
                started = True
            elif started:
                if line.strip() == '':
                    break
                preconditions.append(line)

    if len(preconditions) > 1:
        return smt2_and(preconditions)
    elif len(preconditions) == 1:
        return preconditions[0]
    return ''


def generate_smt2_file(new_c_filepath: str) -> str:
    smt2_filepath: str = replace_extension(new_c_filepath, '.smt2')
    run(f"{seahorn_path} smt --horn-format=pure-smt2 "
        f"{new_c_filepath} -o {smt2_filepath}")
    return smt2_filepath


def generate_generalization_file(smt2_filepath: str) -> str:
    with open(smt2_filepath) as file:
        content: str = file.read()

    var_names: list[str] = []
    var_types: list[str] = []
    matches: Iterator[Match[AnyStr]] = re.finditer(
        r'\([\s]*'
        r'=[\s]+'
        r'main@%_(\d+)_([^\s()]*)[\s]+'
        r'@nd_[^\s()]*[\s]*'
        r'\)', content)
    for match in matches:
        i: int = int(match.group(1))
        suffix: str = match.group(2)
        var_name: str = f"main@%_{i + 1}_{suffix}"
        var_type: str = re.search(rf'\([\s]*{var_name}[\s]+'
                                  rf'([^\s]+)[\s]*\)', content).group(1)
        var_names.append(var_name)
        var_types.append(var_type)

    target_fun_call: str = smt2_fun_call(TARGET_FUNCTION_NAME, var_names)
    p_init_fun_call: str = smt2_fun_call(P_INIT_FUNCTION_NAME, var_names)
    new_lines: list[str] = [
        smt2_declare_fun(TARGET_FUNCTION_NAME, var_types, SMT2_BOOL),
        smt2_declare_fun(P_INIT_FUNCTION_NAME, var_types, SMT2_BOOL),
        smt2_assert(smt2_forall(zip(var_names, var_types), smt2_implies(
            SMT2_TRUE, target_fun_call))),
        smt2_assert(smt2_forall(zip(var_names, var_types), smt2_implies(
            target_fun_call, p_init_fun_call)))]

    first_assert_match: Optional[Match[AnyStr]] = re.search(
        r'([\s]*)'
        r'(\([\s]*'
        r'assert)', content)
    first_nd_var_decl_match: Optional[Match[AnyStr]] = re.search(
        r'([\s]*)'
        r'(\([\s]*'
        r'=[\s]+'
        r'main@%_\d+_[^\s()]*[\s]+'
        r'@nd_[^\s()]*[\s]*'
        r'\))', content)

    assert first_assert_match, "no asserts in smt2 file"
    assert first_nd_var_decl_match, "no variables initialized to nd in smt2 file"

    content = insert_lines_before(content, first_assert_match[2],
                                  new_lines, first_assert_match[1], 1)
    content = insert_lines_before(content, first_nd_var_decl_match[2],
                                  [p_init_fun_call], first_nd_var_decl_match[1])

    new_smt2_filepath: str = replace_extension(smt2_filepath, '.smt2')
    with open(new_smt2_filepath, 'w+') as file:
        file.write(content)

    return new_smt2_filepath


def generate_log_file(new_smt2_filepath: str) -> str:
    log_filepath: str = replace_extension(new_smt2_filepath, '.log')
    run(f"{z3_path} proof=true fp.engine=spacer fp.spacer.order_children=2 "
        f"fp.xform.subsumption_checker=false fp.xform.inline_eager=false "
        f"fp.xform.inline_linear=false fp.spacer.trace_file={log_filepath} "
        f"-v:2 {new_smt2_filepath}")
    return log_filepath


def replace_p_init_variables(s_expr: str, variables: list[str]) -> str:
    while True:
        match: Optional[Match[AnyStr]] = re.search(
            rf'({P_INIT_FUNCTION_NAME}_(\d+)_[^\s()]*)', s_expr)
        if match is None:
            break
        old: str = match.group(1)
        new: str = variables[int(match.group(2))]
        s_expr = s_expr.replace(old, new)
    return s_expr


def add_assumes(assumes: list[c_ast.Node], body: c_ast.Compound):
    for i, node in enumerate(body.block_items):
        if isinstance(node, c_ast.FuncCall):
            # TODO
            # if node.name.name == 'assert':
            body.block_items = body.block_items[:i] + assumes \
                               + body.block_items[i:]
            return


def add_feature_decls(features: list[str], body: c_ast.Compound):
    body.block_items = [c_int_decl(f, c_func_call(ND_FUNCTION_NAME))
                        for f in features] + body.block_items


def get_cex_tuple_name(lines: list[AnyStr]) -> Optional[str]:
    in_nd: bool = False
    for line in lines:
        if not in_nd:
            if line.lstrip().startswith("define i32 @nd("):
                in_nd = True
        else:
            match: Optional[Match[AnyStr]] = re.search(
                r'getelementptr inbounds \(.*(@\d).*\)', line)
            if match:
                return match.group(1)
            elif line.strip() == "}":
                return None
    return None


def get_cex(ll_filepath: str, variables: list[str]):
    cex: dict[str, str] = {}

    with open(ll_filepath) as file:
        lines: list[AnyStr] = file.readlines()

    cex_tuple_name: str = get_cex_tuple_name(lines) or "@0"
    cex_prefix: str = f'{cex_tuple_name} = private constant'

    cex_line: Optional[str] = None
    for line in lines:
        if line.lstrip().startswith(cex_prefix):
            cex_line = line
            break

    if cex_line:
        match: Optional[Match[AnyStr]] = re.search(r'\[.*]\s*\[(.*)].*', cex_line)
        if match:
            cex_values: list[str] = match.group(1).split(',')
            if len(cex_values) >= len(variables):
                for i in range(len(variables)):
                    cex[variables[i]] = cex_values[i].split()[-1]
            else:
                count: int = len(cex_values)
                for i in range(count):
                    cex[variables[i]] = cex_values[count - i - 1].split()[-1]
                for i in range(count, len(variables)):
                    cex[variables[i]] = '0'
        elif cex_line.rstrip('\n').endswith('zeroinitializer'):
            for v in variables:
                cex[v] = '0'
        else:
            assert False, "cex extraction error"
    else:
        # input independent cex
        for v in variables:
            cex[v] = '0'

    return cex


def get_func_def(file_ast: c_ast.FileAST, name: str) -> Optional[c_ast.FuncDef]:
    for func_def in file_ast:
        if isinstance(func_def, c_ast.FuncDef) and func_def.decl.name == name:
            return func_def
    return None


def extant_file(x: str) -> str:
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError(f"{x} is not a file")
    return x


def get_namespace() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="A variability-aware model checker "
                    "using SeaHorn as the backend engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", help="C source code file to be checked",
                        type=extant_file, metavar="FILE")
    parser.add_argument("--features", help="feature variables", nargs='*',
                        required=True)
    parser.add_argument("--cpp", help="C Preprocessor path",
                        metavar='PATH', default=cpp_path)
    parser.add_argument("--sea", help="SeaHorn path",
                        metavar='PATH', default=seahorn_path)
    parser.add_argument("--z3", help="Z3 path",
                        metavar='PATH', default=z3_path)
    parser.add_argument("--timeout", help="timeout in seconds",
                        metavar='SECONDS', type=int, default=timeout)
    parser.add_argument("--wsl", help="enable if using Windows Subsystem "
                                      "for Linux (WSL)",
                        action='store_true')
    return parser.parse_args()


def update_globals(namespace: argparse.Namespace) -> None:
    global cpp_path, seahorn_path, z3_path, timeout, wsl
    cpp_path = namespace.cpp
    seahorn_path = namespace.sea
    z3_path = namespace.z3
    timeout = namespace.timeout
    wsl = namespace.wsl


def has_seahorn_returned_sat(process: subprocess.CompletedProcess) -> bool:
    result: str = process.stdout.rstrip().split('\n')[-1]
    if SEAHORN_UNSAT in result:
        return False
    elif SEAHORN_SAT in result:
        return True
    else:
        assert False, "seahorn does not return sat or unsat"


def get_all_function_names(file_ast: c_ast.FileAST) -> set[str]:
    function_names: set[str] = set()
    for n in file_ast:
        match n:
            case c_ast.FuncDef(decl=c_ast.Decl(name=name)):
                function_names.add(name)
    return function_names


def get_p_init_variable_mapping(s_expr: str, variables: list[str]):
    matches: Iterator[Match[AnyStr]] = re.finditer(
        rf'({P_INIT_FUNCTION_NAME}_(\d+)_[^\s()]*)', s_expr)
    mapping: dict[str, str] = {}
    for match in matches:
        i: int = int(match[2])
        p_init_var_name: str = match[1]
        actual_var_name: str = variables[i]
        mapping[p_init_var_name] = actual_var_name
    return mapping


def get_subs(mapping: dict[str, str]) -> dict[FNode, FNode]:
    subs: dict[FNode, FNode] = {}
    for k, v in mapping.items():
        subs[pysmt.shortcuts.Symbol(k, INT)] = pysmt.shortcuts.Symbol(v, INT)
    return subs


def require_conjunction(formula: FNode) -> None:
    assert formula.is_and(), "formula is not in the conjunction form"


def get_conjuncts(formula: FNode) -> list[FNode]:
    require_conjunction(formula)
    return formula.args()


def split_formula(formula: FNode, features: list[str]
                  ) -> tuple[FNode, FNode]:
    feature_formulas: list[FNode] = []
    input_formulas: list[FNode] = []
    for conjunct in get_conjuncts(formula):
        free_variables: list[FNode] = conjunct.get_free_variables()
        for v in free_variables:
            if str(v) not in features:
                input_formulas.append(conjunct)
                break
        else:
            feature_formulas.append(conjunct)
    return pysmt.shortcuts.And(feature_formulas), pysmt.shortcuts.And(input_formulas)


def simplify_feature_formula(feature_formula: FNode) -> FNode:
    free_variables: list[FNode] = feature_formula.get_free_variables()
    formulas: list[FNode] = []
    for v in free_variables:
        v_is_zero: FNode = pysmt.shortcuts.Equals(
            v, pysmt.shortcuts.Int(0))

        if pysmt.shortcuts.is_valid(pysmt.shortcuts.Implies(
                feature_formula, v_is_zero)):
            formulas.append(v_is_zero)
            continue

        v_is_not_zero: FNode = pysmt.shortcuts.NotEquals(
            v, pysmt.shortcuts.Int(0))

        if pysmt.shortcuts.is_valid(pysmt.shortcuts.Implies(
                feature_formula, v_is_not_zero)):
            formulas.append(v_is_not_zero)
    return pysmt.shortcuts.And(formulas)


def formula_to_c_ast(formula: FNode) -> c_ast.Node:
    return parse_c_expr(formula_to_c_expr(formula))


def bool_formula_to_c_ast(formula: FNode, neg: bool = False) -> Optional[c_ast.Node]:
    left: FNode
    right: FNode
    if formula.is_equals():
        left, right = formula.args()
        if left.is_symbol():
            if right.is_constant():
                value: int = right.constant_value()
                if value == 0:
                    return c_id(left.symbol_name(), not neg)
                else:
                    return c_id(left.symbol_name(), neg)
            elif right.is_symbol():
                if neg:
                    return c_equal(c_id(left.symbol_name()), c_id(right.symbol_name()))
                else:
                    return c_not_equal(c_id(left.symbol_name()), c_id(right.symbol_name()))
        elif right.is_symbol():
            if left.is_constant():
                value: int = left.constant_value()
                if value == 0:
                    return c_id(right.symbol_name(), not neg)
                else:
                    return c_id(right.symbol_name(), neg)
    elif formula.is_lt():
        left, right = formula.args()
        if right.is_symbol():
            if left.is_constant() and left.constant_value() >= 0:
                return c_id(right.symbol_name(), neg)
        elif left.is_symbol():
            if right.is_constant() and right.constant_value() <= 0:
                return c_id(right.symbol_name(), neg)
    elif formula.is_le():
        left, right = formula.args()
        if right.is_symbol():
            if left.is_constant() and left.constant_value() > 0:
                return c_id(right.symbol_name(), neg)
        elif left.is_symbol():
            if right.is_constant() and right.constant_value() < 0:
                return c_id(right.symbol_name(), neg)
    elif formula.is_not():
        return bool_formula_to_c_ast(formula.arg(0), not neg)
    return None


def get_feature_c_ast(feature_formula: FNode) -> c_ast.Node:
    result: Optional[c_ast.Node] = None
    for conjunct in get_conjuncts(feature_formula):
        c_conjunct: Optional[c_ast.Node] = bool_formula_to_c_ast(conjunct)
        if c_conjunct is None:
            return formula_to_c_ast(feature_formula)
        if result is None:
            result = c_conjunct
        else:
            result = c_and(result, c_conjunct)
    return result


def generate_c_code(node: c_ast.Node) -> str:
    return c_generator.CGenerator().visit(node)


def main() -> None:
    namespace: argparse.Namespace = get_namespace()
    update_globals(namespace)

    c_filepath: str = namespace.infile
    if wsl:
        c_filepath = c_filepath.replace('\\', '/')
    features: list[str] = namespace.features
    assumes: list[c_ast.Node] = []

    i: int = 0
    while True:
        file_ast: c_ast.FileAST = parse_file(c_filepath)
        main_func_def: Optional[c_ast.FuncDef] = get_func_def(file_ast, 'main')

        function_names: set[str] = get_all_function_names(file_ast)
        function_names.remove('main')
        feature_adder: FeatureAdder = FeatureAdder(features, function_names)
        feature_adder.visit(file_ast)
        input_initializer: InputInitializer = InputInitializer()
        if main_func_def:
            input_initializer.visit(main_func_def.body)
        inputs: list[str] = input_initializer.inputs

        add_feature_decls(features, main_func_def.body)
        add_assumes(assumes, main_func_def.body)

        new_c_filepath: str = replace_extension(c_filepath, f'_{i}.c')
        with open(new_c_filepath, 'w+') as file:
            file.write(SEAHORN_INCLUDE + '\n' * 2 + generate_c_code(file_ast))

        ll_filepath: str = replace_extension(new_c_filepath, '.ll')
        process: subprocess.CompletedProcess = run(
            f'{seahorn_path} pf {new_c_filepath} --cex={ll_filepath}')

        if not has_seahorn_returned_sat(process):
            print("No more all counter examples.")
            break

        smt2_filepath: str = generate_smt2_file(new_c_filepath)
        new_smt2_filepath: str = generate_generalization_file(smt2_filepath)
        log_filepath: str = generate_log_file(new_smt2_filepath)
        s_expr: str = get_precondition_in_s_expr(log_filepath)

        mapping: dict[str, str] = get_p_init_variable_mapping(s_expr, features + inputs)
        formula: FNode = get_simplified_formula(s_expr, mapping.keys())
        subs: dict[FNode, FNode] = get_subs(mapping)
        formula = formula.substitute(subs)

        feature_formula: FNode
        input_formula: FNode
        feature_formula, input_formula = split_formula(formula, features)
        feature_c_ast: c_ast.Node = get_feature_c_ast(feature_formula)
        assumes.append(c_func_call('assume', [c_neg(feature_c_ast)]))

        print(f"Featured Counter Example {i + 1}")
        print("\tFeatures:", "\t", generate_c_code(feature_c_ast))
        print("\tInputs:  ", "\t", input_formula.serialize())
        print("")

        i += 1


if __name__ == '__main__':
    main()
