import time
from sdc.io.csv_ext import to_varname
from typing import NamedTuple


class TestCase(NamedTuple):
    name: str
    params: str = ''
    size: list = []
    call_expression: str = None



def to_varname_without_excess_underscores(string):
    """Removing excess underscores from the string."""
    return '_'.join(i for i in to_varname(string).split('_') if i)


def gen(cases, method, class_add, typ, prefix=''):
    for test_params in cases:
        test_name_parts = ['test', typ, prefix, test_params.name, test_params.params]
        test_name = to_varname_without_excess_underscores('_'.join(test_name_parts))

        setattr(class_add, test_name, method(test_params.name, test_params.params,
                                             test_params.size, prefix, test_params.call_expression))


def test_gen(name, params, data_length, prefix, call_expression):
    func_name = 'func'
    input_data = 'input_data'
    if call_expression is None:
        prefix_as_list = [prefix] if prefix else []
        expr_parts = ['input_data'] + prefix_as_list + ['{}({})'.format(name, params)]
        call_expression = '.'.join(expr_parts)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{input_data}', '{call_expression}'), '{name}', {data_length})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def test_gen_two_par(name, params, data_length, prefix, call_expression):
    func_name = 'func'
    input_data = 'A, B'
    if call_expression is None:
        prefix_as_list = [prefix] if prefix else []
        expr_parts = ['A'] + prefix_as_list + ['{}(B, {})'.format(name, params)]
        call_expression = '.'.join(expr_parts)

    func_text = f"""\
def {func_name}(self):
  self._test_binary_operations(usecase_gen('{input_data}', '{call_expression}'), '{name}', {data_length})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def usecase_gen(input_data, function_called):
    func_name = 'func'

    func_text = f"""\
def {func_name}({input_data}):
  start_time = time.time()
  res = {function_called}
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    func = loc_vars[func_name]

    return func


def generate_test_cases(cases, class_add, type, prefix=''):
    return gen(cases, test_gen, class_add, type, prefix)


def generate_test_cases_two_params(cases, class_add, type, prefix=''):
    return gen(cases, test_gen_two_par, class_add, type, prefix)
