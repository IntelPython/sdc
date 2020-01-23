import time
from sdc.io.csv_ext import to_varname
from typing import NamedTuple


class TestCase(NamedTuple):
    name: str
    params: str
    size: int = None
    call_expression: str = None


def gen(cases, method, class_add, typ, prefix=''):
    for params in cases:
        test_params = TestCase(*params)
        name = test_params.name
        if test_params.params:
            name += "_" + test_params.params
        func_name = to_varname('test_{}_{}_{}'.format(typ, prefix, name))

        setattr(class_add, func_name, method(test_params.name, test_params.params,
                                             test_params.size, prefix, test_params.call_expression))


def test_gen(name, params, data_length, prefix, call_expression):
    func_name = 'func'
    if call_expression is None:
        call_expression = '{}{}({})'.format(prefix, name, params)
    input_data = 'input_data'
    function_called = '{}.{}'.format(input_data, call_expression)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{input_data}', '{function_called}'), '{name}', {data_length})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def test_gen_two_par(name, params, data_length, prefix, *args, **kwargs):
    func_name = 'func'
    input_data = 'A, B'
    function_called = 'A.{}{}(B, {})'.format(prefix, name, params)

    func_text = f"""\
def {func_name}(self):
  self._test_binary_operations(usecase_gen('{input_data}', '{function_called}'), '{name}', {data_length})
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
