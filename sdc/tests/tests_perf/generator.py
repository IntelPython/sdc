import time
import numpy as np
from sdc.io.csv_ext import to_varname
from typing import NamedTuple


class TestCase(NamedTuple):
    name: str
    params: str = ''
    size: list = []
    call_expression: str = None
    data_num: int = 1
    input_data: list = []



def to_varname_without_excess_underscores(string):
    """Removing excess underscores from the string."""
    return '_'.join(i for i in to_varname(string).split('_') if i)


def generate_test_cases(cases, class_add, typ, prefix=''):
    for test_params in cases:
        test_name_parts = ['test', typ, prefix, test_params.name, test_params.params]
        test_name = to_varname_without_excess_underscores('_'.join(test_name_parts))

        setattr(class_add, test_name, test_gen(test_params.name, test_params.params,
                                               test_params.size, prefix, test_params.call_expression,
                                               test_params.data_num, test_params.input_data))


def test_gen(name, params, data_length, prefix, call_expression, data_num, input_data_func):
    func_name = 'func'
    if data_num == 1:
        input_data = 'input_data'
        if call_expression is None:
            prefix_as_list = [prefix] if prefix else []
            expr_parts = ['input_data'] + prefix_as_list + ['{}({})'.format(name, params)]
            call_expression = '.'.join(expr_parts)

    elif data_num == 2:
        input_data = 'A, B'
        if call_expression is None:
            prefix_as_list = [prefix] if prefix else []
            expr_parts = ['A'] + prefix_as_list + ['{}(B, {})'.format(name, params)]
            call_expression = '.'.join(expr_parts)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{input_data}', '{call_expression}'), name='{name}', total_data_length={data_length}, 
                              data_num={data_num}, input_data={input_data_func})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def usecase_gen(input_data, call_expr):
    func_name = 'func'

    func_text = f"""\
def {func_name}({input_data}):
  start_time = time.time()
  res = {call_expr}
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    func = loc_vars[func_name]

    return func
