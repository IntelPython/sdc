import time
import numpy as np
from sdc.io.csv_ext import to_varname
from typing import NamedTuple


class TestCase(NamedTuple):
    name: str
    size: list
    params: str = ''
    call_expression: str = None
    name_data: str = 'obj'
    names_extra_data: str = ''
    data_num: int = 1
    input_data: list = None


def to_varname_without_excess_underscores(string):
    """Removing excess underscores from the string."""
    return '_'.join(i for i in to_varname(string).split('_') if i)


def generate_test_cases(cases, class_add, typ, prefix=''):
    for test_params in cases:
        test_name_parts = ['test', typ, prefix, test_params.name, test_params.params]
        test_name = to_varname_without_excess_underscores('_'.join(test_name_parts))

        setattr(class_add, test_name, test_gen(test_params, prefix))


def test_gen(test_params, prefix):
    func_name = 'func'
    input_data = ','.join([test_params.name_data, test_params.names_extra_data])
    call_expression = test_params.call_expression
    if test_params.call_expression is None:
        prefix_as_list = [prefix] if prefix else []
        if test_params.names_extra_data != '' and test_params.params != '':
            params = ','.join([test_params.names_extra_data, test_params.params])
        else:
            params = test_params.names_extra_data + test_params.params
        expr_parts = [test_params.name_data] + prefix_as_list + ['{}({})'.format(test_params.name, params)]
        call_expression = '.'.join(expr_parts)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{input_data}', '{call_expression}'), name='{test_params.name}', 
                               total_data_length={test_params.size}, data_num={test_params.data_num},
                               input_data={test_params.input_data})
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
