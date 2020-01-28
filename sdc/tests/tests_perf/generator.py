import time
import numpy as np
from sdc.io.csv_ext import to_varname
from typing import NamedTuple


class TestCase(NamedTuple):
    name: str
    size: list
    params: str = ''
    call_expression: str = None
    usecase_params: str = None
    extra_data_num: int = 0
    input_data: list = None


def to_varname_without_excess_underscores(string):
    """Removing excess underscores from the string."""
    return '_'.join(i for i in to_varname(string).split('_') if i)


def generate_test_cases(cases, class_add, typ, prefix=''):
    for test_case in cases:
        params_parts = test_case.params.split(', ', test_case.extra_data_num)
        if len(params_parts) > test_case.extra_data_num:
            params = params_parts[test_case.extra_data_num]
        else:
            params = ''
        test_name_parts = ['test', typ, prefix, test_case.name, params]
        test_name = to_varname_without_excess_underscores('_'.join(test_name_parts))

        setattr(class_add, test_name, test_gen(test_case, prefix))


def test_gen(test_case, prefix):
    func_name = 'func'
    usecase_params = test_case.usecase_params
    call_expression = test_case.call_expression
    if test_case.call_expression is None:
        if usecase_params is None:
            other = test_case.params.split(', ', test_case.extra_data_num).pop()
            usecase_params_parts = ['data'] + [other]
            usecase_params = ', '.join(usecase_params_parts)
        prefix_as_list = [prefix] if prefix else []
        call_expression_parts = ['data'] + prefix_as_list + ['{}({})'.format(test_case.name, test_case.params)]
        call_expression = '.'.join(call_expression_parts)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{usecase_params}', '{call_expression}'), name='{test_case.name}', 
                               total_data_length={test_case.size}, extra_data_num={test_case.extra_data_num},
                               input_data={test_case.input_data})
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
