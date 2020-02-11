import time
import numpy as np
from typing import NamedTuple
from sdc.io.csv_ext import to_varname
from sdc.tests.test_utils import *


class CallExpression(NamedTuple):
    """
    code: function or method call as a string
    type_: type of function performed (Python, Numba, SDC)
    jitted: option indicating whether to jit call
    """
    code: str
    type_: str
    jitted: bool


class TestCase(NamedTuple):
    """
    name: name of the API item, e.g. method, operator
    size: size of the generated data for tests
    params: method parameters in format 'par1, par2, ...'
    call_expr: call expression as a string, e.g. '(A+B).sum()' where A, B are Series or DF
    usecase_params: input parameters for usecase in format 'par1, par2, ...', e.g. 'data, other'
    data_num: total number of generated data, e.g. 2 (data, other)
    input_data: input data for generating test data
    skip: flag for skipping a test
    """
    name: str
    size: list
    params: list = ['']
    call_expr: str = None
    usecase_params: str = None
    data_num: int = 1
    input_data: list = [test_global_input_data_float64]
    skip: bool = False


def to_varname_without_excess_underscores(string):
    """Removing excess underscores from the string."""
    return '_'.join(i for i in to_varname(string).split('_') if i)


def generate_test_cases(cases, class_add, typ, prefix=''):
    for test_case in cases:
        for params in test_case.params:
            print(params)
            for input_data in test_case.input_data:
                typ_input_data = qualifier_type(input_data)
                if typ_input_data == 'str':
                    test_name_parts = ['test', typ, prefix, test_case.name,
                                       gen_params_wo_data(test_case.data_num, params)]
                else:
                    test_name_parts = ['test', typ, prefix, test_case.name,
                                       gen_params_wo_data(test_case.data_num, params), typ_input_data]
                print(test_name_parts)
                test_name = to_varname_without_excess_underscores('_'.join(test_name_parts))
                print(test_name)
                setattr(class_add, test_name, gen_test(test_case, prefix, params, typ_input_data, input_data))


def gen_params_wo_data(data_num, params):
    """Generate API item parameters without parameters with data, e.g. without parameter other"""
    extra_data_num = data_num - 1
    print(params)
    method_params = params.split(', ')[extra_data_num:]
    print(', '.join(method_params))
    return ', '.join(method_params)


def gen_usecase_params(test_case, params):
    """Generate usecase parameters based on method parameters and number of extra generated data"""
    extra_data_num = test_case.data_num - 1
    extra_usecase_params = params.split(', ')[:extra_data_num]
    usecase_params_parts = ['data'] + extra_usecase_params

    return ', '.join(usecase_params_parts)


def gen_call_expr(test_case, prefix, params):
    """Generate call expression based on method name and parameters and method prefix, e.g. str"""
    prefix_as_list = [prefix] if prefix else []
    call_expr_parts = ['data'] + prefix_as_list + ['{}({})'.format(test_case.name, params)]

    return '.'.join(call_expr_parts)


def qualifier_type(input_data):
    if isinstance(input_data[0], list):
        return qualifier_type(input_data[0])
    else:
        typ = str(type(input_data[0]))
        res = typ[8:len(typ)-2]
        return res


def gen_test(test_case, prefix, params, typ_input_data, input_data):
    func_name = 'func'

    usecase = gen_usecase(test_case, prefix, params)

    skip = '@skip_numba_jit\n' if test_case.skip else ''

    test_name = test_case.name
    if params:
        test_name = f'{test_name}({params})'

    func_text = f"""
{skip}def {func_name}(self):
  self._test_case(usecase, name='{test_name + '_' + typ_input_data}', total_data_length={test_case.size},
                  data_num={test_case.data_num}, input_data=input_data, typ='{typ_input_data}')
"""

    loc_vars = {}
    global_vars = {'usecase': usecase,
                   'skip_numba_jit': skip_numba_jit,
                   'input_data': input_data}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def create_func(usecase_params, call_expr):
    func_name = 'func'

    func_text = f"""
def {func_name}({usecase_params}):
  start_time = time.time()
  res = {call_expr}
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    func = loc_vars[func_name]

    return func


def gen_usecase(test_case, prefix, params):
    usecase_params = test_case.usecase_params
    call_expr = test_case.call_expr
    if call_expr is None:
        if usecase_params is None:
            usecase_params = gen_usecase_params(test_case, params)
        call_expr = gen_call_expr(test_case, prefix, params)

    if isinstance(call_expr, list):
        results = []
        for ce in call_expr:
            results.append({
                'func': create_func(usecase_params, ce.code),
                'type_': ce.type_,
                'jitted': ce.jitted
            })

        return results

    func = create_func(usecase_params, call_expr)
    return func
