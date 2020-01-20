import time
from sdc.io.csv_ext import to_varname


def gen(cases, method, class_add, type, prefix=''):
    for params in cases:
        if len(params) == 4:
            func, param, length, call_expression = params
        elif len(params) == 3:
            func, param, length = params
            call_expression = None
        else:
            func, param = params
            call_expression = None
            length = None
        name = func
        if param:
            name += "_" + to_varname(param)
        func_name = 'test_{}_{}_{}'.format(type, prefix, name)
        func_name = to_varname(func_name).replace('__', '_')

        setattr(class_add, func_name, method(func, param, length, call_expression, prefix))


def test_gen(name, params, data_length, call_expression, prefix):
    func_name = 'func'
    if call_expression is None:
        call_expression = '{}{}({})'.format(prefix, name, params)

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{call_expression}'), '{name}', {data_length})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def usecase_gen(call_expression):
    func_name = 'func'

    func_text = f"""\
def {func_name}(input_data):
  start_time = time.time()
  res = input_data.{call_expression}
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    func = loc_vars[func_name]

    return func


def test_gen_two_par(name, params, data_length, *args, **kwargs):
    func_name = 'func'

    func_text = f"""\
def {func_name}(self):
  self._test_binary_operations(usecase_gen_two_par('{name}', '{params}'),
                                  '{name}', {data_length})
"""

    global_vars = {'usecase_gen_two_par': usecase_gen_two_par}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def usecase_gen_two_par(name, par):
    func_name = 'func'

    func_text = f"""\
def {func_name}(A, B):
  start_time = time.time()
  res = A.{name}(B, {par})
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    func = loc_vars[func_name]

    return func
