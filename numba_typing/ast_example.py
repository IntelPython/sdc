import ast
import inspect
import test_generics
import textwrap
# import test
from pathlib import Path


def get_variable_annotations(func):
    module_path = inspect.getfile(func)
    module_name = (Path(f'{module_path}').stem)
    exec(f'import {module_name}', globals())
    func_code = inspect.getsource(func)
    func_code_with_dedent = textwrap.dedent(func_code)
    func_tree = ast.parse(func_code_with_dedent)
    analyzer = Analyzer(module_name)
    analyzer.visit(func_tree)
    return analyzer.locals_parameter


class Analyzer(ast.NodeVisitor):
    def __init__(self, module_name):
        self.locals_parameter = {}
        self.module_name = module_name
        self.global_parameter = {f'{self.module_name}': globals()[self.module_name]}

    def visit_AnnAssign(self, node):
        target, annotation = node.target, node.annotation
        exec_imports = []
        if isinstance(annotation, ast.Subscript):  # containers and generics
            try:
                container_name = annotation.value.id
                if isinstance(annotation.slice.value, ast.Tuple):
                    types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                    exec_variables = f'{target.id} = {self.module_name}.{container_name}[{types_as_str}]'
                else:
                    exec_variables = f'{target.id} = {self.module_name}.{container_name}[{annotation.slice.value.id}]'
            except AttributeError:  # typing.
                module_import_name = annotation.value.value.id
                container_name = annotation.value.attr
                if isinstance(annotation.slice.value, ast.Tuple):
                    types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                    exec_variables = f'{target.id} = {module_import_name}.{container_name}[{types_as_str}]'
                else:
                    exec_variables = f'{target.id} = {self.module_name}.{module_import_name}.{container_name}[{annotation.slice.value.id}]'
            exec(exec_variables, self.global_parameter, self.locals_parameter)
        else:  # not containers
            try:
                exec(f'{target.id} = {annotation.id}', self.global_parameter, self.locals_parameter)
            except NameError:  # if Any type
                exec_variables = f'{target.id} = {self.module_name}.{annotation.id}'
                exec(exec_variables, self.global_parameter, self.locals_parameter)


if __name__ == '__main__':
    print(get_variable_annotations(test_generics.qwe))
