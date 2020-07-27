import ast
import inspect
import test_generics
import textwrap
from pathlib import Path


def get_variable_annotations(func):
    """Get local variable annotations from the function."""
    module_path = inspect.getfile(func)
    module_object = inspect.getmodule(func)
    module_name = (Path(f'{module_path}').stem)
    func_code = inspect.getsource(func)
    func_code_with_dedent = textwrap.dedent(func_code)
    func_tree = ast.parse(func_code_with_dedent)
    analyzer = Analyzer(module_name, module_object)
    analyzer.visit(func_tree)
    return analyzer.locals_parameter


class Analyzer(ast.NodeVisitor):
    def __init__(self, module_name, module_object):
        self.locals_parameter = {}
        self.module_name = module_name
        self.module_object = module_object
        self.global_parameter = {self.module_name: self.module_object}

    def visit_AnnAssign(self, node):
        target, annotation = node.target, node.annotation
        if isinstance(annotation, ast.Subscript):  # containers and generics
            try:
                container_name = annotation.value.id
                module_import_name = ''
            except AttributeError:  # typing.
                module_import_name = annotation.value.value.id + '.'
                container_name = annotation.value.attr
            if isinstance(annotation.slice.value, ast.Tuple):
                types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                exec_variables = f'{target.id} = [{self.module_name}.{module_import_name}{container_name}[{types_as_str}]]'
            else:
                exec_variables = f'{target.id} = [{self.module_name}.{module_import_name}{container_name}[{annotation.slice.value.id}]]'
            try:
                exec(exec_variables, self.global_parameter, self.locals_parameter)
            except NameError:  # if container_name[TypeVar]
                if isinstance(annotation.slice.value, ast.Tuple):
                    types_as_str = ','.join(self.module_name + '.' + module_import_name +
                                            t for t in types_as_str.split(','))
                    exec_variables = f'{target.id} = [{self.module_name}.{module_import_name}{container_name}[{types_as_str}]]'
                else:
                    exec_variables = f'{target.id} = [{self.module_name}.{module_import_name}{container_name}\
                                                      [{self.module_name}.{annotation.slice.value.id}]]'
                exec(exec_variables, self.global_parameter, self.locals_parameter)
        else:  # not containers
            try:
                exec(f'{target.id} = [{annotation.id}]', self.global_parameter, self.locals_parameter)
            except NameError:  # if TypeVar
                exec_variables = f'{target.id} = [{self.module_name}.{annotation.id}]'
                exec(exec_variables, self.global_parameter, self.locals_parameter)


if __name__ == '__main__':
    print(get_variable_annotations(test_generics.qwe))
