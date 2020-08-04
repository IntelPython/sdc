import ast
import inspect
import textwrap
from pathlib import Path


def get_variable_annotations(func):
    """Get local variable annotations from the function."""
    module_path = inspect.getfile(func)
    func_global_variables = func.__globals__
    func_global_variables.update(inspect.getclosurevars(func).nonlocals)
    module_name = (Path(f'{module_path}').stem)
    func_code = inspect.getsource(func)
    func_code_with_dedent = textwrap.dedent(func_code)
    func_tree = ast.parse(func_code_with_dedent)
    analyzer = Analyzer(module_name, func_global_variables)
    analyzer.visit(func_tree)
    return analyzer.locals_parameter


class Analyzer(ast.NodeVisitor):
    def __init__(self, module_name, func_global_variables):
        self.locals_parameter = {}
        self.module_name = module_name
        self.global_parameter = func_global_variables

    def visit_AnnAssign(self, node):
        target, annotation = node.target, node.annotation
        if isinstance(annotation, ast.Subscript):  # containers and generics
            try:
                container_name = annotation.value.id
                module_container_name = ''
            except AttributeError:  # typing.
                module_container_name = annotation.value.value.id + '.'
                container_name = annotation.value.attr
            if isinstance(annotation.slice.value, ast.Tuple):
                types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                exec_variables = f'{target.id} = [{module_container_name}{container_name}[{types_as_str}]]'
            else:
                exec_variables = f'{target.id} = [{module_container_name}{container_name}[{annotation.slice.value.id}]]'
            exec(exec_variables, self.global_parameter, self.locals_parameter)
        else:  # not containers
            exec(f'{target.id} = [{annotation.id}]', self.global_parameter, self.locals_parameter)
