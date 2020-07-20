import ast
import inspect
import test_generics


def get_variable_annotations(func):
    module_name = inspect.getmodule(func).__name__
    func_code = inspect.getsource(func)
    func_tree = ast.parse(func_code)
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
        # need to check for child functions
        if isinstance(annotation, ast.Subscript):  # containers and generics
            try:
                container_name = annotation.value.id
                exec_imports = f'from {self.module_name} import {container_name}'
                if isinstance(annotation.slice.value, ast.Tuple):
                    types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                    exec_variables = f'{target.id} = {container_name}[{types_as_str}]'
                else:
                    exec_variables = f'{target.id} = {container_name}[{annotation.slice.value.id}]'
            except AttributeError:
                module_name = annotation.value.value.id
                container_name = annotation.value.attr
                exec_imports = f'import {module_name}'
                if isinstance(annotation.slice.value, ast.Tuple):
                    types_as_str = ','.join(elt.id for elt in annotation.slice.value.elts)
                    exec_variables = f'{target.id} = {module_name}.{container_name}[{types_as_str}]'
                else:
                    exec_variables = f'{target.id} = {module_name}.{container_name}[{annotation.slice.value.id}]'
        else:  # not containers
            try:
                exec(f'{target.id} = {annotation.id}', self.global_parameter, self.locals_parameter)
            except NameError:  # if Any type
                exec_imports = f'from {self.module_name} import {annotation.id}'
                exec_variables = f'{target.id} = {annotation.id}'
        if exec_imports:
            exec(exec_imports, self.global_parameter)
            exec(exec_variables, self.global_parameter, self.locals_parameter)


if __name__ == '__main__':
    print(get_variable_annotations(test_generics.qwe))
