import ast
import inspect
import test_generics


def get_variable_annotations(func):
    path = inspect.getsourcefile(func)
    with open(path) as f:
        tree = ast.parse(f.read())
    analyzer = Analyzer()
    analyzer.visit(tree)
    print(analyzer.locals_parameter)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.types = {}
        self.locals_parameter = {}
        self.function_flag = False

    def visit_ImportFrom(self, node):
        for alias in node.names:
            exec(f'from {node.module} import {alias.name}', None, self.locals_parameter)

    def visit_Import(self, node):
        exec(f'import {node.names[0].name}', None, self.locals_parameter)

    def visit_FunctionDef(self, node):
        self.function_flag = True
        for i in node.body:
            self.visit(i)

    def visit_AnnAssign(self, node):
        if self.function_flag:
            target, annotation = node.target, node.annotation
            # need to check for child functions
            if isinstance(annotation, ast.Subscript):  # containers and generics
                container_name = annotation.value.id
                if isinstance(annotation.slice.value, ast.Tuple):
                    type_list_in_str = ','.join([elt.id for elt in annotation.slice.value.elts])
                    exec(f'{target.id} = {container_name}[{type_list_in_str}]', None, self.locals_parameter)
                else:
                    exec(
                        f'{target.id} = {container_name}[{annotation.slice.value.id}]', None, self.locals_parameter)
            else:
                exec(f'{target.id} = {annotation.id}', None, self.locals_parameter)


if __name__ == '__main__':
    get_variable_annotations(test_generics.qwe)
