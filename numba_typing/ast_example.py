import ast
import inspect
import test_generics


def get_variable_annotations(func):
    path = inspect.getsourcefile(func)
    func_code = inspect.getsource(func)
    func_tree = ast.parse(func_code)
    with open(path) as f:
        module_tree = ast.parse(f.read())
    assign_parent(module_tree)
    assign_parent(func_tree)
    analyzer = Analyzer()
    analyzer.visit(module_tree)
    analyzer.visit(func_tree)
    return analyzer.locals_parameter


def assign_parent(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.locals_parameter = {}

    def visit_ImportFrom(self, node):
        for alias in node.names:
            exec(f'from {node.module} import {alias.name}', None, self.locals_parameter)

    def visit_Import(self, node):
        exec(f'import {node.names[0].name}', None, self.locals_parameter)

    def visit_AnnAssign(self, node):
        if not isinstance(node.parent, ast.FunctionDef):
            return
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
    print(get_variable_annotations(test_generics.qwe))
