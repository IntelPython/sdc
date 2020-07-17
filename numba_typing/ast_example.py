import ast
import inspect


def main():
    with open('test_generics.py') as f:
        tree = ast.parse(f.read())
    analyzer = Analyzer()
    analyzer.visit(tree)
    print(analyzer.types)
    print(analyzer.import_from)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.types = {}
        self.import_from = []
        self.module = None
        self.function_flag = False

    def visit_ImportFrom(self, node):  # need a check "import module"
        exec(f'import {node.module}', globals())
        self.module = node.module
        for alias in node.names:
            self.import_from.append(eval(f'{node.module}.{alias.name}'))

    def visit_FunctionDef(self, obj):
        self.function_flag = True
        for i in obj.body:
            self.visit(i)

    def visit_AnnAssign(self, node):
        if self.function_flag:
            # need to check for child functions
            if isinstance(node.annotation, ast.Subscript):  # containers and generics
                container_name = node.annotation.value.id
                if isinstance(node.annotation.slice.value, ast.Tuple):
                    type_list = []
                    for t in node.annotation.slice.value.elts:
                        type_list.append(t.id)
                    list_to_str = ','.join(type_list)
                    self.types[node.target.id] = [eval(f'{self.module}.{container_name}[{list_to_str}]')]
                else:
                    self.types[node.target.id] = [
                        eval(f'{self.module}.{container_name}[{node.annotation.slice.value.id}]')]
            else:
                try:  # not containers
                    self.types[node.target.id] = [eval(f'{node.annotation.id}')]
                except NameError:
                    self.types[node.target.id] = [eval(f'{self.module}.{node.annotation.id}')]  # if Any type


if __name__ == '__main__':
    main()
