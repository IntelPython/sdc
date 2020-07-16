import ast
import inspect
from typing import Optional, Any, Union, List, Tuple, Dict


def main():
    with open('test_generics.py') as f:
        tree = ast.parse(f.read())
    analyzer = Analyzer()
    analyzer.visit(tree)
    print(analyzer.types)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.types = {}
        self.flag = False

    def visit_FunctionDef(self, obj):
        self.flag = True
        for i in obj.body:
            self.visit(i)

    def visit_AnnAssign(self, node):
        if self.flag:
            # need to check for child functions
            if isinstance(node.annotation, ast.Subscript):  # containers and generics
                container_name = node.annotation.value.id
                if isinstance(node.annotation.slice.value, ast.Tuple):
                    type_list = []
                    for t in node.annotation.slice.value.elts:
                        type_list.append(t.id)
                    list_to_str = ','.join(type_list)
                    self.types[node.target.id] = [eval(f'{container_name}[{list_to_str}]')]
                else:
                    self.types[node.target.id] = [eval(f'{container_name}[{node.annotation.slice.value.id}]')]
            else:
                try:  # not containers
                    self.types[node.target.id] = [eval(node.annotation.id)]
                except TypeError:
                    self.types[node.target.id] = [eval(node.annotation.id)]  # if Any type


if __name__ == '__main__':
    main()
