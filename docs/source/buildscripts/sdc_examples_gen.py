import os
from copy import copy


def find_examples(path):
    list_files = []
    files = os.listdir(path)
    for name in files:
        if '.py' in name:
            list_files.append('{}/{}'.format(path, name))
        if not '.' in name:
            list_files += find_examples('{}/{}'.format(path, name))

    return list_files


tmpl = '''
    .. literalinclude:: {path}
       :language: python
       :lines: 26-
       :name: ex{name}
        
    .. command-output:: python .{small_path}
       :cwd: ../../examples
'''


def find_examples_docstring(path_example):
    files = os.listdir('../../../sdc/datatypes')
    file_name = ''
    number = 0
    for fil in files:
        if '.py' in fil:
            name = '../../../sdc/datatypes/{}'.format(fil)
            file = open(name, 'r')
            for index, line in enumerate(file):
                if path_example in line:
                    number = index
                    file_name = name
                    break
    if number == 0:
        path = ''.join(copy(path_example).split('../', 1))
        small_path = copy(path_example).split('../../examples')[1]
        name = '_'.join((copy(small_path).split('.')[0]).split('/'))
        example = tmpl.format(
            ** {'path': path,
                'small_path': small_path,
                'name': name})
    else:
        f = open(file_name)
        data = f.read()
        lines = data.split('\n')[number: number + 9]
        example = ''''''
        for line in lines:
            if '../' in line:
                line = ''.join(line.split('../', 1))
            example += line
            example += '\n'

    return example


title = '''.. _examples:
.. include:: ./ext_links.txt

List of examples
================
'''

if __name__ == "__main__":
    all_files = find_examples('../../../examples')
    examples = title
    for file in all_files:
        examples += find_examples_docstring(file)
        examples += '\n'

    file = open('../examples.rst', 'w')
    file.write(examples)
    file.close()
