# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""
HTML report generator based on ASV output as JSON.

Example usage:
python asvgen.py --asv-results .asv/results --template template/asvgen.html
"""
import argparse
import json
import itertools
from pathlib import Path

import jinja2


class ASVGen:
    machine_json = 'machine.json'

    def __init__(self, results_path, template_path):
        """
        :param results_path: path to ASV results
        :param template_path: path to HTML template
        """
        self.results_path = results_path
        self.template_path = template_path

    @property
    def result_subdirs(self):
        """Result sub-directories"""
        return (p for p in self.results_path.iterdir() if p.is_dir())

    def render_template(self, context):
        """
        Render specified HTML template via specified context

        :param name: name of the template file
        :param context: context to render template
        :param templates_path: path to directory with templates
        :return: rendered template
        """
        template_loader = jinja2.FileSystemLoader(searchpath=self.template_path.parent.as_posix())
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template(self.template_path.name)

        return template.render(context)

    def generate(self):
        """Generate HTML reports based on ASV results"""
        for subdir in self.result_subdirs:
            machine_info = {}
            machine_json_path = subdir / self.machine_json
            if machine_json_path.exists():
                with machine_json_path.open(encoding='utf-8') as fd:
                    machine_info = json.load(fd)
            for res_path in subdir.glob('*.json'):
                if res_path.name == self.machine_json:
                    # Skip machine info file
                    continue

                with res_path.open(encoding='utf-8') as fd:
                    results = json.load(fd)['results']
                data = {}
                for benchmark, result in results.items():
                    # combine benchmarks parameters to match parameters combinations and results, e.g.:
                    # result['params'] = [[0, 1], ['interpreted', 'compiled']]
                    # params = [(0, 'interpreted'), (0, 'compiled'), (1, 'interpreted'), (1, 'compiled')]
                    # result['results'] = [1.87, 1.31, 1.85, 1.28]
                    # def time_smth(0, 'interpreted'): ... => 1.87
                    params = itertools.product(*result.get('params', []))
                    for params, res, stats in zip(params, result['result'], result['stats']):
                        bench_args = ', '.join([str(p) for p in params])
                        data[f'{benchmark}({bench_args})'] = {'result': res, 'stats': stats}
                context = {
                    'extra_info': machine_info,
                    'data': data
                }
                rendered_template = self.render_template(context)
                output_html = res_path.parent / f'{res_path.stem}.html'
                output_html.write_text(rendered_template, encoding='utf-8')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--asv-results', required='.asv/results', type=Path, help='Path to ASV results directory')
    parser.add_argument('--template', default='templates/asvgen.html', type=Path, help='Path to the html template')

    return parser.parse_args()


def main():
    args = parse_args()
    ASVGen(args.asv_results, args.template).generate()


if __name__ == '__main__':
    main()
