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
Performance tests runner runs performance tests implemented as class with attributes:
params (list of lists): values of test parameters
param_names (list): names of test parameters
def setup(): method which runs before each test case
def time_*(): test case
def teardown(): method which runs after each test case

E.g:
class Methods:
    params = [
        [5000513],
        [1, 3, 5, 9, 17, 33],
        ['interpreted_python', 'compiled_python']
    ]
    param_names = ['size', 'nchars', 'implementation']

    def setup(self, size, nchars, implementation):
        self.series = StringSeriesGenerator(size=size, nchars=nchars).generate()

    @staticmethod
    @hpat.jit
    def _len(series):
        return series.str.len()

    def time_len(self, size, nchars, implementation):
        if implementation == Impl.compiled_python.value:
            return self._len(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.len()

Example usages:
1. Run all:
    python runner.py
1. Run tests/strings.py:
    python runner.py --bench tests.strings
"""
import argparse
import inspect
import itertools
import json
import logging
import pkgutil
import platform
import statistics
import timeit

from collections import OrderedDict
from enum import Enum
from importlib import import_module
from pathlib import Path

class BenchmarksType(Enum):
    """Benchmark types"""
    TIME = 'time'


def setup_logging():
    """Setup logger"""
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger


class Benchmark:
    def __init__(self, name, func, param, sources):
        self.name = name
        self.func = func
        self.param = param
        self.source = sources

        self.setup = inspect.getattr_static(sources, 'setup', None)
        self.teardown = inspect.getattr_static(sources, 'teardown', None)

        self.instance = sources()

    def run(self):
        """Run benchmark with its parameters"""
        self.func(self.instance, *self.param)

    def do_setup(self):
        """Run setup method of benchmark"""
        if self.setup:
            self.setup(self.instance, *self.param)

    def redo_setup(self):
        """Run teardown and setup methods of benchmark"""
        self.do_teardown()
        self.do_setup()

    def do_teardown(self):
        """Run teardown method of benchmark"""
        if self.teardown:
            self.teardown(self.instance, *self.param)


class TimeBenchmark(Benchmark):
    def __init__(self,  name, func, param, source, repeat=10, number=1):
        super().__init__(name, func, param, source)
        self.repeat = repeat
        self.number = number

    def run(self):
        """Run benchmark timing"""
        func = lambda: self.func(self.instance, *self.param)
        timer = timeit.Timer(
            stmt=func,
            setup=self.redo_setup,
            timer=timeit.default_timer)

        # Warming up
        timeit.timeit(number=1)

        return timer.repeat(repeat=self.repeat, number=self.number)


def discover_modules(mnodule_name):
    """
    Recursively import a module and all sub-modules in the module

    :param mnodule_name: module name
    :return: modules from the package
    """
    module = import_module(mnodule_name)

    yield module

    if getattr(module, '__path__', None):
        for _, name, _ in pkgutil.iter_modules(module.__path__, f'{mnodule_name}.'):
            yield from discover_modules(name)


def discover_benchmarks(module_name, type_=BenchmarksType.TIME.value, repeat=10, number=1):
    """
    Discover benchmarks in the module

    :param module_name: benchmarks module
    :param type_: benchmark type
    :return: time benchmarks
    """
    for module in discover_modules(module_name):
        for attr_name, module_attr in module.__dict__.items():
            if attr_name.startswith('_'):
                # skip attributes which start with underscore
                continue

            if inspect.isclass(module_attr):
                for name, class_attr in inspect.getmembers(module_attr):
                    if not name.startswith(f'{type_}_'):
                        continue

                    name_parts = module.__name__.split('.', 1)[1:] + [module_attr.__name__, name]
                    benchmark_name = '.'.join(name_parts)
                    func = inspect.getattr_static(module_attr, name)
                    params = inspect.getattr_static(module_attr, 'params', [[]])
                    for param in itertools.product(*params):
                        yield TimeBenchmark(benchmark_name, func, param, module_attr, repeat=repeat, number=number)


def compute_stats(samples):
    """Statistical analysis of the samples"""
    return   {
        'min': min(samples),
        'max': max(samples),
        'mean': statistics.mean(samples),
        'std': statistics.stdev(samples)
    }

def dump_results(results, file_path):
    """Dump benchmarking results to json-file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as fd:
        json.dump(results, fd)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', default='tests', help='Module with performance tests')
    parser.add_argument('--number', default=1, type=int, help='Repeat count')
    parser.add_argument('--repeat', default=10, type=int, help='Number of executions')
    parser.add_argument('--results-dir', default='../build/tests_perf', type=Path,
                        help='Path to directory with benchmarking results')

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    results = {}
    result = []
    stats = []
    params_combinations = []
    logger.info('Running benchmarks in "%s"...', args.bench)
    for benchmark in discover_benchmarks(args.bench, repeat=args.repeat, number=args.number):
        samples = benchmark.run()
        result.append(statistics.median(samples))
        stats.append(compute_stats(samples))
        results[benchmark.name] = {'result': result, 'stats': stats}
        params_combinations.append(benchmark.param)
        logger.info('%s%s: %ss', benchmark.name, benchmark.param, round(statistics.median(samples), 5))

    params = [list(OrderedDict.fromkeys(y)) for y in zip(*params_combinations)]
    data = {'results': {name: {'params': params, **result} for name, result in results.items()}}
    results_json = args.results_dir / platform.node() / 'results.json'
    dump_results(data, results_json)
    logger.info('Results dumped to "%s"', results_json)


if __name__ == '__main__':
    main()
