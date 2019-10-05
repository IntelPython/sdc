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
Loads a benchmark from pickle-file, runs it and dumps results to json-file.
The script is ran under runner.py to run a benchmarking test case in separate process.

Example usages:
python benchmark.py --bench-pickle time_quantile.pickle --res-json time_quantile.json
"""
import argparse
import inspect
import json
import pickle
import timeit

from enum import Enum
from pathlib import Path


class BenchmarksType(Enum):
    """Benchmark types"""
    TIME = 'time'


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
    def __init__(self, name, func, param, source, repeat=10, number=1):
        super().__init__(name, func, param, source)
        self.repeat = repeat
        self.number = number

    @classmethod
    def from_pickle(cls, file_path):
        file_path = Path(file_path)
        with file_path.open('rb') as fd:
            return pickle.load(fd)

    def to_pickle(self, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as fd:
            return pickle.dump(self, fd)

    def run(self):
        """Run benchmark timing"""

        def func():
            return self.func(self.instance, *self.param)

        timer = timeit.Timer(
            stmt=func,
            setup=self.redo_setup,
            timer=timeit.default_timer)

        # Warming up
        timeit.timeit(number=1)

        samples = timer.repeat(repeat=self.repeat, number=self.number)

        return [sample / self.number for sample in samples]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench-pickle', required=True, type=Path, help='Path to pickle with benchmark object')
    parser.add_argument('--res-json', required=True, type=Path, help='Path to json with resulting samples')

    return parser.parse_args()


def main():
    args = parse_args()

    benchmark = TimeBenchmark.from_pickle(args.bench_pickle)
    args.res_json.parent.mkdir(parents=True, exist_ok=True)
    with args.res_json.open('w', encoding='utf-8') as fd:
        json.dump(benchmark.run(), fd)


if __name__ == '__main__':
    main()
