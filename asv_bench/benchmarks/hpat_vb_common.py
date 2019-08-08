import importlib.util
import sys

from enum import Enum
from pathlib import Path

PD_BENCHMARKS_MODULE = 'pd_benchmarks'
pandas_benchmarks_path = Path(__file__).absolute().parents[1] / 'pandas' / 'asv_bench' / 'benchmarks' / '__init__.py'
spec = importlib.util.spec_from_file_location(PD_BENCHMARKS_MODULE, pandas_benchmarks_path)
pd_benchmarks_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_benchmarks_module)
sys.modules[PD_BENCHMARKS_MODULE] = pd_benchmarks_module

from pd_benchmarks.pandas_vb_common import BaseIO, setup

class Tool(Enum):
    pandas = 'pandas'
    hpat = 'hpat'
