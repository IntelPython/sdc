import importlib.util
import sys

from enum import Enum
from pathlib import Path

PD_BENCHMARKS_MODULE = 'pd_benchmarks'
build_dir = Path(__file__).absolute().parents[3] / 'build'
pandas_benchmarks_path = build_dir / 'asv_bench' / 'pandas' / 'asv_bench' / 'benchmarks' / '__init__.py'
spec = importlib.util.spec_from_file_location(PD_BENCHMARKS_MODULE, pandas_benchmarks_path)
pd_benchmarks_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_benchmarks_module)
sys.modules[PD_BENCHMARKS_MODULE] = pd_benchmarks_module

from pd_benchmarks.pandas_vb_common import BaseIO, setup

class Tool(Enum):
    pandas = 'pandas'
    hpat = 'hpat'
