from .svc import SVC
from .naive_bayes import MultinomialNB
from hpat.utils import debug_prints

try:
    from . import d4p
except ImportError:
    if debug_prints():  # pragma: no cover
        print("daal4py import error")
