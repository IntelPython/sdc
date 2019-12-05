# TODO: make it possible to run tests in group. Now it should be run one by one with:
# pytest ./test_import-hooks.py::<test name>

import sys


def unload_sdc():
    """Return to the state when sdc is not imported."""
    for module in list(sys.modules.keys()):
        if module.startswith('sdc'):
            sys.modules.pop(module)


def test_pandas_is_not_enought():
    import pandas
    assert 'sdc' not in sys.modules


def test_numba_is_not_enought():
    import numba
    assert 'sdc' not in sys.modules


def test_numba_and_pandas_is_not_enought():
    import numba
    import pandas
    assert 'sdc' not in sys.modules


def test_numba_and_pandas_and_compilation_is_enought():
    import numba
    import pandas

    @numba.jit('int64()')
    def f():
        return 42

    assert 'sdc' in sys.modules


def test_pandas_after_compilation_is_ok():
    unload_sdc()

    import numba

    # this variable blocks extension initialization more than once
    assert not numba.entrypoints._already_initialized

    @numba.jit('int64()')
    def f():
        return 42
    assert numba.entrypoints._already_initialized
    assert 'sdc' not in sys.modules

    # If extensions initialized sdc module is loaded only if:
    # 1. pandas is imported
    import pandas
    assert 'sdc' in sys.modules


# Old test.

# def test_sdc_loaded():
#     unload_sdc()

#     assert 'sdc' not in sys.modules

#     import numba
#     assert 'sdc' not in sys.modules

#     @numba.jit
#     def f():
#         return 42
#     assert 'sdc' not in sys.modules

#     f.compile('int64()')
#     assert 'sdc' not in sys.modules

#     # Reset compilation for compiling again
#     f.overloads.clear()
#     # Reset _already_initialized for initializing extensions again
#     numba.entrypoints._already_initialized = False

#     # sdc module is loaded only if:
#     # 1. pandas is imported
#     # 2. compilation (or first run, or signature in numba.jit())
#     import pandas
#     assert 'sdc' not in sys.modules

#     f.compile('int64()')
#     assert 'sdc' in sys.modules
