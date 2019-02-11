import numba
import hpat


def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True

    _locals = options.pop('locals', {})

    # put pivots in locals TODO: generalize numba.jit options
    pivots = options.pop('pivots', {})
    for var, vals in pivots.items():
        _locals[var+":pivot"] = vals

    h5_types = options.pop('h5_types', {})
    for var, vals in h5_types.items():
        _locals[var+":h5_types"] = vals

    distributed = set(options.pop('distributed', set()))
    _locals["##distributed"] = distributed

    threaded = set(options.pop('threaded', set()))
    _locals["##threaded"] = threaded

    options['locals'] = _locals

    #options['parallel'] = True
    options['parallel'] = {'comprehension': True,
                           'setitem':       False,  # FIXME: support parallel setitem
                           'reduction':     True,
                           'numpy':         True,
                           'stencil':       True,
                           'fusion':        True,
                           }

    # this is for previous version of pipeline manipulation (numba hpat_req <0.38)
    # from .compiler import add_hpat_stages
    # return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)
    return numba.jit(signature_or_function, pipeline_class=hpat.compiler.HPATPipeline, **options)
