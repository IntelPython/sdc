

@sdc_overload_method(DataFrameRollingType, 'apply')
def sdc_pandas_dataframe_rolling_apply(self, func, raw=None):

    ty_checker = TypeChecker('Method rolling.apply().')
    ty_checker.check(self, DataFrameRollingType)

    raw_accepted = (Omitted, NoneType, Boolean)
    if not isinstance(raw, raw_accepted) and raw is not None:
        ty_checker.raise_exc(raw, 'bool', 'raw')

    return sdc_pandas_dataframe_rolling_apply('apply', self, args=['func'],
                                              kws={'raw': 'None'})


sdc_pandas_dataframe_rolling_apply.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'apply',
    'example_caption': 'Calculate the rolling apply.',
    'limitations_block':
    """
    Limitations
    -----------
    Supported ``raw`` only can be `None` or `True`. Parameters ``args``, ``kwargs`` unsupported.
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    func:
        A single value producer
    raw: :obj:`bool`
        False : passes each row or column as a Series to the function.
        True or None : the passed function will receive ndarray objects instead.
    """
})
