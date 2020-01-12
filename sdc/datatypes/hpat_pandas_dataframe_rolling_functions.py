

@sdc_overload_method(DataFrameRollingType, 'quantile')
def sdc_pandas_dataframe_rolling_quantile(self, quantile, interpolation='linear'):

    ty_checker = TypeChecker('Method rolling.quantile().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(quantile, Number):
        ty_checker.raise_exc(quantile, 'float', 'quantile')

    str_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(interpolation, str_types) and interpolation != 'linear':
        ty_checker.raise_exc(interpolation, 'str', 'interpolation')

    return apply_df_rolling_method('quantile', self, args=['quantile'],
                                   kws={'interpolation': 'linear'})


sdc_pandas_dataframe_rolling_quantile.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'quantile',
    'example_caption': 'Calculate the rolling quantile.',
    'limitations_block':
    """
    Limitations
    -----------
    Supported ``interpolation`` only can be `'linear'`.
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    quantile: :obj:`float`
        Quantile to compute. 0 <= quantile <= 1.
    interpolation: :obj:`str`
        This optional parameter specifies the interpolation method to use.
    """
})
