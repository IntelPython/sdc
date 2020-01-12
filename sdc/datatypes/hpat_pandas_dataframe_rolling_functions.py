

@sdc_overload_method(DataFrameRollingType, 'std')
def sdc_pandas_dataframe_rolling_std(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.std().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return apply_df_rolling_method('std', self, kws={'ddof': '1'})


sdc_pandas_dataframe_rolling_std.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'std',
    'example_caption': 'Calculate rolling standard deviation.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})
