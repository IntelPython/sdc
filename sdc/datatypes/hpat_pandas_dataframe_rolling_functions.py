

@sdc_overload_method(DataFrameRollingType, 'var')
def sdc_pandas_dataframe_rolling_var(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.var().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return apply_df_rolling_method('var', self, kws={'ddof': '1'})


sdc_pandas_dataframe_rolling_var.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'var',
    'example_caption': 'Calculate unbiased rolling variance.',
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
