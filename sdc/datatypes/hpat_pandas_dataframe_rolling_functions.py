

@sdc_overload_method(DataFrameRollingType, 'sum')
def sdc_pandas_dataframe_rolling_sum(self):

    ty_checker = TypeChecker('Method rolling.sum().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('sum', self)


sdc_pandas_dataframe_rolling_sum.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'sum',
    'example_caption': 'Calculate rolling sum of given Series.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})
