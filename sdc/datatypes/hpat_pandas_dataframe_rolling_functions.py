

@sdc_overload_method(DataFrameRollingType, 'mean')
def sdc_pandas_dataframe_rolling_mean(self):

    ty_checker = TypeChecker('Method rolling.mean().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('mean', self)


sdc_pandas_dataframe_rolling_mean.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'mean',
    'example_caption': 'Calculate the rolling mean of the values.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})
