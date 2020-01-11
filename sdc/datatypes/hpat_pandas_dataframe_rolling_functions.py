

@sdc_overload_method(DataFrameRollingType, 'skew')
def sdc_pandas_dataframe_rolling_skew(self):

    ty_checker = TypeChecker('Method rolling.skew().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('skew', self)


sdc_pandas_dataframe_rolling_skew.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'skew',
    'example_caption': 'Unbiased rolling skewness.',
    'limitations_block': '',
    'extra_params': ''
})
