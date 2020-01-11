

@sdc_overload_method(DataFrameRollingType, 'median')
def sdc_pandas_dataframe_rolling_median(self):

    ty_checker = TypeChecker('Method rolling.median().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('median', self)


sdc_pandas_dataframe_rolling_median.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'median',
    'example_caption': 'Calculate the rolling median.',
    'limitations_block': '',
    'extra_params': ''
})
