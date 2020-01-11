

@sdc_overload_method(DataFrameRollingType, 'count')
def sdc_pandas_dataframe_rolling_count(self):

    ty_checker = TypeChecker('Method rolling.count().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('count', self)


sdc_pandas_dataframe_rolling_count.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'count',
    'example_caption': 'Count of any non-NaN observations inside the window.',
    'limitations_block': '',
    'extra_params': ''
})
