

@sdc_overload_method(DataFrameRollingType, 'max')
def sdc_pandas_dataframe_rolling_max(self):

    ty_checker = TypeChecker('Method rolling.max().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('max', self)


sdc_pandas_dataframe_rolling_max.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'max',
    'example_caption': 'Calculate the rolling maximum.',
    'limitations_block': '',
    'extra_params': ''
})
