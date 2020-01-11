

@sdc_overload_method(DataFrameRollingType, 'kurt')
def sdc_pandas_dataframe_rolling_kurt(self):

    ty_checker = TypeChecker('Method rolling.kurt().')
    ty_checker.check(self, DataFrameRollingType)

    return apply_df_rolling_method('kurt', self)


sdc_pandas_dataframe_rolling_kurt.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'kurt',
    'example_caption': 'Calculate unbiased rolling kurtosis.',
    'limitations_block': '',
    'extra_params': ''
})
