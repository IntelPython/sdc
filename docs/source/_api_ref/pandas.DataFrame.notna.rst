.. _pandas.DataFrame.notna:

:orphan:

pandas.DataFrame.notna
**********************

Detect existing (non-missing) values.

Return a boolean same-sized object indicating if the values are not NA.
Non-missing values get mapped to True. Characters such as empty
strings ``''`` or :attr:`numpy.inf` are not considered NA values
(unless you set ``pandas.options.mode.use_inf_as_na = True``).
NA values, such as None or :attr:`numpy.NaN`, get mapped to False
values.

:return: DataFrame
    Mask of bool values for each element in DataFrame that
    indicates whether an element is not an NA value.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

