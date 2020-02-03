.. _pandas.DataFrame.isna:

:orphan:

pandas.DataFrame.isna
*********************

Detect missing values.

Return a boolean same-sized object indicating if the values are NA.
NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
values.
Everything else gets mapped to False values. Characters such as empty
strings ``''`` or :attr:`numpy.inf` are not considered NA values
(unless you set ``pandas.options.mode.use_inf_as_na = True``).

:return: DataFrame
    Mask of bool values for each element in DataFrame that
    indicates whether an element is not an NA value.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

