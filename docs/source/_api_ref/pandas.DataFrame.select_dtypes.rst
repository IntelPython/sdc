.. _pandas.DataFrame.select_dtypes:

:orphan:

pandas.DataFrame.select_dtypes
******************************

Return a subset of the DataFrame's columns based on the column dtypes.

:param include, exclude:
    scalar or list-like
        A selection of dtypes or strings to be included/excluded. At least
        one of these parameters must be supplied.

:return: DataFrame
    The subset of the frame including the dtypes in ``include`` and
    excluding the dtypes in ``exclude``.

:raises:
    ValueError
        - If both of ``include`` and ``exclude`` are empty
        - If ``include`` and ``exclude`` have overlapping elements
        - If any kind of string dtype is passed in.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

