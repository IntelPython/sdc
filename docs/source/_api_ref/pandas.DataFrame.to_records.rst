.. _pandas.DataFrame.to_records:

:orphan:

pandas.DataFrame.to_records
***************************

Convert DataFrame to a NumPy record array.

Index will be included as the first field of the record array if
requested.

:param index:
    bool, default True
        Include index in resulting record array, stored in 'index'
        field or using the index label, if set.

:param convert_datetime64:
    bool, default None
        .. deprecated:: 0.23.0

        Whether to convert the index to datetime.datetime if it is a
        DatetimeIndex.

:param column_dtypes:
    str, type, dict, default None
        .. versionadded:: 0.24.0

        If a string or type, the data type to store all columns. If
        a dictionary, a mapping of column names and indices (zero-indexed)
        to specific data types.

:param index_dtypes:
    str, type, dict, default None
        .. versionadded:: 0.24.0

        If a string or type, the data type to store all index levels. If
        a dictionary, a mapping of index level names and indices
        (zero-indexed) to specific data types.

        This mapping is applied only if `index=True`.

:return: numpy.recarray
    NumPy ndarray with the DataFrame labels as fields and each row
    of the DataFrame as entries.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

