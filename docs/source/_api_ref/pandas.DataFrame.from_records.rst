.. _pandas.DataFrame.from_records:

:orphan:

pandas.DataFrame.from_records
*****************************

Convert structured or record ndarray to DataFrame.

:param data:
    ndarray (structured dtype), list of tuples, dict, or DataFrame

:param index:
    string, list of fields, array-like
        Field of array to use as the index, alternately a specific set of
        input labels to use

:param exclude:
    sequence, default None
        Columns or fields to exclude

:param columns:
    sequence, default None
        Column names to use. If the passed data do not have names
        associated with them, this argument provides names for the
        columns. Otherwise this argument indicates the order of the columns
        in the result (any names not found in the data will become all-NA
        columns)

:param coerce_float:
    boolean, default False
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

:param nrows:
    int, default None
        Number of rows to read if data is an iterator

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

