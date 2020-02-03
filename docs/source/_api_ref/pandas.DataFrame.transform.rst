.. _pandas.DataFrame.transform:

:orphan:

pandas.DataFrame.transform
**************************

Call ``func`` on self producing a DataFrame with transformed values
and that has the same axis length as self.

.. versionadded:: 0.20.0

:param func:
    function, str, list or dict
        Function to use for transforming the data. If a function, must either
        work when passed a DataFrame or when passed to DataFrame.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
        - dict of axis labels -> functions, function names or list of such.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index': apply function to each column.
        If 1 or 'columns': apply function to each row.
        \*args
        Positional arguments to pass to `func`.
        \*\*kwargs
        Keyword arguments to pass to `func`.

:return: DataFrame
    A DataFrame that must have the same length as self.

:raises:
    ValueError : If the returned DataFrame has a different length than self.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

