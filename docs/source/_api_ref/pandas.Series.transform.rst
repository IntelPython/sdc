.. _pandas.Series.transform:

:orphan:

pandas.Series.transform
***********************

Call ``func`` on self producing a Series with transformed values
and that has the same axis length as self.

.. versionadded:: 0.20.0

:param func:
    function, str, list or dict
        Function to use for transforming the data. If a function, must either
        work when passed a Series or when passed to Series.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
        - dict of axis labels -> functions, function names or list of such.

:param axis:
    {0 or 'index'}
        Parameter needed for compatibility with DataFrame.
        \*args
        Positional arguments to pass to `func`.
        \*\*kwargs
        Keyword arguments to pass to `func`.

:return: Series
    A Series that must have the same length as self.

:raises:
    ValueError : If the returned Series has a different length than self.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

