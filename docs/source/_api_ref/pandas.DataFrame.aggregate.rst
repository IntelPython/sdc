.. _pandas.DataFrame.aggregate:

:orphan:

pandas.DataFrame.aggregate
**************************

Aggregate using one or more operations over the specified axis.

.. versionadded:: 0.20.0

:param func:
    function, str, list or dict
        Function to use for aggregating the data. If a function, must either
        work when passed a DataFrame or when passed to DataFrame.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
        - dict of axis labels -> functions, function names or list of such.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index': apply function to each column.
        If 1 or 'columns': apply function to each row.
        \*args
        Positional arguments to pass to `func`.
        \*\*kwargs
        Keyword arguments to pass to `func`.

:return: scalar, Series or DataFrame

    The return can be:

    - scalar : when Series.agg is called with single function
    - Series : when DataFrame.agg is called with a single function
    - DataFrame : when DataFrame.agg is called with several functions

    Return scalar, Series or DataFrame.

    The aggregation operations are always performed over an axis, either the
    index (default) or the column axis. This behavior is different from
    `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
    `var`), where the default is to compute the aggregation of the flattened
    array, e.g., ``numpy.mean(arr_2d)`` as opposed to
    ``numpy.mean(arr_2d, axis=0)``.

    `agg` is an alias for `aggregate`. Use the alias.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

