.. _pandas.core.window.Expanding.aggregate:

:orphan:

pandas.core.window.Expanding.aggregate
**************************************

Aggregate using one or more operations over the specified axis.

:param func:
    function, str, list or dict
        Function to use for aggregating the data. If a function, must either
        work when passed a Series/Dataframe or when passed to Series/Dataframe.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
        - dict of axis labels -> functions, function names or list of such.

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

