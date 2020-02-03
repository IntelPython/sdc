.. _pandas.DataFrame.query:

:orphan:

pandas.DataFrame.query
**********************

Query the columns of a DataFrame with a boolean expression.

:param expr:
    str
        The query string to evaluate.  You can refer to variables
        in the environment by prefixing them with an '@' character like
        ``@a + b``.

        .. versionadded:: 0.25.0

        You can refer to column names that contain spaces by surrounding
        them in backticks.

        For example, if one of your columns is called ``a a`` and you want
        to sum it with ``b``, your query should be ```a a` + b``.

:param inplace:
    bool
        Whether the query should modify the data in place or return
        a modified copy.
        \*\*kwargs
        See the documentation for :func:`eval` for complete details
        on the keyword arguments accepted by :meth:`DataFrame.query`.

        .. versionadded:: 0.18.0

:return: DataFrame
    DataFrame resulting from the provided query expression.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

