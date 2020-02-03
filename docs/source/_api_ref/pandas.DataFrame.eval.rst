.. _pandas.DataFrame.eval:

:orphan:

pandas.DataFrame.eval
*********************

Evaluate a string describing operations on DataFrame columns.

Operates on columns only, not specific rows or elements.  This allows
`eval` to run arbitrary code, which can make you vulnerable to code
injection if you pass user input to this function.

:param expr:
    str
        The expression string to evaluate.

:param inplace:
    bool, default False
        If the expression contains an assignment, whether to perform the
        operation inplace and mutate the existing DataFrame. Otherwise,
        a new DataFrame is returned.

        .. versionadded:: 0.18.0.

:param kwargs:
    dict
        See the documentation for :func:`eval` for complete details
        on the keyword arguments accepted by
        :meth:`~pandas.DataFrame.query`.

:return: ndarray, scalar, or pandas object
    The result of the evaluation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

