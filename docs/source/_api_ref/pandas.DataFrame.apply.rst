.. _pandas.DataFrame.apply:

:orphan:

pandas.DataFrame.apply
**********************

Apply a function along an axis of the DataFrame.

Objects passed to the function are Series objects whose index is
either the DataFrame's index (``axis=0``) or the DataFrame's columns
(``axis=1``). By default (``result_type=None``), the final return type
is inferred from the return type of the applied function. Otherwise,
it depends on the `result_type` argument.

:param func:
    function
        Function to apply to each column or row.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        - 0 or 'index': apply function to each column.
        - 1 or 'columns': apply function to each row.

:param broadcast:
    bool, optional
        Only relevant for aggregation functions:

:param \* ``False`` or ``None``:
    returns a Series whose length is the
        length of the index or the number of columns (based on the
        `axis` parameter)

:param \* ``True``:
    results will be broadcast to the original shape
        of the frame, the original index and columns will be retained.

        .. deprecated:: 0.23.0

        by result_type='broadcast'.

:param raw:
    bool, default False

:param \* ``False``:
    passes each row or column as a Series to the
        function.

:param \* ``True``:
    the passed function will receive ndarray objects
        instead.
        If you are just applying a NumPy reduction function this will
        achieve much better performance.

:param reduce:
    bool or None, default None
        Try to apply reduction procedures. If the DataFrame is empty,
        `apply` will use `reduce` to determine whether the result
        should be a Series or a DataFrame. If ``reduce=None`` (the
        default), `apply`'s return value will be guessed by calling
        `func` on an empty Series
        (note: while guessing, exceptions raised by `func` will be
        ignored).
        If ``reduce=True`` a Series will always be returned, and if
        ``reduce=False`` a DataFrame will always be returned.

        .. deprecated:: 0.23.0

        by ``result_type='reduce'``.

:param result_type:
    {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

:param \* 'expand':
    list-like results will be turned into columns.

:param \* 'reduce':
    returns a Series if possible rather than expanding
        list-like results. This is the opposite of 'expand'.

:param \* 'broadcast':
    results will be broadcast to the original shape
        of the DataFrame, the original index and columns will be
        retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.

        .. versionadded:: 0.23.0

:param args:
    tuple
        Positional arguments to pass to `func` in addition to the
        array/series.
        \*\*kwds
        Additional keyword arguments to pass as keywords arguments to
        `func`.

:return: Series or DataFrame
    Result of applying ``func`` along the given axis of the
    DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

