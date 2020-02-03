.. _pandas.DataFrame.mask:

:orphan:

pandas.DataFrame.mask
*********************

Replace values where the condition is True.

:param cond:
    boolean Series/DataFrame, array-like, or callable
        Where `cond` is False, keep the original value. Where
        True, replace with corresponding value from `other`.
        If `cond` is callable, it is computed on the Series/DataFrame and
        should return boolean Series/DataFrame or array. The callable must
        not change input Series/DataFrame (though pandas doesn't check it).

        .. versionadded:: 0.18.1

:param other:
    scalar, Series/DataFrame, or callable
        Entries where `cond` is True are replaced with
        corresponding value from `other`.
        If other is callable, it is computed on the Series/DataFrame and
        should return scalar or Series/DataFrame. The callable must not
        change input Series/DataFrame (though pandas doesn't check it).

        .. versionadded:: 0.18.1

:param inplace:
    bool, default False
        Whether to perform the operation in place on the data.

:param axis:
    int, default None
        Alignment axis if needed.

:param level:
    int, default None
        Alignment level if needed.

:param errors:
    str, {'raise', 'ignore'}, default 'raise'
        Note that currently this parameter won't affect
        the results and will always coerce to a suitable dtype.

:param - 'raise':
    allow exceptions to be raised.

:param - 'ignore':
    suppress exceptions. On error return original object.

:param try_cast:
    bool, default False
        Try to cast the result back to the input type (if possible).

:return: Same type as caller



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

