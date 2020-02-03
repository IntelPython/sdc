.. _pandas.Series.factorize:

:orphan:

pandas.Series.factorize
***********************

Encode the object as an enumerated type or categorical variable.

This method is useful for obtaining a numeric representation of an
array when all that matters is identifying distinct values. `factorize`
is available as both a top-level function :func:`pandas.factorize`,
and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

:param sort:
    boolean, default False
        Sort `uniques` and shuffle `labels` to maintain the
        relationship.

:param na_sentinel:
    int, default -1
        Value to mark "not found".

:return: labels : ndarray
    An integer ndarray that's an indexer into `uniques`.
    ``uniques.take(labels)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
    The unique valid values. When `values` is Categorical, `uniques`
    is a Categorical. When `values` is some other pandas object, an
    `Index` is returned. Otherwise, a 1-D ndarray is returned.

.. note ::
    Even if there's a missing value in `values`, `uniques` will
    *not* contain an entry for it.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

