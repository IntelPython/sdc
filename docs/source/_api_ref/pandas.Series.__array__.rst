.. _pandas.Series.__array__:

:orphan:

pandas.Series.__array__
***********************

Return the values as a NumPy array.

Users should not call this directly. Rather, it is invoked by
:func:`numpy.array` and :func:`numpy.asarray`.

:param dtype:
    str or numpy.dtype, optional
        The dtype to use for the resulting NumPy array. By default,
        the dtype is inferred from the data.

:return: numpy.ndarray
    The values in the series converted to a :class:`numpy.ndarary`
    with the specified `dtype`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

