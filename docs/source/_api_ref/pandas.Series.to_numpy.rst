.. _pandas.Series.to_numpy:

:orphan:

pandas.Series.to_numpy
**********************

A NumPy ndarray representing the values in this Series or Index.

.. versionadded:: 0.24.0

:param dtype:
    str or numpy.dtype, optional
        The dtype to pass to :meth:`numpy.asarray`

:param copy:
    bool, default False
        Whether to ensure that the returned value is a not a view on
        another array. Note that ``copy=False`` does not *ensure* that
        ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
        a copy is made, even if not strictly necessary.

:return: numpy.ndarray



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

