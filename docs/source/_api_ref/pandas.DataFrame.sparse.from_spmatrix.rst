.. _pandas.DataFrame.sparse.from_spmatrix:

:orphan:

pandas.DataFrame.sparse.from_spmatrix
*************************************

Create a new DataFrame from a scipy sparse matrix.

.. versionadded:: 0.25.0

:param data:
    scipy.sparse.spmatrix
        Must be convertible to csc format.

:param index, columns:
    Index, optional
        Row and column labels to use for the resulting DataFrame.
        Defaults to a RangeIndex.

:return: DataFrame
    Each column of the DataFrame is stored as a
    :class:`SparseArray`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

