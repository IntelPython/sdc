.. _pandas.Series.sparse.from_coo:

:orphan:

pandas.Series.sparse.from_coo
*****************************

Create a SparseSeries from a scipy.sparse.coo_matrix.

:param A:
    scipy.sparse.coo_matrix

:param dense_index:
    bool, default False
        If False (default), the SparseSeries index consists of only the
        coords of the non-null entries of the original coo_matrix.
        If True, the SparseSeries index consists of the full sorted
        (row, col) coordinates of the coo_matrix.

:return: s : SparseSeries



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

