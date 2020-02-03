.. _pandas.Series.sparse.to_coo:

:orphan:

pandas.Series.sparse.to_coo
***************************

Create a scipy.sparse.coo_matrix from a SparseSeries with MultiIndex.

Use row_levels and column_levels to determine the row and column
coordinates respectively. row_levels and column_levels are the names
(labels) or numbers of the levels. {row_levels, column_levels} must be
a partition of the MultiIndex level names (or numbers).

:param row_levels:
    tuple/list

:param column_levels:
    tuple/list

:param sort_labels:
    bool, default False
        Sort the row and column labels before forming the sparse matrix.

:return: y : scipy.sparse.coo_matrix
    rows : list (row labels)
    columns : list (column labels)



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

