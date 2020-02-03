.. _pandas.DataFrame.lookup:

:orphan:

pandas.DataFrame.lookup
***********************

Label-based "fancy indexing" function for DataFrame.

Given equal-length arrays of row and column labels, return an
array of the values corresponding to each (row, col) pair.

:param row_labels:
    sequence
        The row labels to use for lookup

:param col_labels:
    sequence
        The column labels to use for lookup

:return: numpy.ndarray



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

