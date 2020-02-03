.. _pandas.DataFrame.sparse.to_coo:

:orphan:

pandas.DataFrame.sparse.to_coo
******************************

Return the contents of the frame as a sparse SciPy COO matrix.

.. versionadded:: 0.25.0

:return: coo_matrix : scipy.sparse.spmatrix
    If the caller is heterogeneous and contains booleans or objects,
    the result will be of dtype=object. See Notes.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

