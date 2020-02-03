.. _pandas.DataFrame.dot:

:orphan:

pandas.DataFrame.dot
********************

Compute the matrix multiplication between the DataFrame and other.

This method computes the matrix product between the DataFrame and the
values of an other Series, DataFrame or a numpy array.

It can also be called using ``self @ other`` in Python >= 3.5.

:param other:
    Series, DataFrame or array-like
        The other object to compute the matrix product with.

:return: Series or DataFrame
    If other is a Series, return the matrix product between self and
    other as a Serie. If other is a DataFrame or a numpy.array, return
    the matrix product of self and other in a DataFrame of a np.array.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

