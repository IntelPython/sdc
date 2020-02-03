.. _pandas.Series.dot:

:orphan:

pandas.Series.dot
*****************

Compute the dot product between the Series and the columns of other.

This method computes the dot product between the Series and another
one, or the Series and each columns of a DataFrame, or the Series and
each columns of an array.

It can also be called using `self @ other` in Python >= 3.5.

:param other:
    Series, DataFrame or array-like
        The other object to compute the dot product with its columns.

:return: scalar, Series or numpy.ndarray
    Return the dot product of the Series and other if other is a
    Series, the Series of the dot product of Series and each rows of
    other if other is a DataFrame or a numpy.ndarray between the Series
    and each columns of the numpy array.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

