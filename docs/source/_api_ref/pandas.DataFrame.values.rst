.. _pandas.DataFrame.values:

:orphan:

pandas.DataFrame.values
***********************

Return a Numpy representation of the DataFrame.

.. warning::
   We recommend using :meth:`DataFrame.to_numpy` instead.

Only the values in the DataFrame will be returned, the axes labels
will be removed.

:return: numpy.ndarray
    The values of the DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

