.. _pandas.Series.squeeze:

:orphan:

pandas.Series.squeeze
*********************

Squeeze 1 dimensional axis objects into scalars.

Series or DataFrames with a single element are squeezed to a scalar.
DataFrames with a single column or a single row are squeezed to a
Series. Otherwise the object is unchanged.

This method is most useful when you don't know if your
object is a Series or DataFrame, but you do know it has just a single
column. In that case you can safely call `squeeze` to ensure you have a
Series.

:param axis:
    {0 or 'index', 1 or 'columns', None}, default None
        A specific axis to squeeze. By default, all length-1 axes are
        squeezed.

        .. versionadded:: 0.20.0

:return: DataFrame, Series, or scalar
    The projection after squeezing `axis` or all the axes.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

