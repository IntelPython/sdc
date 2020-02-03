.. _pandas.Series.infer_objects:

:orphan:

pandas.Series.infer_objects
***************************

Attempt to infer better dtypes for object columns.

Attempts soft conversion of object-dtyped
columns, leaving non-object and unconvertible
columns unchanged. The inference rules are the
same as during normal Series/DataFrame construction.

.. versionadded:: 0.21.0

:return: converted : same type as input object



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

