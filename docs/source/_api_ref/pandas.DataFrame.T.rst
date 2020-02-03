.. _pandas.DataFrame.T:

:orphan:

pandas.DataFrame.T
******************

Transpose index and columns.

Reflect the DataFrame over its main diagonal by writing rows as columns
and vice-versa. The property :attr:`.T` is an accessor to the method
:meth:`transpose`.

:param copy:
    bool, default False
        If True, the underlying data is copied. Otherwise (default), no
        copy is made if possible.
        \*args, \*\*kwargs
        Additional keywords have no effect but might be accepted for
        compatibility with numpy.

:return: DataFrame
    The transposed DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

