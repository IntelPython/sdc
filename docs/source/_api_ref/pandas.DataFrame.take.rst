.. _pandas.DataFrame.take:

:orphan:

pandas.DataFrame.take
*********************

Return the elements in the given *positional* indices along an axis.

This means that we are not indexing according to actual values in
the index attribute of the object. We are indexing according to the
actual position of the element in the object.

:param indices:
    array-like
        An array of ints indicating which positions to take.

:param axis:
    {0 or 'index', 1 or 'columns', None}, default 0
        The axis on which to select elements. ``0`` means that we are
        selecting rows, ``1`` means that we are selecting columns.

:param is_copy:
    bool, default True
        Whether to return a copy of the original object or not.
        \*\*kwargs
        For compatibility with :meth:`numpy.take`. Has no effect on the
        output.

:return: taken : same type as caller
    An array-like containing the elements taken from the object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

