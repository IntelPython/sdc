.. _pandas.Series.array:

:orphan:

pandas.Series.array
*******************

The ExtensionArray of the data backing this Series or Index.

.. versionadded:: 0.24.0

:return: ExtensionArray
    An ExtensionArray of the values stored within. For extension
    types, this is the actual array. For NumPy native types, this
    is a thin (no copy) wrapper around :class:`numpy.ndarray`.

    ``.array`` differs ``.values`` which may require converting the
    data to a different form.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

