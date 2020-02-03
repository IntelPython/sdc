.. _pandas.Series.rename_axis:

:orphan:

pandas.Series.rename_axis
*************************

Set the name of the axis for the index or columns.

:param mapper:
    scalar, list-like, optional
        Value to set the axis name attribute.

:param index, columns:
    scalar, list-like, dict-like or function, optional
        A scalar, list-like, dict-like or functions transformations to
        apply to that axis' values.

        Use either ``mapper`` and ``axis`` to
        specify the axis to target with ``mapper``, or ``index``
        and/or ``columns``.

        .. versionchanged:: 0.24.0

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to rename.

:param copy:
    bool, default True
        Also copy underlying data.

:param inplace:
    bool, default False
        Modifies the object directly, instead of creating a new Series
        or DataFrame.

:return: Series, DataFrame, or None
    The same type as the caller or None if `inplace` is True.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

