.. _pandas.DataFrame.rename:

:orphan:

pandas.DataFrame.rename
***********************

Alter axes labels.

Function / dict values must be unique (1-to-1). Labels not contained in
a dict / Series will be left as-is. Extra labels listed don't throw an
error.

See the :ref:`user guide <basics.rename>` for more.

:param mapper:
    dict-like or function
        Dict-like or functions transformations to apply to
        that axis' values. Use either ``mapper`` and ``axis`` to
        specify the axis to target with ``mapper``, or ``index`` and
        ``columns``.

:param index:
    dict-like or function
        Alternative to specifying axis (``mapper, axis=0``
        is equivalent to ``index=mapper``).

:param columns:
    dict-like or function
        Alternative to specifying axis (``mapper, axis=1``
        is equivalent to ``columns=mapper``).

:param axis:
    int or str
        Axis to target with ``mapper``. Can be either the axis name
        ('index', 'columns') or number (0, 1). The default is 'index'.

:param copy:
    bool, default True
        Also copy underlying data.

:param inplace:
    bool, default False
        Whether to return a new DataFrame. If True then value of copy is
        ignored.

:param level:
    int or level name, default None
        In case of a MultiIndex, only rename labels in the specified
        level.

:param errors:
    {'ignore', 'raise'}, default 'ignore'
        If 'raise', raise a `KeyError` when a dict-like `mapper`, `index`,
        or `columns` contains labels that are not present in the Index
        being transformed.
        If 'ignore', existing keys will be renamed and extra keys will be
        ignored.

:return: DataFrame
    DataFrame with the renamed axis labels.

:raises:
    KeyError
        If any of the labels is not found in the selected axis and
        "errors='raise'".



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

