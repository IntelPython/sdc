.. _pandas.DataFrame.drop:

:orphan:

pandas.DataFrame.drop
*********************

Drop specified labels from rows or columns.

Remove rows or columns by specifying label names and corresponding
axis, or by specifying directly index or column names. When using a
multi-index, labels on different levels can be removed by specifying
the level.

:param labels:
    single label or list-like
        Index or column labels to drop.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Whether to drop labels from the index (0 or 'index') or
        columns (1 or 'columns').

:param index:
    single label or list-like
        Alternative to specifying axis (``labels, axis=0``
        is equivalent to ``index=labels``).

        .. versionadded:: 0.21.0

:param columns:
    single label or list-like
        Alternative to specifying axis (``labels, axis=1``
        is equivalent to ``columns=labels``).

        .. versionadded:: 0.21.0

:param level:
    int or level name, optional
        For MultiIndex, level from which the labels will be removed.

:param inplace:
    bool, default False
        If True, do operation inplace and return None.

:param errors:
    {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are
        dropped.

:return: DataFrame
    DataFrame without the removed index or column labels.

:raises:
    KeyError
        If any of the labels is not found in the selected axis.

Limitations
-----------
Parameter columns is expected to be a Literal value with one column name or Tuple with columns names.

Examples
--------
.. literalinclude:: ../../../examples/dataframe/dataframe_drop.py
    :language: python
    :lines: 37-
    :caption: Drop specified columns from DataFrame.
    :name: ex_dataframe_drop

.. command-output:: python ./dataframe/dataframe_drop.py
    :cwd: ../../../examples

.. note::
    Parameters axis, index, level, inplace, errors are currently unsupported
    by Intel Scalable Dataframe Compiler
    Currently multi-indexing is not supported.

.. seealso::
    :ref:`DataFrame.loc <pandas.DataFrame.loc>`
        Label-location based indexer for selection by label.
    :ref:`DataFrame.dropna <pandas.DataFrame.dropna>`
        Return DataFrame with labels on given axis omitted where (all or any) data are missing.
    :ref:`DataFrame.drop_duplicates <pandas.DataFrame.drop_duplicates>`
        Return DataFrame with duplicate rows removed, optionally only considering certain columns.
    :ref:`Series.drop <pandas.Series.drop>`
        Return Series with specified index labels removed.

