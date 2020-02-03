.. _pandas.DataFrame.pivot:

:orphan:

pandas.DataFrame.pivot
**********************

Return reshaped DataFrame organized by given index / column values.

Reshape data (produce a "pivot" table) based on column values. Uses
unique values from specified `index` / `columns` to form axes of the
resulting DataFrame. This function does not support data
aggregation, multiple values will result in a MultiIndex in the
columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

:param index:
    string or object, optional
        Column to use to make new frame's index. If None, uses
        existing index.

:param columns:
    string or object
        Column to use to make new frame's columns.

:param values:
    string, object or a list of the previous, optional
        Column(s) to use for populating new frame's values. If not
        specified, all remaining columns will be used and the result will
        have hierarchically indexed columns.

        .. versionchanged :: 0.23.0

:return: DataFrame
    Returns reshaped DataFrame.

:raises:
    ValueError:
        When there are any `index`, `columns` combinations with multiple
        values. `DataFrame.pivot_table` when you need to aggregate.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

