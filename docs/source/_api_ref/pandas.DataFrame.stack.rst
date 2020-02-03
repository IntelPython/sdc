.. _pandas.DataFrame.stack:

:orphan:

pandas.DataFrame.stack
**********************

Stack the prescribed level(s) from columns to index.

Return a reshaped DataFrame or Series having a multi-level
index with one or more new inner-most levels compared to the current
DataFrame. The new inner-most levels are created by pivoting the
columns of the current dataframe:

  - if the columns have a single level, the output is a Series;
  - if the columns have multiple levels, the new index
      level(s) is (are) taken from the prescribed level(s) and
      the output is a DataFrame.

The new index levels are sorted.

:param level:
    int, str, list, default -1
        Level(s) to stack from the column axis onto the index
        axis, defined as one index or label, or a list of indices
        or labels.

:param dropna:
    bool, default True
        Whether to drop rows in the resulting Frame/Series with
        missing values. Stacking a column level onto the index
        axis can create combinations of index and column values
        that are missing from the original dataframe. See Examples
        section.

:return: DataFrame or Series
    Stacked dataframe or series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

