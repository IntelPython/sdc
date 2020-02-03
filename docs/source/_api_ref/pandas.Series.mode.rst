.. _pandas.Series.mode:

:orphan:

pandas.Series.mode
******************

Return the mode(s) of the dataset.

Always returns Series even if only one value is returned.

:param dropna:
    bool, default True
        Don't consider counts of NaN/NaT.

        .. versionadded:: 0.24.0

:return: Series
    Modes of the Series in sorted order.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

