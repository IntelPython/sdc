.. _pandas.Series.append:

:orphan:

pandas.Series.append
********************

Concatenate two or more Series.

:param to_append:
    Series or list/tuple of Series
        Series to append with self.

:param ignore_index:
    bool, default False
        If True, do not use the index labels.

        .. versionadded:: 0.19.0

:param verify_integrity:
    bool, default False
        If True, raise Exception on creating index with duplicates.

:return: Series
    Concatenated Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

