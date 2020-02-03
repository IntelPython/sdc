.. _pandas.Series.xs:

:orphan:

pandas.Series.xs
****************

Return cross-section from the Series/DataFrame.

This method takes a `key` argument to select data at a particular
level of a MultiIndex.

:param key:
    label or tuple of label
        Label contained in the index, or partially in a MultiIndex.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Axis to retrieve cross-section on.

:param level:
    object, defaults to first n levels (n=1 or len(key))
        In case of a key partially contained in a MultiIndex, indicate
        which levels are used. Levels can be referred by label or position.

:param drop_level:
    bool, default True
        If False, returns object with same levels as self.

:return: Series or DataFrame
    Cross-section from the original Series or DataFrame
    corresponding to the selected index levels.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

