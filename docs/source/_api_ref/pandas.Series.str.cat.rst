.. _pandas.Series.str.cat:

:orphan:

pandas.Series.str.cat
*********************

Concatenate strings in the Series/Index with given separator.

If `others` is specified, this function concatenates the Series/Index
and elements of `others` element-wise.
If `others` is not passed, then all values in the Series/Index are
concatenated into a single string with a given `sep`.

:param others:
    Series, Index, DataFrame, np.ndarray or list-like
        Series, Index, DataFrame, np.ndarray (one- or two-dimensional) and
        other list-likes of strings must have the same length as the
        calling Series/Index, with the exception of indexed objects (i.e.
        Series/Index/DataFrame) if `join` is not None.

        If others is a list-like that contains a combination of Series,
        Index or np.ndarray (1-dim), then all elements will be unpacked and
        must satisfy the above criteria individually.

        If others is None, the method returns the concatenation of all
        strings in the calling Series/Index.

:param sep:
    str, default ''
        The separator between the different elements/columns. By default
        the empty string `''` is used.

:param na_rep:
    str or None, default None
        Representation that is inserted for all missing values:

        - If `na_rep` is None, and `others` is None, missing values in the
            Series/Index are omitted from the result.
        - If `na_rep` is None, and `others` is not None, a row containing a
            missing value in any of the columns (before concatenation) will
            have a missing value in the result.

:param join:
    {'left', 'right', 'outer', 'inner'}, default None
        Determines the join-style between the calling Series/Index and any
        Series/Index/DataFrame in `others` (objects without an index need
        to match the length of the calling Series/Index). If None,
        alignment is disabled, but this option will be removed in a future
        version of pandas and replaced with a default of `'left'`. To
        disable alignment, use `.values` on any Series/Index/DataFrame in
        `others`.

        .. versionadded:: 0.23.0

:return: str, Series or Index
    If `others` is None, `str` is returned, otherwise a `Series/Index`
    (same type as caller) of objects is returned.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

