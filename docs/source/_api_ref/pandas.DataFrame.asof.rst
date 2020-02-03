.. _pandas.DataFrame.asof:

:orphan:

pandas.DataFrame.asof
*********************

Return the last row(s) without any NaNs before `where`.

The last row (for each element in `where`, if list) without any
NaN is taken.
In case of a :class:`~pandas.DataFrame`, the last row without NaN
considering only the subset of columns (if not `None`)

.. versionadded:: 0.19.0 For DataFrame

If there is no good value, NaN is returned for a Series or
a Series of NaN values for a DataFrame

:param where:
    date or array-like of dates
        Date(s) before which the last row(s) are returned.

:param subset:
    str or array-like of str, default `None`
        For DataFrame, if not `None`, only use these columns to
        check for NaNs.

:return: scalar, Series, or DataFrame

    The return can be:

    - scalar : when `self` is a Series and `where` is a scalar
    - Series: when `self` is a Series and `where` is an array-like,
        or when `self` is a DataFrame and `where` is a scalar
    - DataFrame : when `self` is a DataFrame and `where` is an
        array-like

    Return scalar, Series, or DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

