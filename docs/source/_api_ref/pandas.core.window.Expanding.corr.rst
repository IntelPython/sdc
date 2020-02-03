.. _pandas.core.window.Expanding.corr:

:orphan:

pandas.core.window.Expanding.corr
*********************************

Calculate expanding correlation.

:param other:
    Series, DataFrame, or ndarray, optional
        If not supplied then will default to self.

:param pairwise:
    bool, default None
        Calculate pairwise combinations of columns within a
        DataFrame. If `other` is not specified, defaults to `True`,
        otherwise defaults to `False`.
        Not relevant for :class:`~pandas.Series`.
        \*\*kwargs
        Unused.

:return: Series or DataFrame
    Returned object type is determined by the caller of the
    expanding calculation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

