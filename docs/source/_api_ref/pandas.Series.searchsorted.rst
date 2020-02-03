.. _pandas.Series.searchsorted:

:orphan:

pandas.Series.searchsorted
**************************

Find indices where elements should be inserted to maintain order.

Find the indices into a sorted Series `self` such that, if the
corresponding elements in `value` were inserted before the indices,
the order of `self` would be preserved.

:param value:
    array_like
        Values to insert into `self`.

:param side:
    {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `self`).

:param sorter:
    1-D array_like, optional
        Optional array of integer indices that sort `self` into ascending
        order. They are typically the result of ``np.argsort``.

:return: int or array of int
    A scalar or array of insertion points with the
    same shape as `value`.

    .. versionchanged :: 0.24.0

    Previously, scalar inputs returned an 1-item array for
    :class:`Series` and :class:`Categorical`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

