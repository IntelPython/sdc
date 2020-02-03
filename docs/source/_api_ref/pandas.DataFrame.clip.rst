.. _pandas.DataFrame.clip:

:orphan:

pandas.DataFrame.clip
*********************

Trim values at input threshold(s).

Assigns values outside boundary to boundary values. Thresholds
can be singular values or array like, and in the latter case
the clipping is performed element-wise in the specified axis.

:param lower:
    float or array_like, default None
        Minimum threshold value. All values below this
        threshold will be set to it.

:param upper:
    float or array_like, default None
        Maximum threshold value. All values above this
        threshold will be set to it.

:param axis:
    int or str axis name, optional
        Align object with lower and upper along the given axis.

:param inplace:
    bool, default False
        Whether to perform the operation in place on the data.

        .. versionadded:: 0.21.0

        Additional keywords have no effect but might be accepted
        for compatibility with numpy.

:return: Series or DataFrame
    Same type as calling object with the values outside the
    clip boundaries replaced.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

