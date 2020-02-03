.. _pandas.Series.expanding:

:orphan:

pandas.Series.expanding
***********************

Provide expanding transformations.

.. versionadded:: 0.18.0

:param min_periods:
    int, default 1
        Minimum number of observations in window required to have a value
        (otherwise result is NA).

:param center:
    bool, default False
        Set the labels at the center of the window.

:param axis:
    int or str, default 0

:return: a Window sub-classed for the particular operation



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

