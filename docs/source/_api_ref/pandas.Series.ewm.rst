.. _pandas.Series.ewm:

:orphan:

pandas.Series.ewm
*****************

Provide exponential weighted functions.

.. versionadded:: 0.18.0

:param com:
    float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com),\text{ for } com \geq 0`.

:param span:
    float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1),\text{ for } span \geq 1`.

:param halflife:
    float, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - exp(log(0.5) / halflife),\text{for} halflife > 0`.

:param alpha:
    float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.

        .. versionadded:: 0.18.0

:param min_periods:
    int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).

:param adjust:
    bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings
        (viewing EWMA as a moving average).

:param ignore_na:
    bool, default False
        Ignore missing values when calculating weights;
        specify True to reproduce pre-0.15.0 behavior.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. The value 0 identifies the rows, and 1
        identifies the columns.

:return: DataFrame
    A Window sub-classed for the particular operation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

