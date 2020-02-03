.. _pandas.Series.quantile:

:orphan:

pandas.Series.quantile
**********************

Return value at the given quantile.

:param q:
    float or array-like, default 0.5 (50% quantile)
        0 <= q <= 1, the quantile(s) to compute.

:param interpolation:
    {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        .. versionadded:: 0.18.0

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - linear: `i + (j - i) \* fraction`, where `fraction` is the
            fractional part of the index surrounded by `i` and `j`.
        - lower: `i`.
        - higher: `j`.
        - nearest: `i` or `j` whichever is nearest.
        - midpoint: (`i` + `j`) / 2.

:return: float or Series
    If ``q`` is an array, a Series will be returned where the
    index is ``q`` and the values are the quantiles, otherwise
    a float will be returned.

Examples
--------
.. literalinclude:: ../../../examples/series/series_quantile.py
   :language: python
   :lines: 27-
   :caption: Computing quantile for the Series
   :name: ex_series_quantile

.. command-output:: python ./series/series_quantile.py
   :cwd: ../../../examples

.. note::
    Parameter interpolation is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    `numpy.absolute <https://docs.scipy.org/doc/numpy/reference/generated/numpy.percentile.html#numpy.percentile>`_

