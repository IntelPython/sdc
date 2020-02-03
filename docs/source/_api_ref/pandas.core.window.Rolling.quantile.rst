.. _pandas.core.window.Rolling.quantile:

:orphan:

pandas.core.window.Rolling.quantile
***********************************

Calculate the rolling quantile.

:param quantile:
    float
        Quantile to compute. 0 <= quantile <= 1.

:param interpolation:
    {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        .. versionadded:: 0.23.0

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - linear: `i + (j - i) \* fraction`, where `fraction` is the
            fractional part of the index surrounded by `i` and `j`.
        - lower: `i`.
        - higher: `j`.
        - nearest: `i` or `j` whichever is nearest.
        - midpoint: (`i` + `j`) / 2.
            \*\*kwargs:
            For compatibility with other rolling methods. Has no effect on
            the result.

:return: Series or DataFrame
    Returned object type is determined by the caller of the rolling
    calculation.

Limitations
-----------
Supported ``interpolation`` only can be `'linear'`.
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_quantile.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling quantile.
   :name: ex_series_rolling_quantile

.. command-output:: python ./series/rolling/series_rolling_quantile.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.quantile <pandas.Series.quantile>`
        Similar method for Series.
    :ref:`DataFrame.quantile <pandas.DataFrame.quantile>`
        Similar method for DataFrame.

