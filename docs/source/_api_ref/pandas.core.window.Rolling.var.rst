.. _pandas.core.window.Rolling.var:

:orphan:

pandas.core.window.Rolling.var
******************************

Calculate unbiased rolling variance.

Normalized by N-1 by default. This can be changed using the `ddof`
argument.

:param ddof:
    int, default 1
        Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        \*args, \*\*kwargs
        For NumPy compatibility. No additional arguments are used.

:return: Series or DataFrame
    Returns the same object type as the caller of the rolling calculation.

Limitations
-----------
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_var.py
   :language: python
   :lines: 27-
   :caption: Calculate unbiased rolling variance.
   :name: ex_series_rolling_var

.. command-output:: python ./series/rolling/series_rolling_var.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.var <pandas.Series.var>`
        Similar method for Series.
    :ref:`DataFrame.var <pandas.DataFrame.var>`
        Similar method for DataFrame.

