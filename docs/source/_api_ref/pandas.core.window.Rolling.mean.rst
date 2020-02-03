.. _pandas.core.window.Rolling.mean:

:orphan:

pandas.core.window.Rolling.mean
*******************************

Calculate the rolling mean of the values.

\*args
Under Review.
\*\*kwargs
Under Review.
:return: Series or DataFrame
    Returned object type is determined by the caller of the rolling
    calculation.

Limitations
-----------
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_mean.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling mean of the values.
   :name: ex_series_rolling_mean

.. command-output:: python ./series/rolling/series_rolling_mean.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.mean <pandas.Series.mean>`
        Similar method for Series.
    :ref:`DataFrame.mean <pandas.DataFrame.mean>`
        Similar method for DataFrame.

