.. _pandas.core.window.Rolling.skew:

:orphan:

pandas.core.window.Rolling.skew
*******************************

Unbiased rolling skewness.

\*\*kwargs
Keyword arguments to be passed into func.
        :return: Series or DataFrame
            Return type is determined by the caller.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_skew.py
   :language: python
   :lines: 27-
   :caption: Unbiased rolling skewness.
   :name: ex_series_rolling_skew

.. command-output:: python ./series/rolling/series_rolling_skew.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.skew <pandas.Series.skew>`
        Similar method for Series.
    :ref:`DataFrame.skew <pandas.DataFrame.skew>`
        Similar method for DataFrame.

