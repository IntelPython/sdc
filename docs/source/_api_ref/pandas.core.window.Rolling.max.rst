.. _pandas.core.window.Rolling.max:

:orphan:

pandas.core.window.Rolling.max
******************************

Calculate the rolling maximum.

\*args, \*\*kwargs
Arguments and keyword arguments to be passed into func.
        :return: Series or DataFrame
            Return type is determined by the caller.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_max.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling maximum.
   :name: ex_series_rolling_max

.. command-output:: python ./series/rolling/series_rolling_max.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.max <pandas.Series.max>`
        Similar method for Series.
    :ref:`DataFrame.max <pandas.DataFrame.max>`
        Similar method for DataFrame.

