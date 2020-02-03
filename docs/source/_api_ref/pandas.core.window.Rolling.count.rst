.. _pandas.core.window.Rolling.count:

:orphan:

pandas.core.window.Rolling.count
********************************

The rolling count of any non-NaN observations inside the window.

:return: Series or DataFrame
    Returned object type is determined by the caller of the rolling
    calculation.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_count.py
   :language: python
   :lines: 27-
   :caption: Count of any non-NaN observations inside the window.
   :name: ex_series_rolling_count

.. command-output:: python ./series/rolling/series_rolling_count.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.count <pandas.Series.count>`
        Similar method for Series.
    :ref:`DataFrame.count <pandas.DataFrame.count>`
        Similar method for DataFrame.

