.. _pandas.core.window.Rolling.median:

:orphan:

pandas.core.window.Rolling.median
*********************************

Calculate the rolling median.

\*\*kwargs
For compatibility with other rolling methods. Has no effect
on the computed median.
:return: Series or DataFrame
    Returned type is the same as the original object.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_median.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling median.
   :name: ex_series_rolling_median

.. command-output:: python ./series/rolling/series_rolling_median.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.median <pandas.Series.median>`
        Similar method for Series.
    :ref:`DataFrame.median <pandas.DataFrame.median>`
        Similar method for DataFrame.

