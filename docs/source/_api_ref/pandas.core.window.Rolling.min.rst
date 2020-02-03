.. _pandas.core.window.Rolling.min:

:orphan:

pandas.core.window.Rolling.min
******************************

Calculate the rolling minimum.

\*\*kwargs
Under Review.
:return: Series or DataFrame
    Returned object type is determined by the caller of the rolling
    calculation.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_min.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling minimum.
   :name: ex_series_rolling_min

.. command-output:: python ./series/rolling/series_rolling_min.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.min <pandas.Series.min>`
        Similar method for Series.
    :ref:`DataFrame.min <pandas.DataFrame.min>`
        Similar method for DataFrame.

