.. _pandas.core.window.Rolling.sum:

:orphan:

pandas.core.window.Rolling.sum
******************************

Calculate rolling sum of given DataFrame or Series.

\*args, \*\*kwargs
For compatibility with other rolling methods. Has no effect
on the computed value.
:return: Series or DataFrame
    Same type as the input, with the same index, containing the
    rolling sum.

Limitations
-----------
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_sum.py
   :language: python
   :lines: 27-
   :caption: Calculate rolling sum of given Series.
   :name: ex_series_rolling_sum

.. command-output:: python ./series/rolling/series_rolling_sum.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.sum <pandas.Series.sum>`
        Similar method for Series.
    :ref:`DataFrame.sum <pandas.DataFrame.sum>`
        Similar method for DataFrame.

