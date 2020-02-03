.. _pandas.core.window.Rolling.kurt:

:orphan:

pandas.core.window.Rolling.kurt
*******************************

Calculate unbiased rolling kurtosis.

This function uses Fisher's definition of kurtosis without bias.

\*\*kwargs
Under Review.
:return: Series or DataFrame
    Returned object type is determined by the caller of the rolling
    calculation.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_kurt.py
   :language: python
   :lines: 27-
   :caption: Calculate unbiased rolling kurtosis.
   :name: ex_series_rolling_kurt

.. command-output:: python ./series/rolling/series_rolling_kurt.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.kurt <pandas.Series.kurt>`
        Similar method for Series.
    :ref:`DataFrame.kurt <pandas.DataFrame.kurt>`
        Similar method for DataFrame.

