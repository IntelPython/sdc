.. _pandas.core.window.Rolling.cov:

:orphan:

pandas.core.window.Rolling.cov
******************************

Calculate the rolling sample covariance.

:param other:
    Series, DataFrame, or ndarray, optional
        If not supplied then will default to self and produce pairwise
        output.

:param pairwise:
    bool, default None
        If False then only matching columns between self and other will be
        used and the output will be a DataFrame.
        If True then all pairwise combinations will be calculated and the
        output will be a MultiIndexed DataFrame in the case of DataFrame
        inputs. In the case of missing elements, only complete pairwise
        observations will be used.

:param ddof:
    int, default 1
        Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        \*\*kwargs
        Keyword arguments to be passed into func.

        :return: Series or DataFrame
            Return type is determined by the caller.

Limitations
-----------
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
Resulting Series has default index and name.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_cov.py
   :language: python
   :lines: 27-
   :caption: Calculate rolling covariance.
   :name: ex_series_rolling_cov

.. command-output:: python ./series/rolling/series_rolling_cov.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.cov <pandas.Series.cov>`
        Similar method for Series.
    :ref:`DataFrame.cov <pandas.DataFrame.cov>`
        Similar method for DataFrame.

