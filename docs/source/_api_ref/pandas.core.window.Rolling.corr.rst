.. _pandas.core.window.Rolling.corr:

:orphan:

pandas.core.window.Rolling.corr
*******************************

Calculate rolling correlation.

:param other:
    Series, DataFrame, or ndarray, optional
        If not supplied then will default to self.

:param pairwise:
    bool, default None
        Calculate pairwise combinations of columns within a
        DataFrame. If `other` is not specified, defaults to `True`,
        otherwise defaults to `False`.
        Not relevant for :class:`~pandas.Series`.
        \*\*kwargs
        Unused.

:return: Series or DataFrame
    Returned object type is determined by the caller of the
    rolling calculation.

Limitations
-----------
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
Resulting Series has default index and name.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_corr.py
   :language: python
   :lines: 27-
   :caption: Calculate rolling correlation.
   :name: ex_series_rolling_corr

.. command-output:: python ./series/rolling/series_rolling_corr.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.corr <pandas.Series.corr>`
        Similar method for Series.
    :ref:`DataFrame.corr <pandas.DataFrame.corr>`
        Similar method for DataFrame.

