.. _pandas.Series.nunique:

:orphan:

pandas.Series.nunique
*********************

Return number of unique elements in the object.

Excludes NA values by default.

:param dropna:
    bool, default True
        Don't include NaN in the count.

:return: int

Examples
--------
.. literalinclude:: ../../../examples/series/series_nunique.py
   :language: python
   :lines: 27-
   :caption: Return number of unique elements in the object.
   :name: ex_series_nunique

.. command-output:: python ./series/series_nunique.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.nunique <pandas.DataFrame.nunique>`
        Method nunique for DataFrame.

    :ref:`Series.count <pandas.Series.count>`
        Count non-NA/null observations in the Series.

