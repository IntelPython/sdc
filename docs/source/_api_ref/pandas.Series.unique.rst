.. _pandas.Series.unique:

:orphan:

pandas.Series.unique
********************

Return unique values of Series object.

Uniques are returned in order of appearance. Hash table-based unique,
therefore does NOT sort.

:return: ndarray or ExtensionArray
    The unique values returned as a NumPy array. See Notes.

Examples
--------
.. literalinclude:: ../../../examples/series/series_unique.py
   :language: python
   :lines: 27-
   :caption: Getting unique values in Series
   :name: ex_series_unique

.. command-output:: python ./series/series_unique.py
   :cwd: ../../../examples

