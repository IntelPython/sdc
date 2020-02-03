.. _pandas.Series.count:

:orphan:

pandas.Series.count
*******************

Return number of non-NA/null observations in the Series.

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a smaller Series.

:return: int or Series (if level specified)
    Number of non-null values in the Series.

Examples
--------
.. literalinclude:: ../../../examples/series/series_count.py
   :language: python
   :lines: 27-
   :caption: Counting non-NaN values in Series
   :name: ex_series_count

.. command-output:: python ./series/series_count.py
   :cwd: ../../../examples

.. note::
    Parameter level is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.value_counts <pandas.Series.value_counts>`
    :ref:`Series.value_counts <pandas.Series.value_counts>`
    :ref:`Series.str.len <pandas.Series.str.len>`

