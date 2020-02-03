.. _pandas.Series.dropna:

:orphan:

pandas.Series.dropna
********************

Return a new Series with missing values removed.

See the :ref:`User Guide <missing_data>` for more on which values are
considered missing, and how to work with missing data.

:param axis:
    {0 or 'index'}, default 0
        There is only one axis to drop values from.

:param inplace:
    bool, default False
        If True, do operation inplace and return None.
        \*\*kwargs
        Not in use.

:return: Series
    Series with NA entries dropped from it.

Limitations
-----------
- Parameter inplace is currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_dropna.py
   :language: python
   :lines: 27-
   :caption: Return a new Series with missing values removed.
   :name: ex_series_dropna

.. command-output:: python ./series/series_dropna.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.isna <pandas.Series.isna>`
        Indicate missing values.

    :ref:`Series.notna <pandas.Series.notna>`
        Indicate existing (non-missing) values.

    :ref:`Series.fillna <pandas.Series.fillna>`
        Replace missing values.

    :ref:`DataFrame.dropna <pandas.DataFrame.dropna>`
        Drop rows or columns which contain NA values.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.dropna.html#pandas.Index.dropna>`_
        Return Index without NA/NaN values

