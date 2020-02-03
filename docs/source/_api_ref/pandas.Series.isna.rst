.. _pandas.Series.isna:

:orphan:

pandas.Series.isna
******************

Detect missing values.

Return a boolean same-sized object indicating if the values are NA.
NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
values.
Everything else gets mapped to False values. Characters such as empty
strings ``''`` or :attr:`numpy.inf` are not considered NA values
(unless you set ``pandas.options.mode.use_inf_as_na = True``).

:return: Series
    Mask of bool values for each element in Series that
    indicates whether an element is not an NA value.

Examples
--------
.. literalinclude:: ../../../examples/series/series_isna.py
   :language: python
   :lines: 27-
   :caption: Detect missing values.
   :name: ex_series_isna

.. command-output:: python ./series/series_isna.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.isnull <pandas.Series.isnull>`
        Alias of isna.

    :ref:`Series.notna <pandas.Series.notna>`
        Boolean inverse of isna.

    :ref:`Series.dropna <pandas.Series.dropna>`
        Omit axes labels with missing values.

    `pandas.absolute <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna>`_
        Top-level isna.

