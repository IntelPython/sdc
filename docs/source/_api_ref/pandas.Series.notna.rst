.. _pandas.Series.notna:

:orphan:

pandas.Series.notna
*******************

Detect existing (non-missing) values.

Return a boolean same-sized object indicating if the values are not NA.
Non-missing values get mapped to True. Characters such as empty
strings ``''`` or :attr:`numpy.inf` are not considered NA values
(unless you set ``pandas.options.mode.use_inf_as_na = True``).
NA values, such as None or :attr:`numpy.NaN`, get mapped to False
values.

:return: Series
    Mask of bool values for each element in Series that
    indicates whether an element is not an NA value.

Examples
--------
.. literalinclude:: ../../../examples/series/series_notna.py
   :language: python
   :lines: 27-
   :caption: Detect existing (non-missing) values.
   :name: ex_series_notna

.. command-output:: python ./series/series_notna.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.notnull <pandas.Series.notnull>`
        Alias of notna.

    :ref:`Series.isna <pandas.Series.isna>`
        Boolean inverse of notna.

    :ref:`Series.dropna <pandas.Series.dropna>`
        Omit axes labels with missing values.

    `pandas.absolute <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.notna.html#pandas.notna>`_
        Top-level notna.

