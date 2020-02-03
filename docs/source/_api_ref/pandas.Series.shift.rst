.. _pandas.Series.shift:

:orphan:

pandas.Series.shift
*******************

Shift index by desired number of periods with an optional time `freq`.

When `freq` is not passed, shift the index without realigning the data.
If `freq` is passed (in this case, the index must be date or datetime,
or it will raise a `NotImplementedError`), the index will be
increased using the periods and the `freq`.

:param periods:
    int
        Number of periods to shift. Can be positive or negative.

:param freq:
    DateOffset, tseries.offsets, timedelta, or str, optional
        Offset to use from the tseries module or time rule (e.g. 'EOM').
        If `freq` is specified then the index values are shifted but the
        data is not realigned. That is, use `freq` if you would like to
        extend the index when shifting and preserve the original data.

:param axis:
    {0 or 'index', 1 or 'columns', None}, default None
        Shift direction.

:param fill_value:
    object, optional
        The scalar value to use for newly introduced missing values.
        the default depends on the dtype of `self`.
        For numeric data, ``np.nan`` is used.
        For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
        For extension dtypes, ``self.dtype.na_value`` is used.

        .. versionchanged:: 0.24.0

:return: Series
    Copy of input object, shifted.

Limitations
-----------
- Parameter freq is currently unsupported by Intel Scalable Dataframe Compiler

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_shift.py
   :language: python
   :lines: 27-
   :caption: Shift index by desired number of periods with an optional time freq.
   :name: ex_series_shift

.. command-output:: python ./series/series_shift.py
   :cwd: ../../../examples

.. seealso::

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.shift.html#pandas.Index.shift>`_
        Shift index by desired number of time frequency increments.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.tshift.html#pandas.Series.tshift>`_
        Shift the time index, using the index’s frequency if available.

