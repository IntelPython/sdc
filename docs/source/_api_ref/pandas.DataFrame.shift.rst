.. _pandas.DataFrame.shift:

:orphan:

pandas.DataFrame.shift
**********************

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

:return: DataFrame
    Copy of input object, shifted.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

