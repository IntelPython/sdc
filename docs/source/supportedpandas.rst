Supported Pandas Operations
---------------------------

Below is the list of the Pandas operators that HPAT supports.
Optional arguments are not supported unless if specified.
Since Numba doesn't support Pandas, only these operations
can be used for both large and small datasets.

In addition:

* Accessing columns using both getitem (e.g. ``df['A']``) and attribute
  (e.g. ``df.A``) is supported.
* Using columns similar to Numpy arrays and performing data-parallel operations
  listed previously is supported.
* Filtering data frames using boolean arrays is supported
  (e.g. ``df[df.A > .5]``).


Integer NaN Issue
~~~~~~~~~~~~~~~~~

DataFrame columns with integer data need special care. Pandas dynamically
converts integer columns to floating point when NaN values are needed.
This is because Numpy does not support NaN values for integers.
HPAT does not perform this conversion unless enough information is
available at compilation time. Hence, the user is responsible for manual
conversion of integer data to floating point data if needed.

Input/Output
~~~~~~~~~~~~

* :func:`pandas.read_csv`

   * Arguments ``filepath_or_buffer``, ``sep``, ``delimiter``, ``names``, ``usecols``, ``dtype``, and ``parse_dates`` are supported.
   * ``filepath_or_buffer``, ``names`` and ``dtype`` arguments are required.
   * ``names``, ``usecols``, ``parse_dates`` should be constant lists.
   * ``dtype`` should be a constant dictionary of strings and types.

* :func:`pandas.read_parquet`

   * If filename is constant, HPAT finds the schema from file at compilation time. Otherwise, schema should be provided.

General functions
~~~~~~~~~~~~~~~~~

* :func:`pandas.merge`

   * Arguments ``left``, ``right``, ``as_of``, ``how``, ``on``, ``left_on`` and ``right_on`` are supported.
   * ``on``, ``left_on`` and ``right_on`` should be constant strings or constant list of strings.

* :func:`pandas.concat`

   * Input list or tuple of dataframes or series is supported.

Series
~~~~~~

* :func:`pandas.Series`

   * Argument ``data`` can be a list or array.


Attributes:

* :attr:`Series.values`
* :attr:`Series.shape`
* :attr:`Series.ndim`
* :attr:`Series.size`

Methods:

* :meth:`Series.copy`

Indexing, iteration:

* :meth:`Series.iat`
* :meth:`Series.iloc`

Binary operator functions:

* :meth:`Series.add`
* :meth:`Series.sub`
* :meth:`Series.mul`
* :meth:`Series.div`
* :meth:`Series.truediv`
* :meth:`Series.floordiv`
* :meth:`Series.mod`
* :meth:`Series.pow`
* :meth:`Series.combine`
* :meth:`Series.lt`
* :meth:`Series.gt`
* :meth:`Series.le`
* :meth:`Series.ge`
* :meth:`Series.ne`

Function application, GroupBy & Window:

* :meth:`Series.apply`
* :meth:`Series.map`
* :meth:`Series.rolling`

Computations / Descriptive Stats:

* :meth:`Series.abs`
* :meth:`Series.corr`
* :meth:`Series.count`
* :meth:`Series.cov`
* :meth:`Series.cumsum`
* :meth:`Series.describe` currently returns a string instead of Series object.
* :meth:`Series.max`
* :meth:`Series.mean`
* :meth:`Series.median`
* :meth:`Series.min`
* :meth:`Series.nlargest`
* :meth:`Series.nsmallest`
* :meth:`Series.pct_change`
* :meth:`Series.prod`
* :meth:`Series.quantile`
* :meth:`Series.std`
* :meth:`Series.sum`
* :meth:`Series.var`
* :meth:`Series.unique`
* :meth:`Series.nunique`

Reindexing / Selection / Label manipulation:

* :meth:`Series.head`
* :meth:`Series.idxmax`
* :meth:`Series.idxmin`
* :meth:`Series.take`

Missing data handling:

* :meth:`Series.isna`
* :meth:`Series.notna`
* :meth:`Series.dropna`
* :meth:`Series.fillna`

Reshaping, sorting:

* :meth:`Series.argsort`
* :meth:`Series.sort_values`
* :meth:`Series.append`

Time series-related:

* :meth:`Series.shift`

String handling:

* :meth:`Series.str.contains`
* :meth:`Series.str.len`

DataFrame
~~~~~~~~~

* :func:`pandas.DataFrame`

   Only ``data`` argument with a dictionary input is supported.

Attributes and underlying data:

* :attr:`DataFrame.values`

Indexing, iteration:

* :meth:`DataFrame.head`
* :meth:`DataFrame.iat`
* :meth:`DataFrame.iloc`
* :meth:`DataFrame.isin`

Function application, GroupBy & Window:

* :meth:`DataFrame.apply`
* :meth:`DataFrame.groupby`
* :meth:`DataFrame.rolling`

Computations / Descriptive Stats:

* :meth:`DataFrame.describe`

Missing data handling:

* :meth:`DataFrame.dropna`
* :meth:`DataFrame.fillna`

Reshaping, sorting, transposing

* :meth:`DataFrame.pivot_table`

   * Arguments ``values``, ``index``, ``columns`` and ``aggfunc`` are supported.
   * Annotation of pivot values is required.
     For example, `@hpat.jit(pivots={'pt': ['small', 'large']})` declares the output pivot table `pt` will have columns called `small` and `large`.

* :meth:`DataFrame.sort_values` `by` argument should be constant string or constant list of strings.
* :meth:`DataFrame.append`

DatetimeIndex
~~~~~~~~~~~~~

* :attr:`DatetimeIndex.year`
* :attr:`DatetimeIndex.month`
* :attr:`DatetimeIndex.day`
* :attr:`DatetimeIndex.hour`
* :attr:`DatetimeIndex.minute`
* :attr:`DatetimeIndex.second`
* :attr:`DatetimeIndex.microsecond`
* :attr:`DatetimeIndex.nanosecond`
* :attr:`DatetimeIndex.date`

* :meth:`DatetimeIndex.min`
* :meth:`DatetimeIndex.max`


TimedeltaIndex
~~~~~~~~~~~~~~

* :attr:`TimedeltaIndex.days`
* :attr:`TimedeltaIndex.second`
* :attr:`TimedeltaIndex.microsecond`
* :attr:`TimedeltaIndex.nanosecond`


Timestamp
~~~~~~~~~

* :attr:`Timestamp.day`
* :attr:`Timestamp.hour`
* :attr:`Timestamp.microsecond`
* :attr:`Timestamp.month`
* :attr:`Timestamp.nanosecond`
* :attr:`Timestamp.second`
* :attr:`Timestamp.year`

* :meth:`Timestamp.date`

Window
~~~~~~

* :meth:`Rolling.count`
* :meth:`Rolling.sum`
* :meth:`Rolling.mean`
* :meth:`Rolling.median`
* :meth:`Rolling.var`
* :meth:`Rolling.std`
* :meth:`Rolling.min`
* :meth:`Rolling.max`
* :meth:`Rolling.corr`
* :meth:`Rolling.cov`
* :meth:`Rolling.apply`


GroupBy
~~~~~~~


* :meth:`GroupBy.apply`
* :meth:`GroupBy.count`
* :meth:`GroupBy.max`
* :meth:`GroupBy.mean`
* :meth:`GroupBy.median`
* :meth:`GroupBy.min`
* :meth:`GroupBy.prod`
* :meth:`GroupBy.std`
* :meth:`GroupBy.sum`
* :meth:`GroupBy.var`
