.. _api_ref.pandas.series:
.. include:: ./../ext_links.txt

Pandas Series
=============
.. currentmodule:: pandas

This is basic `Pandas*`_ data structure representing a dataframe column. In `NumPy*`_ terms this is
one-dimensional ndarray with axis labels.

Constructor
-----------

   :ref:`Series <pandas.Series>`
       One-dimensional ndarray with axis labels (including time series).

        **Unsupported by Intel SDC**.

Attributes/Operators
--------------------

   :ref:`Series.index <pandas.Series.index>`
       The index (axis labels) of the Series.
   :ref:`Series.array <pandas.Series.array>`
       The ExtensionArray of the data backing this Series or Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.values <pandas.Series.values>`
       Return Series as ndarray or ndarray-like depending on the dtype.
   :ref:`Series.dtype <pandas.Series.dtype>`
       Return the dtype object of the underlying data.

        **Unsupported by Intel SDC**.
   :ref:`Series.shape <pandas.Series.shape>`
       Return a tuple of the shape of the underlying data.
   :ref:`Series.nbytes <pandas.Series.nbytes>`
       Return the number of bytes in the underlying data.

        **Unsupported by Intel SDC**.
   :ref:`Series.ndim <pandas.Series.ndim>`
       Number of dimensions of the underlying data, by definition 1.
   :ref:`Series.size <pandas.Series.size>`
       Return the number of elements in the underlying data.
   :ref:`Series.T <pandas.Series.T>`
       Return the transpose, which is by definition self.
   :ref:`Series.memory_usage <pandas.Series.memory_usage>`
       Return the memory usage of the Series.

        **Unsupported by Intel SDC**.
   :ref:`Series.hasnans <pandas.Series.hasnans>`
       Return if I have any nans; enables various perf speedups.

        **Unsupported by Intel SDC**.
   :ref:`Series.empty <pandas.Series.empty>`
       **Unsupported by Intel SDC**.
   :ref:`Series.dtypes <pandas.Series.dtypes>`
       Return the dtype object of the underlying data.

        **Unsupported by Intel SDC**.
   :ref:`Series.name <pandas.Series.name>`
       Return name of the Series.

        **Unsupported by Intel SDC**.
   :ref:`Series.put <pandas.Series.put>`
       Apply the `put` method to its `values` attribute if it has one.

        **Unsupported by Intel SDC**.

Type Conversions
----------------

   :ref:`Series.astype <pandas.Series.astype>`
       Cast a pandas object to a specified dtype ``dtype``.
   :ref:`Series.infer_objects <pandas.Series.infer_objects>`
       Attempt to infer better dtypes for object columns.

        **Unsupported by Intel SDC**.
   :ref:`Series.copy <pandas.Series.copy>`
       Make a copy of this object's indices and data.
   :ref:`Series.bool <pandas.Series.bool>`
       Return the bool of a single element PandasObject.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_numpy <pandas.Series.to_numpy>`
       A NumPy ndarray representing the values in this Series or Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_period <pandas.Series.to_period>`
       Convert Series from DatetimeIndex to PeriodIndex with desired frequency (inferred from index if not passed).

        **Unsupported by Intel SDC**.
   :ref:`Series.to_timestamp <pandas.Series.to_timestamp>`
       Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_list <pandas.Series.to_list>`
       Return a list of the values.

        **Unsupported by Intel SDC**.
   :ref:`Series.get_values <pandas.Series.get_values>`
       Same as values (but handles sparseness conversions); is a view.

        **Unsupported by Intel SDC**.
   :ref:`Series.__array__ <pandas.Series.__array__>`
       Return the values as a NumPy array.

        **Unsupported by Intel SDC**.

Indexing and Iteration
----------------------

   :ref:`Series.get <pandas.Series.get>`
       Get item from object for given key (ex: DataFrame column).

        **Unsupported by Intel SDC**.
   :ref:`Series.at <pandas.Series.at>`
       Access a single value for a row/column label pair.
   :ref:`Series.iat <pandas.Series.iat>`
       Access a single value for a row/column pair by integer position.
   :ref:`Series.loc <pandas.Series.loc>`
       Access a group of rows and columns by label(s) or a boolean array.
   :ref:`Series.iloc <pandas.Series.iloc>`
       Purely integer-location based indexing for selection by position.
   :ref:`Series.__iter__ <pandas.Series.__iter__>`
       Return an iterator of the values.

        **Unsupported by Intel SDC**.
   :ref:`Series.items <pandas.Series.items>`
       Lazily iterate over (index, value) tuples.

        **Unsupported by Intel SDC**.
   :ref:`Series.iteritems <pandas.Series.iteritems>`
       Lazily iterate over (index, value) tuples.

        **Unsupported by Intel SDC**.
   :ref:`Series.keys <pandas.Series.keys>`
       Return alias for index.

        **Unsupported by Intel SDC**.
   :ref:`Series.pop <pandas.Series.pop>`
       Return item and drop from frame. Raise KeyError if not found.

        **Unsupported by Intel SDC**.
   :ref:`Series.item <pandas.Series.item>`
       Return the first element of the underlying data as a python scalar.

        **Unsupported by Intel SDC**.
   :ref:`Series.xs <pandas.Series.xs>`
       Return cross-section from the Series/DataFrame.

        **Unsupported by Intel SDC**.

For more information on ``.at``, ``.iat``, ``.loc``, and
``.iloc``,  see the :ref:`indexing documentation <indexing>`.

Binary Operator Functions
-------------------------

   :ref:`Series.add <pandas.Series.add>`
       Return Addition of series and other, element-wise (binary operator `add`).
   :ref:`Series.sub <pandas.Series.sub>`
       Return Subtraction of series and other, element-wise (binary operator `sub`).
   :ref:`Series.mul <pandas.Series.mul>`
       Return Multiplication of series and other, element-wise (binary operator `mul`).
   :ref:`Series.div <pandas.Series.div>`
       Return Floating division of series and other, element-wise (binary operator `truediv`).
   :ref:`Series.truediv <pandas.Series.truediv>`
       Return Floating division of series and other, element-wise (binary operator `truediv`).
   :ref:`Series.floordiv <pandas.Series.floordiv>`
       Return Integer division of series and other, element-wise (binary operator `floordiv`).
   :ref:`Series.mod <pandas.Series.mod>`
       Return Modulo of series and other, element-wise (binary operator `mod`).
   :ref:`Series.pow <pandas.Series.pow>`
       Return Exponential power of series and other, element-wise (binary operator `pow`).
   :ref:`Series.radd <pandas.Series.radd>`
       Return Addition of series and other, element-wise (binary operator `radd`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rsub <pandas.Series.rsub>`
       Return Subtraction of series and other, element-wise (binary operator `rsub`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rmul <pandas.Series.rmul>`
       Return Multiplication of series and other, element-wise (binary operator `rmul`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rdiv <pandas.Series.rdiv>`
       Return Floating division of series and other, element-wise (binary operator `rtruediv`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rtruediv <pandas.Series.rtruediv>`
       Return Floating division of series and other, element-wise (binary operator `rtruediv`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rfloordiv <pandas.Series.rfloordiv>`
       Return Integer division of series and other, element-wise (binary operator `rfloordiv`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rmod <pandas.Series.rmod>`
       Return Modulo of series and other, element-wise (binary operator `rmod`).

        **Unsupported by Intel SDC**.
   :ref:`Series.rpow <pandas.Series.rpow>`
       Return Exponential power of series and other, element-wise (binary operator `rpow`).

        **Unsupported by Intel SDC**.
   :ref:`Series.combine <pandas.Series.combine>`
       Combine the Series with a Series or scalar according to `func`.

        **Unsupported by Intel SDC**.
   :ref:`Series.combine_first <pandas.Series.combine_first>`
       Combine Series values, choosing the calling Series's values first.

        **Unsupported by Intel SDC**.
   :ref:`Series.round <pandas.Series.round>`
       Round each value in a Series to the given number of decimals.

        **Unsupported by Intel SDC**.
   :ref:`Series.lt <pandas.Series.lt>`
       Return Less than of series and other, element-wise (binary operator `lt`).
   :ref:`Series.gt <pandas.Series.gt>`
       Return Greater than of series and other, element-wise (binary operator `gt`).
   :ref:`Series.le <pandas.Series.le>`
       Return Less than or equal to of series and other, element-wise (binary operator `le`).
   :ref:`Series.ge <pandas.Series.ge>`
       Return Greater than or equal to of series and other, element-wise (binary operator `ge`).
   :ref:`Series.ne <pandas.Series.ne>`
       Return Not equal to of series and other, element-wise (binary operator `ne`).
   :ref:`Series.eq <pandas.Series.eq>`
       Return Equal to of series and other, element-wise (binary operator `eq`).
   :ref:`Series.product <pandas.Series.product>`
       Return the product of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.dot <pandas.Series.dot>`
       Compute the dot product between the Series and the columns of other.

        **Unsupported by Intel SDC**.

User-Defined Functions, GroupBy, Window
---------------------------------------

   :ref:`Series.apply <pandas.Series.apply>`
       Invoke function on values of Series.
   :ref:`Series.agg <pandas.Series.agg>`
       Aggregate using one or more operations over the specified axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.aggregate <pandas.Series.aggregate>`
       Aggregate using one or more operations over the specified axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.transform <pandas.Series.transform>`
       Call ``func`` on self producing a Series with transformed values and that has the same axis length as self.

        **Unsupported by Intel SDC**.
   :ref:`Series.map <pandas.Series.map>`
       Map values of Series according to input correspondence.
   :ref:`Series.groupby <pandas.Series.groupby>`
       Group DataFrame or Series using a mapper or by a Series of columns.

        **Unsupported by Intel SDC**.
   :ref:`Series.rolling <pandas.Series.rolling>`
       Provide rolling window calculations.
   :ref:`Series.expanding <pandas.Series.expanding>`
       Provide expanding transformations.

        **Unsupported by Intel SDC**.
   :ref:`Series.ewm <pandas.Series.ewm>`
       Provide exponential weighted functions.

        **Unsupported by Intel SDC**.
   :ref:`Series.pipe <pandas.Series.pipe>`
       Apply func(self, \\*args, \\*\\*kwargs).

        **Unsupported by Intel SDC**.

.. _api_ref.pandas.series.stats:

Computations, Descriptive Statistics
------------------------------------

   :ref:`Series.abs <pandas.Series.abs>`
       Return a Series/DataFrame with absolute numeric value of each element.
   :ref:`Series.all <pandas.Series.all>`
       Return whether all elements are True, potentially over an axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.any <pandas.Series.any>`
       Return whether any element is True, potentially over an axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.autocorr <pandas.Series.autocorr>`
       Compute the lag-N autocorrelation.

        **Unsupported by Intel SDC**.
   :ref:`Series.between <pandas.Series.between>`
       Return boolean Series equivalent to left <= series <= right.

        **Unsupported by Intel SDC**.
   :ref:`Series.clip <pandas.Series.clip>`
       Trim values at input threshold(s).

        **Unsupported by Intel SDC**.
   :ref:`Series.corr <pandas.Series.corr>`
       Compute correlation with `other` Series, excluding missing values.
   :ref:`Series.count <pandas.Series.count>`
       Return number of non-NA/null observations in the Series.
   :ref:`Series.cov <pandas.Series.cov>`
       Compute covariance with Series, excluding missing values.

        **Unsupported by Intel SDC**.
   :ref:`Series.cummax <pandas.Series.cummax>`
       Return cumulative maximum over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.cummin <pandas.Series.cummin>`
       Return cumulative minimum over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.cumprod <pandas.Series.cumprod>`
       Return cumulative product over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.cumsum <pandas.Series.cumsum>`
       Return cumulative sum over a DataFrame or Series axis.
   :ref:`Series.describe <pandas.Series.describe>`
       Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding ``NaN`` values.

        **Unsupported by Intel SDC**.
   :ref:`Series.diff <pandas.Series.diff>`
       First discrete difference of element.

        **Unsupported by Intel SDC**.
   :ref:`Series.factorize <pandas.Series.factorize>`
       Encode the object as an enumerated type or categorical variable.

        **Unsupported by Intel SDC**.
   :ref:`Series.kurt <pandas.Series.kurt>`
       Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`Series.mad <pandas.Series.mad>`
       Return the mean absolute deviation of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.max <pandas.Series.max>`
       Return the maximum of the values for the requested axis.
   :ref:`Series.mean <pandas.Series.mean>`
       Return the mean of the values for the requested axis.
   :ref:`Series.median <pandas.Series.median>`
       Return the median of the values for the requested axis.
   :ref:`Series.min <pandas.Series.min>`
       Return the minimum of the values for the requested axis.
   :ref:`Series.mode <pandas.Series.mode>`
       Return the mode(s) of the dataset.

        **Unsupported by Intel SDC**.
   :ref:`Series.nlargest <pandas.Series.nlargest>`
       Return the largest `n` elements.
   :ref:`Series.nsmallest <pandas.Series.nsmallest>`
       Return the smallest `n` elements.
   :ref:`Series.pct_change <pandas.Series.pct_change>`
       Percentage change between the current and a prior element.

        **Unsupported by Intel SDC**.
   :ref:`Series.prod <pandas.Series.prod>`
       Return the product of the values for the requested axis.
   :ref:`Series.quantile <pandas.Series.quantile>`
       Return value at the given quantile.
   :ref:`Series.rank <pandas.Series.rank>`
       Compute numerical data ranks (1 through n) along axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.sem <pandas.Series.sem>`
       Return unbiased standard error of the mean over requested axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.skew <pandas.Series.skew>`
       Return unbiased skew over requested axis Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`Series.std <pandas.Series.std>`
       Return sample standard deviation over requested axis.
   :ref:`Series.sum <pandas.Series.sum>`
       Return the sum of the values for the requested axis.
   :ref:`Series.var <pandas.Series.var>`
       Return unbiased variance over requested axis.
   :ref:`Series.kurtosis <pandas.Series.kurtosis>`
       Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`Series.unique <pandas.Series.unique>`
       Return unique values of Series object.
   :ref:`Series.nunique <pandas.Series.nunique>`
       Return number of unique elements in the object.
   :ref:`Series.is_unique <pandas.Series.is_unique>`
       Return boolean if values in the object are unique.

        **Unsupported by Intel SDC**.
   :ref:`Series.is_monotonic <pandas.Series.is_monotonic>`
       Return boolean if values in the object are monotonic_increasing.

        **Unsupported by Intel SDC**.
   :ref:`Series.is_monotonic_increasing <pandas.Series.is_monotonic_increasing>`
       Return boolean if values in the object are monotonic_increasing.

        **Unsupported by Intel SDC**.
   :ref:`Series.is_monotonic_decreasing <pandas.Series.is_monotonic_decreasing>`
       Return boolean if values in the object are monotonic_decreasing.

        **Unsupported by Intel SDC**.
   :ref:`Series.value_counts <pandas.Series.value_counts>`
       Return a Series containing counts of unique values.
   :ref:`Series.compound <pandas.Series.compound>`
       Return the compound percentage of the values for the requested axis.

        **Unsupported by Intel SDC**.

Re-Indexing, Selection, Label Manipulation
------------------------------------------

   :ref:`Series.align <pandas.Series.align>`
       Align two objects on their axes with the specified join method for each axis Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.drop <pandas.Series.drop>`
       Return Series with specified index labels removed.

        **Unsupported by Intel SDC**.
   :ref:`Series.droplevel <pandas.Series.droplevel>`
       Return DataFrame with requested index / column level(s) removed.

        **Unsupported by Intel SDC**.
   :ref:`Series.drop_duplicates <pandas.Series.drop_duplicates>`
       Return Series with duplicate values removed.

        **Unsupported by Intel SDC**.
   :ref:`Series.duplicated <pandas.Series.duplicated>`
       Indicate duplicate Series values.

        **Unsupported by Intel SDC**.
   :ref:`Series.equals <pandas.Series.equals>`
       Test whether two objects contain the same elements.

        **Unsupported by Intel SDC**.
   :ref:`Series.first <pandas.Series.first>`
       Convenience method for subsetting initial periods of time series data based on a date offset.

        **Unsupported by Intel SDC**.
   :ref:`Series.head <pandas.Series.head>`
       Return the first `n` rows.
   :ref:`Series.idxmax <pandas.Series.idxmax>`
       Return the row label of the maximum value.
   :ref:`Series.idxmin <pandas.Series.idxmin>`
       Return the row label of the minimum value.
   :ref:`Series.isin <pandas.Series.isin>`
       Check whether `values` are contained in Series.
   :ref:`Series.last <pandas.Series.last>`
       Convenience method for subsetting final periods of time series data based on a date offset.

        **Unsupported by Intel SDC**.
   :ref:`Series.reindex <pandas.Series.reindex>`
       Conform Series to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. A new object is produced unless the new index is equivalent to the current one and ``copy=False``.

        **Unsupported by Intel SDC**.
   :ref:`Series.reindex_like <pandas.Series.reindex_like>`
       Return an object with matching indices as other object.

        **Unsupported by Intel SDC**.
   :ref:`Series.rename <pandas.Series.rename>`
       Alter Series index labels or name.
   :ref:`Series.rename_axis <pandas.Series.rename_axis>`
       Set the name of the axis for the index or columns.

        **Unsupported by Intel SDC**.
   :ref:`Series.reset_index <pandas.Series.reset_index>`
       Generate a new DataFrame or Series with the index reset.

        **Unsupported by Intel SDC**.
   :ref:`Series.sample <pandas.Series.sample>`
       Return a random sample of items from an axis of object.

        **Unsupported by Intel SDC**.
   :ref:`Series.set_axis <pandas.Series.set_axis>`
       Assign desired index to given axis.

        **Unsupported by Intel SDC**.
   :ref:`Series.take <pandas.Series.take>`
       Return the elements in the given *positional* indices along an axis.
   :ref:`Series.tail <pandas.Series.tail>`
       Return the last `n` rows.

        **Unsupported by Intel SDC**.
   :ref:`Series.truncate <pandas.Series.truncate>`
       Truncate a Series or DataFrame before and after some index value.

        **Unsupported by Intel SDC**.
   :ref:`Series.where <pandas.Series.where>`
       Replace values where the condition is False.

        **Unsupported by Intel SDC**.
   :ref:`Series.mask <pandas.Series.mask>`
       Replace values where the condition is True.

        **Unsupported by Intel SDC**.
   :ref:`Series.add_prefix <pandas.Series.add_prefix>`
       Prefix labels with string `prefix`.

        **Unsupported by Intel SDC**.
   :ref:`Series.add_suffix <pandas.Series.add_suffix>`
       Suffix labels with string `suffix`.

        **Unsupported by Intel SDC**.
   :ref:`Series.filter <pandas.Series.filter>`
       Subset rows or columns of dataframe according to labels in the specified index.

        **Unsupported by Intel SDC**.

Missing Data Handling
---------------------

   :ref:`Series.isna <pandas.Series.isna>`
       Detect missing values.
   :ref:`Series.notna <pandas.Series.notna>`
       Detect existing (non-missing) values.
   :ref:`Series.dropna <pandas.Series.dropna>`
       Return a new Series with missing values removed.
   :ref:`Series.fillna <pandas.Series.fillna>`
       Fill NA/NaN values using the specified method.
   :ref:`Series.interpolate <pandas.Series.interpolate>`
       Interpolate values according to different methods.

        **Unsupported by Intel SDC**.

Re-Shaping, Sorting
-------------------

   :ref:`Series.argsort <pandas.Series.argsort>`
       Override ndarray.argsort. Argsorts the value, omitting NA/null values, and places the result in the same locations as the non-NA values.
   :ref:`Series.argmin <pandas.Series.argmin>`
       Return the row label of the minimum value.

        **Unsupported by Intel SDC**.
   :ref:`Series.argmax <pandas.Series.argmax>`
       Return the row label of the maximum value.

        **Unsupported by Intel SDC**.
   :ref:`Series.reorder_levels <pandas.Series.reorder_levels>`
       Rearrange index levels using input order.

        **Unsupported by Intel SDC**.
   :ref:`Series.sort_values <pandas.Series.sort_values>`
       Sort by the values.
   :ref:`Series.sort_index <pandas.Series.sort_index>`
       Sort Series by index labels.

        **Unsupported by Intel SDC**.
   :ref:`Series.swaplevel <pandas.Series.swaplevel>`
       Swap levels i and j in a MultiIndex.

        **Unsupported by Intel SDC**.
   :ref:`Series.unstack <pandas.Series.unstack>`
       Unstack, a.k.a. pivot, Series with MultiIndex to produce DataFrame. The level involved will automatically get sorted.

        **Unsupported by Intel SDC**.
   :ref:`Series.explode <pandas.Series.explode>`
       Transform each element of a list-like to a row, replicating the index values.

        **Unsupported by Intel SDC**.
   :ref:`Series.searchsorted <pandas.Series.searchsorted>`
       Find indices where elements should be inserted to maintain order.

        **Unsupported by Intel SDC**.
   :ref:`Series.ravel <pandas.Series.ravel>`
       Return the flattened underlying data as an ndarray.

        **Unsupported by Intel SDC**.
   :ref:`Series.repeat <pandas.Series.repeat>`
       Repeat elements of a Series.

        **Unsupported by Intel SDC**.
   :ref:`Series.squeeze <pandas.Series.squeeze>`
       Squeeze 1 dimensional axis objects into scalars.

        **Unsupported by Intel SDC**.
   :ref:`Series.view <pandas.Series.view>`
       Create a new view of the Series.

        **Unsupported by Intel SDC**.

Combining, Joining, Merging
-----------------------------

   :ref:`Series.append <pandas.Series.append>`
       Concatenate two or more Series.

        **Unsupported by Intel SDC**.
   :ref:`Series.replace <pandas.Series.replace>`
       Replace values given in `to_replace` with `value`.

        **Unsupported by Intel SDC**.
   :ref:`Series.update <pandas.Series.update>`
       Modify Series in place using non-NA values from passed Series. Aligns on index.

        **Unsupported by Intel SDC**.

Time Series
-----------

   :ref:`Series.asfreq <pandas.Series.asfreq>`
       Convert TimeSeries to specified frequency.

        **Unsupported by Intel SDC**.
   :ref:`Series.asof <pandas.Series.asof>`
       Return the last row(s) without any NaNs before `where`.

        **Unsupported by Intel SDC**.
   :ref:`Series.shift <pandas.Series.shift>`
       Shift index by desired number of periods with an optional time `freq`.
   :ref:`Series.first_valid_index <pandas.Series.first_valid_index>`
       Return index for first non-NA/null value.

        **Unsupported by Intel SDC**.
   :ref:`Series.last_valid_index <pandas.Series.last_valid_index>`
       Return index for last non-NA/null value.

        **Unsupported by Intel SDC**.
   :ref:`Series.resample <pandas.Series.resample>`
       Resample time-series data.

        **Unsupported by Intel SDC**.
   :ref:`Series.tz_convert <pandas.Series.tz_convert>`
       Convert tz-aware axis to target time zone.

        **Unsupported by Intel SDC**.
   :ref:`Series.tz_localize <pandas.Series.tz_localize>`
       Localize tz-naive index of a Series or DataFrame to target time zone.

        **Unsupported by Intel SDC**.
   :ref:`Series.at_time <pandas.Series.at_time>`
       Select values at particular time of day (e.g. 9:30AM).

        **Unsupported by Intel SDC**.
   :ref:`Series.between_time <pandas.Series.between_time>`
       Select values between particular times of the day (e.g., 9:00-9:30 AM).

        **Unsupported by Intel SDC**.
   :ref:`Series.tshift <pandas.Series.tshift>`
       Shift the time index, using the index's frequency if available.

        **Unsupported by Intel SDC**.
   :ref:`Series.slice_shift <pandas.Series.slice_shift>`
       Equivalent to `shift` without copying data. The shifted data will not include the dropped periods and the shifted axis will be smaller than the original.

        **Unsupported by Intel SDC**.

Accessors
---------

Pandas provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`Series` that only apply
to specific data types.

=========================== =================================
Data Type                   Accessor
=========================== =================================
Datetime, Timedelta, Period :ref:`dt <api_ref.pandas.series.dt>`
String                      :ref:`str <api_ref.pandas.series.str>`
Categorical                 :ref:`cat <api_ref.pandas.series.cat>`
Sparse                      :ref:`sparse <api_ref.pandas.series.sparse>`
=========================== =================================

.. _api_ref.pandas.series.dt:

Datetimelike properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

Datetime properties
^^^^^^^^^^^^^^^^^^^

   :ref:`Series.dt.date <pandas.Series.dt.date>`
       Returns numpy array of python datetime.date objects (namely, the date part of Timestamps without timezone information).

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.time <pandas.Series.dt.time>`
       Returns numpy array of datetime.time. The time part of the Timestamps.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.timetz <pandas.Series.dt.timetz>`
       Returns numpy array of datetime.time also containing timezone information. The time part of the Timestamps.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.year <pandas.Series.dt.year>`
       The year of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.month <pandas.Series.dt.month>`
       The month as January=1, December=12.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.day <pandas.Series.dt.day>`
       The days of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.hour <pandas.Series.dt.hour>`
       The hours of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.minute <pandas.Series.dt.minute>`
       The minutes of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.second <pandas.Series.dt.second>`
       The seconds of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.microsecond <pandas.Series.dt.microsecond>`
       The microseconds of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.nanosecond <pandas.Series.dt.nanosecond>`
       The nanoseconds of the datetime.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.week <pandas.Series.dt.week>`
       The week ordinal of the year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.weekofyear <pandas.Series.dt.weekofyear>`
       The week ordinal of the year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.dayofweek <pandas.Series.dt.dayofweek>`
       The day of the week with Monday=0, Sunday=6.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.weekday <pandas.Series.dt.weekday>`
       The day of the week with Monday=0, Sunday=6.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.dayofyear <pandas.Series.dt.dayofyear>`
       The ordinal day of the year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.quarter <pandas.Series.dt.quarter>`
       The quarter of the date.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_month_start <pandas.Series.dt.is_month_start>`
       Indicates whether the date is the first day of the month.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_month_end <pandas.Series.dt.is_month_end>`
       Indicates whether the date is the last day of the month.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_quarter_start <pandas.Series.dt.is_quarter_start>`
       Indicator for whether the date is the first day of a quarter.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_quarter_end <pandas.Series.dt.is_quarter_end>`
       Indicator for whether the date is the last day of a quarter.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_year_start <pandas.Series.dt.is_year_start>`
       Indicate whether the date is the first day of a year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_year_end <pandas.Series.dt.is_year_end>`
       Indicate whether the date is the last day of the year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.is_leap_year <pandas.Series.dt.is_leap_year>`
       Boolean indicator if the date belongs to a leap year.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.daysinmonth <pandas.Series.dt.daysinmonth>`
       The number of days in the month.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.days_in_month <pandas.Series.dt.days_in_month>`
       The number of days in the month.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.tz <pandas.Series.dt.tz>`
       Return timezone, if any.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.freq <pandas.Series.dt.freq>`
       **Unsupported by Intel SDC**.

Datetime methods
^^^^^^^^^^^^^^^^

   :ref:`Series.dt.to_period <pandas.Series.dt.to_period>`
       Cast to PeriodArray/Index at a particular frequency.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.to_pydatetime <pandas.Series.dt.to_pydatetime>`
       Return the data as an array of native Python datetime objects.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.tz_localize <pandas.Series.dt.tz_localize>`
       Localize tz-naive Datetime Array/Index to tz-aware Datetime Array/Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.tz_convert <pandas.Series.dt.tz_convert>`
       Convert tz-aware Datetime Array/Index from one time zone to another.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.normalize <pandas.Series.dt.normalize>`
       Convert times to midnight.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.strftime <pandas.Series.dt.strftime>`
       Convert to Index using specified date_format.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.round <pandas.Series.dt.round>`
       Perform round operation on the data to the specified `freq`.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.floor <pandas.Series.dt.floor>`
       Perform floor operation on the data to the specified `freq`.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.ceil <pandas.Series.dt.ceil>`
       Perform ceil operation on the data to the specified `freq`.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.month_name <pandas.Series.dt.month_name>`
       Return the month names of the DateTimeIndex with specified locale.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.day_name <pandas.Series.dt.day_name>`
       Return the day names of the DateTimeIndex with specified locale.

        **Unsupported by Intel SDC**.

Period properties
^^^^^^^^^^^^^^^^^

   :ref:`Series.dt.qyear <pandas.Series.dt.qyear>`
       **Unsupported by Intel SDC**.
   :ref:`Series.dt.start_time <pandas.Series.dt.start_time>`
       **Unsupported by Intel SDC**.
   :ref:`Series.dt.end_time <pandas.Series.dt.end_time>`
       **Unsupported by Intel SDC**.

Timedelta properties
^^^^^^^^^^^^^^^^^^^^

   :ref:`Series.dt.days <pandas.Series.dt.days>`
       Number of days for each element.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.seconds <pandas.Series.dt.seconds>`
       Number of seconds (>= 0 and less than 1 day) for each element.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.microseconds <pandas.Series.dt.microseconds>`
       Number of microseconds (>= 0 and less than 1 second) for each element.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.nanoseconds <pandas.Series.dt.nanoseconds>`
       Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.components <pandas.Series.dt.components>`
       Return a Dataframe of the components of the Timedeltas.

        **Unsupported by Intel SDC**.

Timedelta methods
^^^^^^^^^^^^^^^^^

   :ref:`Series.dt.to_pytimedelta <pandas.Series.dt.to_pytimedelta>`
       Return an array of native `datetime.timedelta` objects.

        **Unsupported by Intel SDC**.
   :ref:`Series.dt.total_seconds <pandas.Series.dt.total_seconds>`
       Return total duration of each element expressed in seconds.

        **Unsupported by Intel SDC**.

.. _api_ref.pandas.series.str:

String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

   :ref:`Series.str.capitalize <pandas.Series.str.capitalize>`
       Convert strings in the Series/Index to be capitalized.
   :ref:`Series.str.casefold <pandas.Series.str.casefold>`
       Convert strings in the Series/Index to be casefolded.
   :ref:`Series.str.cat <pandas.Series.str.cat>`
       Concatenate strings in the Series/Index with given separator.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.center <pandas.Series.str.center>`
       Filling left and right side of strings in the Series/Index with an additional character. Equivalent to :meth:`str.center`.
   :ref:`Series.str.contains <pandas.Series.str.contains>`
       Test if pattern or regex is contained within a string of a Series or Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.count <pandas.Series.str.count>`
       Count occurrences of pattern in each string of the Series/Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.decode <pandas.Series.str.decode>`
       Decode character string in the Series/Index using indicated encoding. Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in python3.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.encode <pandas.Series.str.encode>`
       Encode character string in the Series/Index using indicated encoding. Equivalent to :meth:`str.encode`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.endswith <pandas.Series.str.endswith>`
       Test if the end of each string element matches a pattern.
   :ref:`Series.str.extract <pandas.Series.str.extract>`
       Extract capture groups in the regex `pat` as columns in a DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.extractall <pandas.Series.str.extractall>`
       For each subject string in the Series, extract groups from all matches of regular expression pat. When each subject string in the Series has exactly one match, extractall(pat).xs(0, level='match') is the same as extract(pat).

        **Unsupported by Intel SDC**.
   :ref:`Series.str.find <pandas.Series.str.find>`
       Return lowest indexes in each strings in the Series/Index where the substring is fully contained between [start:end]. Return -1 on failure. Equivalent to standard :meth:`str.find`.
   :ref:`Series.str.findall <pandas.Series.str.findall>`
       Find all occurrences of pattern or regular expression in the Series/Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.get <pandas.Series.str.get>`
       Extract element from each component at specified position.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.index <pandas.Series.str.index>`
       Return lowest indexes in each strings where the substring is fully contained between [start:end]. This is the same as ``str.find`` except instead of returning -1, it raises a ValueError when the substring is not found. Equivalent to standard ``str.index``.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.join <pandas.Series.str.join>`
       Join lists contained as elements in the Series/Index with passed delimiter.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.len <pandas.Series.str.len>`
       Compute the length of each element in the Series/Index. The element may be a sequence (such as a string, tuple or list) or a collection (such as a dictionary).
   :ref:`Series.str.ljust <pandas.Series.str.ljust>`
       Filling right side of strings in the Series/Index with an additional character. Equivalent to :meth:`str.ljust`.
   :ref:`Series.str.lower <pandas.Series.str.lower>`
       Convert strings in the Series/Index to lowercase.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.lstrip <pandas.Series.str.lstrip>`
       Remove leading and trailing characters.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.match <pandas.Series.str.match>`
       Determine if each string matches a regular expression.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.normalize <pandas.Series.str.normalize>`
       Return the Unicode normal form for the strings in the Series/Index. For more information on the forms, see the :func:`unicodedata.normalize`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.pad <pandas.Series.str.pad>`
       Pad strings in the Series/Index up to width.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.partition <pandas.Series.str.partition>`
       Split the string at the first occurrence of `sep`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.repeat <pandas.Series.str.repeat>`
       Duplicate each string in the Series or Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.replace <pandas.Series.str.replace>`
       Replace occurrences of pattern/regex in the Series/Index with some other string. Equivalent to :meth:`str.replace` or :func:`re.sub`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.rfind <pandas.Series.str.rfind>`
       Return highest indexes in each strings in the Series/Index where the substring is fully contained between [start:end]. Return -1 on failure. Equivalent to standard :meth:`str.rfind`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.rindex <pandas.Series.str.rindex>`
       Return highest indexes in each strings where the substring is fully contained between [start:end]. This is the same as ``str.rfind`` except instead of returning -1, it raises a ValueError when the substring is not found. Equivalent to standard ``str.rindex``.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.rjust <pandas.Series.str.rjust>`
       Filling left side of strings in the Series/Index with an additional character. Equivalent to :meth:`str.rjust`.
   :ref:`Series.str.rpartition <pandas.Series.str.rpartition>`
       Split the string at the last occurrence of `sep`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.rstrip <pandas.Series.str.rstrip>`
       Remove leading and trailing characters.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.slice <pandas.Series.str.slice>`
       Slice substrings from each element in the Series or Index.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.slice_replace <pandas.Series.str.slice_replace>`
       Replace a positional slice of a string with another value.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.split <pandas.Series.str.split>`
       Split strings around given separator/delimiter.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.rsplit <pandas.Series.str.rsplit>`
       Split strings around given separator/delimiter.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.startswith <pandas.Series.str.startswith>`
       Test if the start of each string element matches a pattern.
   :ref:`Series.str.strip <pandas.Series.str.strip>`
       Remove leading and trailing characters.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.swapcase <pandas.Series.str.swapcase>`
       Convert strings in the Series/Index to be swapcased.
   :ref:`Series.str.title <pandas.Series.str.title>`
       Convert strings in the Series/Index to titlecase.
   :ref:`Series.str.translate <pandas.Series.str.translate>`
       Map all characters in the string through the given mapping table. Equivalent to standard :meth:`str.translate`.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.upper <pandas.Series.str.upper>`
       Convert strings in the Series/Index to uppercase.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.wrap <pandas.Series.str.wrap>`
       Wrap long strings in the Series/Index to be formatted in paragraphs with length less than a given width.

        **Unsupported by Intel SDC**.
   :ref:`Series.str.zfill <pandas.Series.str.zfill>`
       Pad strings in the Series/Index by prepending '0' characters.
   :ref:`Series.str.isalnum <pandas.Series.str.isalnum>`
       Check whether all characters in each string are alphanumeric.
   :ref:`Series.str.isalpha <pandas.Series.str.isalpha>`
       Check whether all characters in each string are alphabetic.
   :ref:`Series.str.isdigit <pandas.Series.str.isdigit>`
       Check whether all characters in each string are digits.
   :ref:`Series.str.isspace <pandas.Series.str.isspace>`
       Check whether all characters in each string are whitespace.
   :ref:`Series.str.islower <pandas.Series.str.islower>`
       Check whether all characters in each string are lowercase.
   :ref:`Series.str.isupper <pandas.Series.str.isupper>`
       Check whether all characters in each string are uppercase.
   :ref:`Series.str.istitle <pandas.Series.str.istitle>`
       Check whether all characters in each string are titlecase.
   :ref:`Series.str.isnumeric <pandas.Series.str.isnumeric>`
       Check whether all characters in each string are numeric.
   :ref:`Series.str.isdecimal <pandas.Series.str.isdecimal>`
       Check whether all characters in each string are decimal.
   :ref:`Series.str.get_dummies <pandas.Series.str.get_dummies>`
       Split each string in the Series by sep and return a DataFrame of dummy/indicator variables.

        **Unsupported by Intel SDC**.

.. _api_ref.pandas.series.cat:

Categorical Accessor
~~~~~~~~~~~~~~~~~~~~

Categorical-dtype specific methods and attributes are available under
the ``Series.cat`` accessor.

   :ref:`Series.cat.categories <pandas.Series.cat.categories>`
       The categories of this categorical.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.ordered <pandas.Series.cat.ordered>`
       Whether the categories have an ordered relationship.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.codes <pandas.Series.cat.codes>`
       Return Series of codes as well as the index.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.rename_categories <pandas.Series.cat.rename_categories>`
       Rename categories.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.reorder_categories <pandas.Series.cat.reorder_categories>`
       Reorder categories as specified in new_categories.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.add_categories <pandas.Series.cat.add_categories>`
       Add new categories.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.remove_categories <pandas.Series.cat.remove_categories>`
       Remove the specified categories.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.remove_unused_categories <pandas.Series.cat.remove_unused_categories>`
       Remove categories which are not used.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.set_categories <pandas.Series.cat.set_categories>`
       Set the categories to the specified new_categories.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.as_ordered <pandas.Series.cat.as_ordered>`
       Set the Categorical to be ordered.

        **Unsupported by Intel SDC**.
   :ref:`Series.cat.as_unordered <pandas.Series.cat.as_unordered>`
       Set the Categorical to be unordered.

        **Unsupported by Intel SDC**.


.. _api_ref.pandas.series.sparse:

Sparse Accessor
~~~~~~~~~~~~~~~

Sparse-dtype specific methods and attributes are provided under the
``Series.sparse`` accessor.

   :ref:`Series.sparse.npoints <pandas.Series.sparse.npoints>`
       The number of non- ``fill_value`` points.

        **Unsupported by Intel SDC**.
   :ref:`Series.sparse.density <pandas.Series.sparse.density>`
       The percent of non- ``fill_value`` points, as decimal.

        **Unsupported by Intel SDC**.
   :ref:`Series.sparse.fill_value <pandas.Series.sparse.fill_value>`
       Elements in `data` that are `fill_value` are not stored.

        **Unsupported by Intel SDC**.
   :ref:`Series.sparse.sp_values <pandas.Series.sparse.sp_values>`
       An ndarray containing the non- ``fill_value`` values.

        **Unsupported by Intel SDC**.
   :ref:`Series.sparse.from_coo <pandas.Series.sparse.from_coo>`
       Create a SparseSeries from a scipy.sparse.coo_matrix.

        **Unsupported by Intel SDC**.
   :ref:`Series.sparse.to_coo <pandas.Series.sparse.to_coo>`
       Create a scipy.sparse.coo_matrix from a SparseSeries with MultiIndex.

        **Unsupported by Intel SDC**.

.. _api_ref.pandas.series.metadata:

Plotting
--------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

   :ref:`Series.plot <pandas.Series.plot>`
       Make plots of Series or DataFrame using the backend specified by the option ``plotting.backend``. By default, matplotlib is used.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.area <pandas.Series.plot.area>`
       Draw a stacked area plot.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.bar <pandas.Series.plot.bar>`
       Vertical bar plot.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.barh <pandas.Series.plot.barh>`
       Make a horizontal bar plot.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.box <pandas.Series.plot.box>`
       Make a box plot of the DataFrame columns.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.density <pandas.Series.plot.density>`
       Generate Kernel Density Estimate plot using Gaussian kernels.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.hist <pandas.Series.plot.hist>`
       Draw one histogram of the DataFrame's columns.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.kde <pandas.Series.plot.kde>`
       Generate Kernel Density Estimate plot using Gaussian kernels.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.line <pandas.Series.plot.line>`
       Plot Series or DataFrame as lines.

        **Unsupported by Intel SDC**.
   :ref:`Series.plot.pie <pandas.Series.plot.pie>`
       Generate a pie plot.

        **Unsupported by Intel SDC**.
   :ref:`Series.hist <pandas.Series.hist>`
       Draw histogram of the input series using matplotlib.

        **Unsupported by Intel SDC**.

Serialization, Input-Output, Conversion
---------------------------------------

   :ref:`Series.to_pickle <pandas.Series.to_pickle>`
       Pickle (serialize) object to file.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_csv <pandas.Series.to_csv>`
       Write object to a comma-separated values (csv) file.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_dict <pandas.Series.to_dict>`
       Convert Series to {label -> value} dict or dict-like object.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_excel <pandas.Series.to_excel>`
       Write object to an Excel sheet.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_frame <pandas.Series.to_frame>`
       Convert Series to DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_xarray <pandas.Series.to_xarray>`
       Return an xarray object from the pandas object.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_hdf <pandas.Series.to_hdf>`
       Write the contained data to an HDF5 file using HDFStore.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_sql <pandas.Series.to_sql>`
       Write records stored in a DataFrame to a SQL database.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_msgpack <pandas.Series.to_msgpack>`
       Serialize object to input file path using msgpack format.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_json <pandas.Series.to_json>`
       Convert the object to a JSON string.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_string <pandas.Series.to_string>`
       Render a string representation of the Series.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_clipboard <pandas.Series.to_clipboard>`
       Copy object to the system clipboard.

        **Unsupported by Intel SDC**.
   :ref:`Series.to_latex <pandas.Series.to_latex>`
       Render an object to a LaTeX tabular environment table.

        **Unsupported by Intel SDC**.
