.. _api_ref.pandas.dataframe:
.. include:: ./../ext_links.txt

DataFrame
=========
.. currentmodule:: pandas

This is the main `Pandas*`_ data structure representing a table of rows and columns.

DataFrame is a two-dimensional structure with labeled axes. It can be thought of as a dictionary-like
container for :class:`Series <pandas.Series>`

Constructor
-----------

   :ref:`DataFrame <pandas.DataFrame>`
       Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.

        **Unsupported by Intel SDC**.

Attributes/Operators
--------------------

   :ref:`DataFrame.index <pandas.DataFrame.index>`
       The index (row labels) of the DataFrame.
   :ref:`DataFrame.columns <pandas.DataFrame.columns>`
       The column labels of the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.dtypes <pandas.DataFrame.dtypes>`
       Return the dtypes in the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.select_dtypes <pandas.DataFrame.select_dtypes>`
       Return a subset of the DataFrame's columns based on the column dtypes.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.values <pandas.DataFrame.values>`
       Return a Numpy representation of the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.axes <pandas.DataFrame.axes>`
       Return a list representing the axes of the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.ndim <pandas.DataFrame.ndim>`
       Return an int representing the number of axes / array dimensions.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.size <pandas.DataFrame.size>`
       Return an int representing the number of elements in this object.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.shape <pandas.DataFrame.shape>`
       Return a tuple representing the dimensionality of the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.memory_usage <pandas.DataFrame.memory_usage>`
       Return the memory usage of each column in bytes.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.empty <pandas.DataFrame.empty>`
       Indicator whether DataFrame is empty.

        **Unsupported by Intel SDC**.

Type Conversions
----------------

   :ref:`DataFrame.astype <pandas.DataFrame.astype>`
       Cast a pandas object to a specified dtype ``dtype``.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.infer_objects <pandas.DataFrame.infer_objects>`
       Attempt to infer better dtypes for object columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.copy <pandas.DataFrame.copy>`
       Make a copy of this object's indices and data.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.isna <pandas.DataFrame.isna>`
       Detect missing values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.notna <pandas.DataFrame.notna>`
       Detect existing (non-missing) values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.bool <pandas.DataFrame.bool>`
       Return the bool of a single element PandasObject.

        **Unsupported by Intel SDC**.

Indexing and Iteration
----------------------

   :ref:`DataFrame.head <pandas.DataFrame.head>`
       Return the first `n` rows.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.at <pandas.DataFrame.at>`
       Access a single value for a row/column label pair.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.iat <pandas.DataFrame.iat>`
       Access a single value for a row/column pair by integer position.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.loc <pandas.DataFrame.loc>`
       Access a group of rows and columns by label(s) or a boolean array.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
       Purely integer-location based indexing for selection by position.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.insert <pandas.DataFrame.insert>`
       Insert column into DataFrame at specified location.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.__iter__ <pandas.DataFrame.__iter__>`
       Iterate over info axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.items <pandas.DataFrame.items>`
       Iterator over (column name, Series) pairs.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.iteritems <pandas.DataFrame.iteritems>`
       Iterator over (column name, Series) pairs.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.keys <pandas.DataFrame.keys>`
       Get the 'info axis' (see Indexing for more)

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.iterrows <pandas.DataFrame.iterrows>`
       Iterate over DataFrame rows as (index, Series) pairs.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.itertuples <pandas.DataFrame.itertuples>`
       Iterate over DataFrame rows as namedtuples.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.lookup <pandas.DataFrame.lookup>`
       Label-based "fancy indexing" function for DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pop <pandas.DataFrame.pop>`
       Return item and drop from frame. Raise KeyError if not found.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.tail <pandas.DataFrame.tail>`
       Return the last `n` rows.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.xs <pandas.DataFrame.xs>`
       Return cross-section from the Series/DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.get <pandas.DataFrame.get>`
       Get item from object for given key (ex: DataFrame column).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.isin <pandas.DataFrame.isin>`
       Whether each element in the DataFrame is contained in values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.where <pandas.DataFrame.where>`
       Replace values where the condition is False.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mask <pandas.DataFrame.mask>`
       Replace values where the condition is True.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.query <pandas.DataFrame.query>`
       Query the columns of a DataFrame with a boolean expression.

        **Unsupported by Intel SDC**.

For more information on ``.at``, ``.iat``, ``.loc``, and
``.iloc``,  see the :ref:`indexing documentation <indexing>`.

Binary Operator Functions
-------------------------

   :ref:`DataFrame.add <pandas.DataFrame.add>`
       Get Addition of dataframe and other, element-wise (binary operator `add`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sub <pandas.DataFrame.sub>`
       Get Subtraction of dataframe and other, element-wise (binary operator `sub`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mul <pandas.DataFrame.mul>`
       Get Multiplication of dataframe and other, element-wise (binary operator `mul`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.div <pandas.DataFrame.div>`
       Get Floating division of dataframe and other, element-wise (binary operator `truediv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.truediv <pandas.DataFrame.truediv>`
       Get Floating division of dataframe and other, element-wise (binary operator `truediv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.floordiv <pandas.DataFrame.floordiv>`
       Get Integer division of dataframe and other, element-wise (binary operator `floordiv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mod <pandas.DataFrame.mod>`
       Get Modulo of dataframe and other, element-wise (binary operator `mod`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pow <pandas.DataFrame.pow>`
       Get Exponential power of dataframe and other, element-wise (binary operator `pow`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.dot <pandas.DataFrame.dot>`
       Compute the matrix multiplication between the DataFrame and other.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.radd <pandas.DataFrame.radd>`
       Get Addition of dataframe and other, element-wise (binary operator `radd`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rsub <pandas.DataFrame.rsub>`
       Get Subtraction of dataframe and other, element-wise (binary operator `rsub`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rmul <pandas.DataFrame.rmul>`
       Get Multiplication of dataframe and other, element-wise (binary operator `rmul`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rdiv <pandas.DataFrame.rdiv>`
       Get Floating division of dataframe and other, element-wise (binary operator `rtruediv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rtruediv <pandas.DataFrame.rtruediv>`
       Get Floating division of dataframe and other, element-wise (binary operator `rtruediv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rfloordiv <pandas.DataFrame.rfloordiv>`
       Get Integer division of dataframe and other, element-wise (binary operator `rfloordiv`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rmod <pandas.DataFrame.rmod>`
       Get Modulo of dataframe and other, element-wise (binary operator `rmod`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rpow <pandas.DataFrame.rpow>`
       Get Exponential power of dataframe and other, element-wise (binary operator `rpow`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.lt <pandas.DataFrame.lt>`
       Get Less than of dataframe and other, element-wise (binary operator `lt`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.gt <pandas.DataFrame.gt>`
       Get Greater than of dataframe and other, element-wise (binary operator `gt`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.le <pandas.DataFrame.le>`
       Get Less than or equal to of dataframe and other, element-wise (binary operator `le`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.ge <pandas.DataFrame.ge>`
       Get Greater than or equal to of dataframe and other, element-wise (binary operator `ge`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.ne <pandas.DataFrame.ne>`
       Get Not equal to of dataframe and other, element-wise (binary operator `ne`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.eq <pandas.DataFrame.eq>`
       Get Equal to of dataframe and other, element-wise (binary operator `eq`).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.combine <pandas.DataFrame.combine>`
       Perform column-wise combine with another DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.combine_first <pandas.DataFrame.combine_first>`
       Update null elements with value in the same location in `other`.

        **Unsupported by Intel SDC**.

User-Defined Functions, GroupBy & Window
----------------------------------------

   :ref:`DataFrame.apply <pandas.DataFrame.apply>`
       Apply a function along an axis of the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.applymap <pandas.DataFrame.applymap>`
       Apply a function to a Dataframe elementwise.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pipe <pandas.DataFrame.pipe>`
       Apply func(self, \\*args, \\*\\*kwargs).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.agg <pandas.DataFrame.agg>`
       Aggregate using one or more operations over the specified axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.aggregate <pandas.DataFrame.aggregate>`
       Aggregate using one or more operations over the specified axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.transform <pandas.DataFrame.transform>`
       Call ``func`` on self producing a DataFrame with transformed values and that has the same axis length as self.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.groupby <pandas.DataFrame.groupby>`
       Group DataFrame or Series using a mapper or by a Series of columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
       Provide rolling window calculations.
   :ref:`DataFrame.expanding <pandas.DataFrame.expanding>`
       Provide expanding transformations.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.ewm <pandas.DataFrame.ewm>`
       Provide exponential weighted functions.

        **Unsupported by Intel SDC**.

.. _api_ref.dataframe.stats:

Computations, Descriptive Statistics
------------------------------------

   :ref:`DataFrame.abs <pandas.DataFrame.abs>`
       Return a Series/DataFrame with absolute numeric value of each element.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.all <pandas.DataFrame.all>`
       Return whether all elements are True, potentially over an axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.any <pandas.DataFrame.any>`
       Return whether any element is True, potentially over an axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.clip <pandas.DataFrame.clip>`
       Trim values at input threshold(s).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.corr <pandas.DataFrame.corr>`
       Compute pairwise correlation of columns, excluding NA/null values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.corrwith <pandas.DataFrame.corrwith>`
       Compute pairwise correlation between rows or columns of DataFrame with rows or columns of Series or DataFrame.  DataFrames are first aligned along both axes before computing the correlations.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.count <pandas.DataFrame.count>`
       Count non-NA cells for each column or row.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.cov <pandas.DataFrame.cov>`
       Compute pairwise covariance of columns, excluding NA/null values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.cummax <pandas.DataFrame.cummax>`
       Return cumulative maximum over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.cummin <pandas.DataFrame.cummin>`
       Return cumulative minimum over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.cumprod <pandas.DataFrame.cumprod>`
       Return cumulative product over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.cumsum <pandas.DataFrame.cumsum>`
       Return cumulative sum over a DataFrame or Series axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.describe <pandas.DataFrame.describe>`
       Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding ``NaN`` values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.diff <pandas.DataFrame.diff>`
       First discrete difference of element.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.eval <pandas.DataFrame.eval>`
       Evaluate a string describing operations on DataFrame columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.kurt <pandas.DataFrame.kurt>`
       Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.kurtosis <pandas.DataFrame.kurtosis>`
       Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mad <pandas.DataFrame.mad>`
       Return the mean absolute deviation of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.max <pandas.DataFrame.max>`
       Return the maximum of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mean <pandas.DataFrame.mean>`
       Return the mean of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.median <pandas.DataFrame.median>`
       Return the median of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.min <pandas.DataFrame.min>`
       Return the minimum of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.mode <pandas.DataFrame.mode>`
       Get the mode(s) of each element along the selected axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pct_change <pandas.DataFrame.pct_change>`
       Percentage change between the current and a prior element.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.prod <pandas.DataFrame.prod>`
       Return the product of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.product <pandas.DataFrame.product>`
       Return the product of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.quantile <pandas.DataFrame.quantile>`
       Return values at the given quantile over requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rank <pandas.DataFrame.rank>`
       Compute numerical data ranks (1 through n) along axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.round <pandas.DataFrame.round>`
       Round a DataFrame to a variable number of decimal places.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sem <pandas.DataFrame.sem>`
       Return unbiased standard error of the mean over requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.skew <pandas.DataFrame.skew>`
       Return unbiased skew over requested axis Normalized by N-1.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sum <pandas.DataFrame.sum>`
       Return the sum of the values for the requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.std <pandas.DataFrame.std>`
       Return sample standard deviation over requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.var <pandas.DataFrame.var>`
       Return unbiased variance over requested axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.nunique <pandas.DataFrame.nunique>`
       Count distinct observations over requested axis.

        **Unsupported by Intel SDC**.

Re-Indexing, Selection, Label Manipulation
------------------------------------------

   :ref:`DataFrame.add_prefix <pandas.DataFrame.add_prefix>`
       Prefix labels with string `prefix`.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.add_suffix <pandas.DataFrame.add_suffix>`
       Suffix labels with string `suffix`.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.align <pandas.DataFrame.align>`
       Align two objects on their axes with the specified join method for each axis Index.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.at_time <pandas.DataFrame.at_time>`
       Select values at particular time of day (e.g. 9:30AM).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.between_time <pandas.DataFrame.between_time>`
       Select values between particular times of the day (e.g., 9:00-9:30 AM).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.drop <pandas.DataFrame.drop>`
       Drop specified labels from rows or columns.
   :ref:`DataFrame.drop_duplicates <pandas.DataFrame.drop_duplicates>`
       Return DataFrame with duplicate rows removed, optionally only considering certain columns. Indexes, including time indexes are ignored.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.duplicated <pandas.DataFrame.duplicated>`
       Return boolean Series denoting duplicate rows, optionally only considering certain columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.equals <pandas.DataFrame.equals>`
       Test whether two objects contain the same elements.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.filter <pandas.DataFrame.filter>`
       Subset rows or columns of dataframe according to labels in the specified index.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.first <pandas.DataFrame.first>`
       Convenience method for subsetting initial periods of time series data based on a date offset.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.head <pandas.DataFrame.head>`
       Return the first `n` rows.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
       Return index of first occurrence of maximum over requested axis. NA/null values are excluded.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
       Return index of first occurrence of minimum over requested axis. NA/null values are excluded.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.last <pandas.DataFrame.last>`
       Convenience method for subsetting final periods of time series data based on a date offset.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.reindex <pandas.DataFrame.reindex>`
       Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. A new object is produced unless the new index is equivalent to the current one and ``copy=False``.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.reindex_like <pandas.DataFrame.reindex_like>`
       Return an object with matching indices as other object.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rename <pandas.DataFrame.rename>`
       Alter axes labels.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.rename_axis <pandas.DataFrame.rename_axis>`
       Set the name of the axis for the index or columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.reset_index <pandas.DataFrame.reset_index>`
       Reset the index, or a level of it.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sample <pandas.DataFrame.sample>`
       Return a random sample of items from an axis of object.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.set_axis <pandas.DataFrame.set_axis>`
       Assign desired index to given axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.set_index <pandas.DataFrame.set_index>`
       Set the DataFrame index using existing columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.tail <pandas.DataFrame.tail>`
       Return the last `n` rows.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.take <pandas.DataFrame.take>`
       Return the elements in the given *positional* indices along an axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.truncate <pandas.DataFrame.truncate>`
       Truncate a Series or DataFrame before and after some index value.

        **Unsupported by Intel SDC**.

Missing Data Handling
---------------------

   :ref:`DataFrame.dropna <pandas.DataFrame.dropna>`
       Remove missing values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.fillna <pandas.DataFrame.fillna>`
       Fill NA/NaN values using the specified method.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.replace <pandas.DataFrame.replace>`
       Replace values given in `to_replace` with `value`.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.interpolate <pandas.DataFrame.interpolate>`
       Interpolate values according to different methods.

        **Unsupported by Intel SDC**.

Re-Shaping, Sorting, Transposing
--------------------------------

   :ref:`DataFrame.droplevel <pandas.DataFrame.droplevel>`
       Return DataFrame with requested index / column level(s) removed.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pivot <pandas.DataFrame.pivot>`
       Return reshaped DataFrame organized by given index / column values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.pivot_table <pandas.DataFrame.pivot_table>`
       Create a spreadsheet-style pivot table as a DataFrame. The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the index and columns of the result DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.reorder_levels <pandas.DataFrame.reorder_levels>`
       Rearrange index levels using input order. May not drop or duplicate levels.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sort_values <pandas.DataFrame.sort_values>`
       Sort by the values along either axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sort_index <pandas.DataFrame.sort_index>`
       Sort object by labels (along an axis).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.nlargest <pandas.DataFrame.nlargest>`
       Return the first `n` rows ordered by `columns` in descending order.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.nsmallest <pandas.DataFrame.nsmallest>`
       Return the first `n` rows ordered by `columns` in ascending order.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.swaplevel <pandas.DataFrame.swaplevel>`
       Swap levels i and j in a MultiIndex on a particular axis.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.stack <pandas.DataFrame.stack>`
       Stack the prescribed level(s) from columns to index.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.unstack <pandas.DataFrame.unstack>`
       Pivot a level of the (necessarily hierarchical) index labels, returning a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.swapaxes <pandas.DataFrame.swapaxes>`
       Interchange axes and swap values axes appropriately.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.melt <pandas.DataFrame.melt>`
       Unpivot a DataFrame from wide format to long format, optionally leaving identifier variables set.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.explode <pandas.DataFrame.explode>`
       Transform each element of a list-like to a row, replicating the index values.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.squeeze <pandas.DataFrame.squeeze>`
       Squeeze 1 dimensional axis objects into scalars.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_xarray <pandas.DataFrame.to_xarray>`
       Return an xarray object from the pandas object.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.T <pandas.DataFrame.T>`
       Transpose index and columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.transpose <pandas.DataFrame.transpose>`
       Transpose index and columns.

        **Unsupported by Intel SDC**.

Combining, Joining, Merging
-----------------------------

   :ref:`DataFrame.append <pandas.DataFrame.append>`
       Append rows of `other` to the end of caller, returning a new object.
   :ref:`DataFrame.assign <pandas.DataFrame.assign>`
       Assign new columns to a DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.join <pandas.DataFrame.join>`
       Join columns of another DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.merge <pandas.DataFrame.merge>`
       Merge DataFrame or named Series objects with a database-style join.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.update <pandas.DataFrame.update>`
       Modify in place using non-NA values from another DataFrame.

        **Unsupported by Intel SDC**.

Time Series
-----------

   :ref:`DataFrame.asfreq <pandas.DataFrame.asfreq>`
       Convert TimeSeries to specified frequency.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.asof <pandas.DataFrame.asof>`
       Return the last row(s) without any NaNs before `where`.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.shift <pandas.DataFrame.shift>`
       Shift index by desired number of periods with an optional time `freq`.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.slice_shift <pandas.DataFrame.slice_shift>`
       Equivalent to `shift` without copying data. The shifted data will not include the dropped periods and the shifted axis will be smaller than the original.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.tshift <pandas.DataFrame.tshift>`
       Shift the time index, using the index's frequency if available.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.first_valid_index <pandas.DataFrame.first_valid_index>`
       Return index for first non-NA/null value.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.last_valid_index <pandas.DataFrame.last_valid_index>`
       Return index for last non-NA/null value.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.resample <pandas.DataFrame.resample>`
       Resample time-series data.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_period <pandas.DataFrame.to_period>`
       Convert DataFrame from DatetimeIndex to PeriodIndex with desired frequency (inferred from index if not passed).

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_timestamp <pandas.DataFrame.to_timestamp>`
       Cast to DatetimeIndex of timestamps, at *beginning* of period.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.tz_convert <pandas.DataFrame.tz_convert>`
       Convert tz-aware axis to target time zone.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.tz_localize <pandas.DataFrame.tz_localize>`
       Localize tz-naive index of a Series or DataFrame to target time zone.

        **Unsupported by Intel SDC**.

.. _api_ref.dataframe.plotting:

Plotting
--------
``DataFrame.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``DataFrame.plot.<kind>``.

   :ref:`DataFrame.plot <pandas.DataFrame.plot>`
       Make plots of Series or DataFrame using the backend specified by the option ``plotting.backend``. By default, matplotlib is used.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.area <pandas.DataFrame.plot.area>`
       Draw a stacked area plot.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.bar <pandas.DataFrame.plot.bar>`
       Vertical bar plot.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.barh <pandas.DataFrame.plot.barh>`
       Make a horizontal bar plot.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.box <pandas.DataFrame.plot.box>`
       Make a box plot of the DataFrame columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.density <pandas.DataFrame.plot.density>`
       Generate Kernel Density Estimate plot using Gaussian kernels.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.hexbin <pandas.DataFrame.plot.hexbin>`
       Generate a hexagonal binning plot.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.hist <pandas.DataFrame.plot.hist>`
       Draw one histogram of the DataFrame's columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.kde <pandas.DataFrame.plot.kde>`
       Generate Kernel Density Estimate plot using Gaussian kernels.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.line <pandas.DataFrame.plot.line>`
       Plot Series or DataFrame as lines.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.pie <pandas.DataFrame.plot.pie>`
       Generate a pie plot.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.plot.scatter <pandas.DataFrame.plot.scatter>`
       Create a scatter plot with varying marker point size and color.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.boxplot <pandas.DataFrame.boxplot>`
       Make a box plot from DataFrame columns.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.hist <pandas.DataFrame.hist>`
       Make a histogram of the DataFrame's.

        **Unsupported by Intel SDC**.

.. _api_ref.dataframe.sparse:

Sparse Accessor
---------------

Sparse-``dtype`` specific methods and attributes are provided under the
``DataFrame.sparse`` accessor.

   :ref:`DataFrame.sparse.density <pandas.DataFrame.sparse.density>`
       Ratio of non-sparse points to total (dense) data points represented in the DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sparse.from_spmatrix <pandas.DataFrame.sparse.from_spmatrix>`
       Create a new DataFrame from a scipy sparse matrix.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sparse.to_coo <pandas.DataFrame.sparse.to_coo>`
       Return the contents of the frame as a sparse SciPy COO matrix.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.sparse.to_dense <pandas.DataFrame.sparse.to_dense>`
       Convert a DataFrame with sparse values to dense.

        **Unsupported by Intel SDC**.

Serialization, Input-Output, Conversion
---------------------------------------

   :ref:`DataFrame.from_dict <pandas.DataFrame.from_dict>`
       Construct DataFrame from dict of array-like or dicts.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.from_records <pandas.DataFrame.from_records>`
       Convert structured or record ndarray to DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.info <pandas.DataFrame.info>`
       Print a concise summary of a DataFrame.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_parquet <pandas.DataFrame.to_parquet>`
       Write a DataFrame to the binary parquet format.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_pickle <pandas.DataFrame.to_pickle>`
       Pickle (serialize) object to file.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_csv <pandas.DataFrame.to_csv>`
       Write object to a comma-separated values (csv) file.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_hdf <pandas.DataFrame.to_hdf>`
       Write the contained data to an HDF5 file using HDFStore.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_sql <pandas.DataFrame.to_sql>`
       Write records stored in a DataFrame to a SQL database.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_dict <pandas.DataFrame.to_dict>`
       Convert the DataFrame to a dictionary.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_excel <pandas.DataFrame.to_excel>`
       Write object to an Excel sheet.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_json <pandas.DataFrame.to_json>`
       Convert the object to a JSON string.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_html <pandas.DataFrame.to_html>`
       Render a DataFrame as an HTML table.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_feather <pandas.DataFrame.to_feather>`
       Write out the binary feather-format for DataFrames.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_latex <pandas.DataFrame.to_latex>`
       Render an object to a LaTeX tabular environment table.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_stata <pandas.DataFrame.to_stata>`
       Export DataFrame object to Stata dta format.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_msgpack <pandas.DataFrame.to_msgpack>`
       Serialize object to input file path using msgpack format.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_gbq <pandas.DataFrame.to_gbq>`
       Write a DataFrame to a Google BigQuery table.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_records <pandas.DataFrame.to_records>`
       Convert DataFrame to a NumPy record array.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_string <pandas.DataFrame.to_string>`
       Render a DataFrame to a console-friendly tabular output.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.to_clipboard <pandas.DataFrame.to_clipboard>`
       Copy object to the system clipboard.

        **Unsupported by Intel SDC**.
   :ref:`DataFrame.style <pandas.DataFrame.style>`
       Property returning a Styler object containing methods for building a styled HTML representation fo the DataFrame.

        **Unsupported by Intel SDC**.
