.. _api_ref.pandas.groupby:
.. include:: ./../ext_links.txt

GroupBy
=======
.. currentmodule:: pandas.core.groupby

This section covers operations for grouping data in series and dataframes.

GroupBy objects are returned by groupby calls: :func:`pandas.DataFrame.groupby`, :func:`pandas.Series.groupby`, etc.

Indexing and Iteration
----------------------

.. sdc_toctree
GroupBy.__iter__
GroupBy.groups
GroupBy.indices
GroupBy.get_group

.. currentmodule:: pandas

.. sdc_toctree
Grouper

.. currentmodule:: pandas.core.groupby

User-Defined Functions
----------------------

.. sdc_toctree
GroupBy.apply
GroupBy.agg
GroupBy.aggregate
GroupBy.transform
GroupBy.pipe

Computations, Descriptive Statistics
------------------------------------

.. sdc_toctree
GroupBy.all
GroupBy.any
GroupBy.bfill
GroupBy.count
GroupBy.cumcount
GroupBy.cummax
GroupBy.cummin
GroupBy.cumprod
GroupBy.cumsum
GroupBy.ffill
GroupBy.first
GroupBy.head
GroupBy.last
GroupBy.max
GroupBy.mean
GroupBy.median
GroupBy.min
GroupBy.ngroup
GroupBy.nth
GroupBy.ohlc
GroupBy.prod
GroupBy.rank
GroupBy.pct_change
GroupBy.size
GroupBy.sem
GroupBy.std
GroupBy.sum
GroupBy.var
GroupBy.tail

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. sdc_toctree
DataFrameGroupBy.all
DataFrameGroupBy.any
DataFrameGroupBy.bfill
DataFrameGroupBy.corr
DataFrameGroupBy.count
DataFrameGroupBy.cov
DataFrameGroupBy.cummax
DataFrameGroupBy.cummin
DataFrameGroupBy.cumprod
DataFrameGroupBy.cumsum
DataFrameGroupBy.describe
DataFrameGroupBy.diff
DataFrameGroupBy.ffill
DataFrameGroupBy.fillna
DataFrameGroupBy.filter
DataFrameGroupBy.hist
DataFrameGroupBy.idxmax
DataFrameGroupBy.idxmin
DataFrameGroupBy.mad
DataFrameGroupBy.nunique
DataFrameGroupBy.pct_change
DataFrameGroupBy.plot
DataFrameGroupBy.quantile
DataFrameGroupBy.rank
DataFrameGroupBy.resample
DataFrameGroupBy.shift
DataFrameGroupBy.size
DataFrameGroupBy.skew
DataFrameGroupBy.take
DataFrameGroupBy.tshift

The following methods are available only for ``SeriesGroupBy`` objects.

.. sdc_toctree
SeriesGroupBy.nlargest
SeriesGroupBy.nsmallest
SeriesGroupBy.nunique
SeriesGroupBy.unique
SeriesGroupBy.value_counts
SeriesGroupBy.is_monotonic_increasing
SeriesGroupBy.is_monotonic_decreasing

The following methods are available only for ``DataFrameGroupBy`` objects.

.. sdc_toctree
DataFrameGroupBy.corrwith
DataFrameGroupBy.boxplot
