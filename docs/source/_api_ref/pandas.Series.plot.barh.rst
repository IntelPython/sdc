.. _pandas.Series.plot.barh:

:orphan:

pandas.Series.plot.barh
***********************

Make a horizontal bar plot.

A horizontal bar plot is a plot that presents quantitative data with
rectangular bars with lengths proportional to the values that they
represent. A bar plot shows comparisons among discrete categories. One
axis of the plot shows the specific categories being compared, and the
other axis represents a measured value.

:param x:
    label or position, default DataFrame.index
        Column to be used for categories.

:param y:
    label or position, default All numeric columns in dataframe
        Columns to be plotted from the DataFrame.
        \*\*kwds
        Keyword arguments to pass on to :meth:`DataFrame.plot`.

:return: :class:`matplotlib.axes.Axes` or numpy.ndarray of them



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

