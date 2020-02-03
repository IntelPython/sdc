.. _pandas.Series.plot.bar:

:orphan:

pandas.Series.plot.bar
**********************

Vertical bar plot.

A bar plot is a plot that presents categorical data with
rectangular bars with lengths proportional to the values that they
represent. A bar plot shows comparisons among discrete categories. One
axis of the plot shows the specific categories being compared, and the
other axis represents a measured value.

:param x:
    label or position, optional
        Allows plotting of one column versus another. If not specified,
        the index of the DataFrame is used.

:param y:
    label or position, optional
        Allows plotting of one column versus another. If not specified,
        all numerical columns are used.
        \*\*kwds
        Additional keyword arguments are documented in
        :meth:`DataFrame.plot`.

:return: matplotlib.axes.Axes or np.ndarray of them
    An ndarray is returned with one :class:`matplotlib.axes.Axes`
    per column when ``subplots=True``.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

