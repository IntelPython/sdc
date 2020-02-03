.. _pandas.DataFrame.plot.line:

:orphan:

pandas.DataFrame.plot.line
**************************

Plot Series or DataFrame as lines.

This function is useful to plot lines using DataFrame's values
as coordinates.

:param x:
    int or str, optional
        Columns to use for the horizontal axis.
        Either the location or the label of the columns to be used.
        By default, it will use the DataFrame indices.

:param y:
    int, str, or list of them, optional
        The values to be plotted.
        Either the location or the label of the columns to be used.
        By default, it will use the remaining DataFrame numeric columns.
        \*\*kwds
        Keyword arguments to pass on to :meth:`DataFrame.plot`.

:return: :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray`
    Return an ndarray when ``subplots=True``.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

