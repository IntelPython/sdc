.. _pandas.DataFrame.plot.area:

:orphan:

pandas.DataFrame.plot.area
**************************

Draw a stacked area plot.

An area plot displays quantitative data visually.
This function wraps the matplotlib area function.

:param x:
    label or position, optional
        Coordinates for the X axis. By default uses the index.

:param y:
    label or position, optional
        Column to plot. By default uses all columns.

:param stacked:
    bool, default True
        Area plots are stacked by default. Set to False to create a
        unstacked plot.

:param \*\*kwds:
    optional
        Additional keyword arguments are documented in
        :meth:`DataFrame.plot`.

:return: matplotlib.axes.Axes or numpy.ndarray
    Area plot, or array of area plots if subplots is True.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

