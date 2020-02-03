.. _pandas.DataFrame.plot.pie:

:orphan:

pandas.DataFrame.plot.pie
*************************

Generate a pie plot.

A pie plot is a proportional representation of the numerical data in a
column. This function wraps :meth:`matplotlib.pyplot.pie` for the
specified column. If no column reference is passed and
``subplots=True`` a pie plot is drawn for each numerical column
independently.

:param y:
    int or label, optional
        Label or position of the column to plot.
        If not provided, ``subplots=True`` argument must be passed.
        \*\*kwds
        Keyword arguments to pass on to :meth:`DataFrame.plot`.

:return: matplotlib.axes.Axes or np.ndarray of them
    A NumPy array is returned when `subplots` is True.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

