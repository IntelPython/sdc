.. _pandas.DataFrame.hist:

:orphan:

pandas.DataFrame.hist
*********************

Make a histogram of the DataFrame's.

A `histogram`_ is a representation of the distribution of data.
This function calls :meth:`matplotlib.pyplot.hist`, on each series in
the DataFrame, resulting in one histogram per column.

.. _histogram: https://en.wikipedia.org/wiki/Histogram

:param data:
    DataFrame
        The pandas object holding the data.

:param column:
    string or sequence
        If passed, will be used to limit data to a subset of columns.

:param by:
    object, optional
        If passed, then used to form histograms for separate groups.

:param grid:
    bool, default True
        Whether to show axis grid lines.

:param xlabelsize:
    int, default None
        If specified changes the x-axis label size.

:param xrot:
    float, default None
        Rotation of x axis labels. For example, a value of 90 displays the
        x labels rotated 90 degrees clockwise.

:param ylabelsize:
    int, default None
        If specified changes the y-axis label size.

:param yrot:
    float, default None
        Rotation of y axis labels. For example, a value of 90 displays the
        y labels rotated 90 degrees clockwise.

:param ax:
    Matplotlib axes object, default None
        The axes to plot the histogram on.

:param sharex:
    bool, default True if ax is None else False
        In case subplots=True, share x axis and set some x axis labels to
        invisible; defaults to True if ax is None otherwise False if an ax
        is passed in.
        Note that passing in both an ax and sharex=True will alter all x axis
        labels for all subplots in a figure.

:param sharey:
    bool, default False
        In case subplots=True, share y axis and set some y axis labels to
        invisible.

:param figsize:
    tuple
        The size in inches of the figure to create. Uses the value in
        `matplotlib.rcParams` by default.

:param layout:
    tuple, optional
        Tuple of (rows, columns) for the layout of the histograms.

:param bins:
    integer or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
        \*\*kwds
        All other plotting keyword arguments to be passed to
        :meth:`matplotlib.pyplot.hist`.

:return: matplotlib.AxesSubplot or numpy.ndarray of them



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

