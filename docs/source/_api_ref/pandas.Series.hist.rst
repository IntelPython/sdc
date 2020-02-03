.. _pandas.Series.hist:

:orphan:

pandas.Series.hist
******************

Draw histogram of the input series using matplotlib.

:param by:
    object, optional
        If passed, then used to form histograms for separate groups

:param ax:
    matplotlib axis object
        If not passed, uses gca()

:param grid:
    bool, default True
        Whether to show axis grid lines

:param xlabelsize:
    int, default None
        If specified changes the x-axis label size

:param xrot:
    float, default None
        rotation of x axis labels

:param ylabelsize:
    int, default None
        If specified changes the y-axis label size

:param yrot:
    float, default None
        rotation of y axis labels

:param figsize:
    tuple, default None
        figure size in inches by default

:param bins:
    integer or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.

:param `\*\*kwds`:
    keywords
        To be passed to the actual plotting function

:return: matplotlib.AxesSubplot
    A histogram plot.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

