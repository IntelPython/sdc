.. _pandas.DataFrame.plot:

:orphan:

pandas.DataFrame.plot
*********************

Make plots of Series or DataFrame using the backend specified by the
option ``plotting.backend``. By default, matplotlib is used.

:param data:
    Series or DataFrame
        The object for which the method is called

:param x:
    label or position, default None
        Only used if data is a DataFrame.

:param y:
    label, position or list of label, positions, default None
        Allows plotting of one column versus another. Only used if data is a
        DataFrame.

:param kind:
    str

:param - 'line':
    line plot (default)

:param - 'bar':
    vertical bar plot

:param - 'barh':
    horizontal bar plot

:param - 'hist':
    histogram

:param - 'box':
    boxplot

:param - 'kde':
    Kernel Density Estimation plot

:param - 'density':
    same as 'kde'

:param - 'area':
    area plot

:param - 'pie':
    pie plot

:param - 'scatter':
    scatter plot

:param - 'hexbin':
    hexbin plot

:param figsize:
    a tuple (width, height) in inches

:param use_index:
    bool, default True
        Use index as ticks for x axis

:param title:
    string or list
        Title to use for the plot. If a string is passed, print the string
        at the top of the figure. If a list is passed and `subplots` is
        True, print each item in the list above the corresponding subplot.

:param grid:
    bool, default None (matlab style default)
        Axis grid lines

:param legend:
    False/True/'reverse'
        Place legend on axis subplots

:param style:
    list or dict
        matplotlib line style per column

:param logx:
    bool or 'sym', default False
        Use log scaling or symlog scaling on x axis
        .. versionchanged:: 0.25.0

:param logy:
    bool or 'sym' default False
        Use log scaling or symlog scaling on y axis
        .. versionchanged:: 0.25.0

:param loglog:
    bool or 'sym', default False
        Use log scaling or symlog scaling on both x and y axes
        .. versionchanged:: 0.25.0

:param xticks:
    sequence
        Values to use for the xticks

:param yticks:
    sequence
        Values to use for the yticks

:param xlim:
    2-tuple/list

:param ylim:
    2-tuple/list

:param rot:
    int, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal
        plots)

:param fontsize:
    int, default None
        Font size for xticks and yticks

:param colormap:
    str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.

:param colorbar:
    bool, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin'
        plots)

:param position:
    float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
        (center)

:param table:
    bool, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data
        will be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a
        table.

:param yerr:
    DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.

:param xerr:
    DataFrame, Series, array-like, dict and str
        Equivalent to yerr.

:param mark_right:
    bool, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend

:param `\*\*kwds`:
    keywords
        Options to pass to matplotlib plotting method

:return: :class:`matplotlib.axes.Axes` or numpy.ndarray of them
    If the backend is not the default matplotlib one, the return value
    will be the object returned by the backend.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

