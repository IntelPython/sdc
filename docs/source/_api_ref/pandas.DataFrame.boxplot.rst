.. _pandas.DataFrame.boxplot:

:orphan:

pandas.DataFrame.boxplot
************************

Make a box plot from DataFrame columns.

Make a box-and-whisker plot from DataFrame columns, optionally grouped
by some other columns. A box plot is a method for graphically depicting
groups of numerical data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. The position of the whiskers
is set by default to `1.5 \* IQR (IQR = Q3 - Q1)` from the edges of the box.
Outlier points are those past the end of the whiskers.

For further details see
Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

:param column:
    str or list of str, optional
        Column name or list of names, or vector.
        Can be any valid input to :meth:`pandas.DataFrame.groupby`.

:param by:
    str or array-like, optional
        Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
        One box-plot will be done per value of columns in `by`.

:param ax:
    object of class matplotlib.axes.Axes, optional
        The matplotlib axes to be used by boxplot.

:param fontsize:
    float or str
        Tick label font size in points or as a string (e.g., `large`).

:param rot:
    int or float, default 0
        The rotation angle of labels (in degrees)
        with respect to the screen coordinate system.

:param grid:
    bool, default True
        Setting this to True will show the grid.

:param figsize:
    A tuple (width, height) in inches
        The size of the figure to create in matplotlib.

:param layout:
    tuple (rows, columns), optional
        For example, (3, 5) will display the subplots
        using 3 columns and 5 rows, starting from the top-left.

:param return_type:
    {'axes', 'dict', 'both'} or None, default 'axes'
        The kind of object to return. The default is ``axes``.

        - 'axes' returns the matplotlib axes the boxplot is drawn on.
        - 'dict' returns a dictionary whose values are the matplotlib
            Lines of the boxplot.
        - 'both' returns a namedtuple with the axes and dict.
        - when grouping with ``by``, a Series mapping columns to
            ``return_type`` is returned.

        If ``return_type`` is `None`, a NumPy array
        of axes with the same shape as ``layout`` is returned.
        \*\*kwds
        All other plotting keyword arguments to be passed to
        :func:`matplotlib.pyplot.boxplot`.

:return: result
    See Notes.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

