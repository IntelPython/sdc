.. _pandas.DataFrame.plot.box:

:orphan:

pandas.DataFrame.plot.box
*************************

Make a box plot of the DataFrame columns.

A box plot is a method for graphically depicting groups of numerical
data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. The position of the whiskers
is set by default to 1.5\*IQR (IQR = Q3 - Q1) from the edges of the
box. Outlier points are those past the end of the whiskers.

For further details see Wikipedia's
entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

A consideration when using this chart is that the box and the whiskers
can overlap, which is very common when plotting small sets of data.

:param by:
    string or sequence
        Column in the DataFrame to group by.

:param \*\*kwds:
    optional
        Additional keywords are documented in
        :meth:`DataFrame.plot`.

:return: :class:`matplotlib.axes.Axes` or numpy.ndarray of them



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

