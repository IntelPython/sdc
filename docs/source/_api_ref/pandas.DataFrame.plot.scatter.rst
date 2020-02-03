.. _pandas.DataFrame.plot.scatter:

:orphan:

pandas.DataFrame.plot.scatter
*****************************

Create a scatter plot with varying marker point size and color.

The coordinates of each point are defined by two dataframe columns and
filled circles are used to represent each point. This kind of plot is
useful to see complex correlations between two variables. Points could
be for instance natural 2D coordinates like longitude and latitude in
a map or, in general, any pair of metrics that can be plotted against
each other.

:param x:
    int or str
        The column name or column position to be used as horizontal
        coordinates for each point.

:param y:
    int or str
        The column name or column position to be used as vertical
        coordinates for each point.

:param s:
    scalar or array_like, optional
        The size of each point. Possible values are:

        - A single scalar so all points have the same size.

        - A sequence of scalars, which will be used for each point's size
            recursively. For instance, when passing [2,14] all points size
            will be either 2 or 14, alternatively.

:param c:
    str, int or array_like, optional
        The color of each point. Possible values are:

        - A single color string referred to by name, RGB or RGBA code,
            for instance 'red' or '#a98d19'.

        - A sequence of color strings referred to by name, RGB or RGBA
            code, which will be used for each point's color recursively. For
            instance ['green','yellow'] all points will be filled in green or
            yellow, alternatively.

        - A column name or position whose values will be used to color the
            marker points according to a colormap.

        \*\*kwds
        Keyword arguments to pass on to :meth:`DataFrame.plot`.

:return: :class:`matplotlib.axes.Axes` or numpy.ndarray of them



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

