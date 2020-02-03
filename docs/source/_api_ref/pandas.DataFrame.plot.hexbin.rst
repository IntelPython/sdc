.. _pandas.DataFrame.plot.hexbin:

:orphan:

pandas.DataFrame.plot.hexbin
****************************

Generate a hexagonal binning plot.

Generate a hexagonal binning plot of `x` versus `y`. If `C` is `None`
(the default), this is a histogram of the number of occurrences
of the observations at ``(x[i], y[i])``.

If `C` is specified, specifies values at given coordinates
``(x[i], y[i])``. These values are accumulated for each hexagonal
bin and then reduced according to `reduce_C_function`,
having as default the NumPy's mean function (:meth:`numpy.mean`).
(If `C` is specified, it must also be a 1-D sequence
of the same length as `x` and `y`, or a column label.)

:param x:
    int or str
        The column label or position for x points.

:param y:
    int or str
        The column label or position for y points.

:param C:
    int or str, optional
        The column label or position for the value of `(x, y)` point.

:param reduce_C_function:
    callable, default `np.mean`
        Function of one argument that reduces all the values in a bin to
        a single number (e.g. `np.mean`, `np.max`, `np.sum`, `np.std`).

:param gridsize:
    int or tuple of (int, int), default 100
        The number of hexagons in the x-direction.
        The corresponding number of hexagons in the y-direction is
        chosen in a way that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements
        specifying the number of hexagons in the x-direction and the
        y-direction.
        \*\*kwds
        Additional keyword arguments are documented in
        :meth:`DataFrame.plot`.

:return: matplotlib.AxesSubplot
    The matplotlib ``Axes`` on which the hexbin is plotted.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

