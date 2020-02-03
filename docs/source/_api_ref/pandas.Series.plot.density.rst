.. _pandas.Series.plot.density:

:orphan:

pandas.Series.plot.density
**************************

Generate Kernel Density Estimate plot using Gaussian kernels.

In statistics, `kernel density estimation`_ (KDE) is a non-parametric
way to estimate the probability density function (PDF) of a random
variable. This function uses Gaussian kernels and includes automatic
bandwidth determination.

.. _kernel density estimation:

:param bw_method:
    str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be
        'scott', 'silverman', a scalar constant or a callable.
        If None (default), 'scott' is used.
        See :class:`scipy.stats.gaussian_kde` for more information.

:param ind:
    NumPy array or integer, optional
        Evaluation points for the estimated PDF. If None (default),
        1000 equally spaced points are used. If `ind` is a NumPy array, the
        KDE is evaluated at the points passed. If `ind` is an integer,
        `ind` number of equally spaced points are used.

:param \*\*kwds:
    optional
        Additional keyword arguments are documented in
        :meth:`pandas.%(this-datatype)s.plot`.

:return: matplotlib.axes.Axes or numpy.ndarray of them



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

