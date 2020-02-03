.. _pandas.DataFrame.interpolate:

:orphan:

pandas.DataFrame.interpolate
****************************

Interpolate values according to different methods.

Please note that only ``method='linear'`` is supported for
DataFrame/Series with a MultiIndex.

:param method:
    str, default 'linear'
        Interpolation technique to use. One of:

        - 'linear': Ignore the index and treat the values as equally
            spaced. This is the only method supported on MultiIndexes.
        - 'time': Works on daily and higher resolution data to interpolate
            given length of interval.
        - 'index', 'values': use the actual numerical values of the index.
        - 'pad': Fill in NaNs using existing values.
        - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
            'barycentric', 'polynomial': Passed to
            `scipy.interpolate.interp1d`. These methods use the numerical
            values of the index.  Both 'polynomial' and 'spline' require that
            you also specify an `order` (int), e.g.
            ``df.interpolate(method='polynomial', order=5)``.
        - 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima':
            Wrappers around the SciPy interpolation methods of similar
            names. See `Notes`.
        - 'from_derivatives': Refers to
            `scipy.interpolate.BPoly.from_derivatives` which
            replaces 'piecewise_polynomial' interpolation method in
            scipy 0.18.

        .. versionadded:: 0.18.1

        Added support for the 'akima' method.
        Added interpolate method 'from_derivatives' which replaces
        'piecewise_polynomial' in SciPy 0.18; backwards-compatible with
        SciPy < 0.18

:param axis:
    {0 or 'index', 1 or 'columns', None}, default None
        Axis to interpolate along.

:param limit:
    int, optional
        Maximum number of consecutive NaNs to fill. Must be greater than
        0.

:param inplace:
    bool, default False
        Update the data in place if possible.

:param limit_direction:
    {'forward', 'backward', 'both'}, default 'forward'
        If limit is specified, consecutive NaNs will be filled in this
        direction.

:param limit_area:
    {`None`, 'inside', 'outside'}, default None
        If limit is specified, consecutive NaNs will be filled with this
        restriction.

        - ``None``: No fill restriction.
        - 'inside': Only fill NaNs surrounded by valid values
            (interpolate).
        - 'outside': Only fill NaNs outside valid values (extrapolate).

        .. versionadded:: 0.23.0

:param downcast:
    optional, 'infer' or None, defaults to None
        Downcast dtypes if possible.
        \*\*kwargs
        Keyword arguments to pass on to the interpolating function.

:return: Series or DataFrame
    Returns the same object type as the caller, interpolated at
    some or all ``NaN`` values.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

