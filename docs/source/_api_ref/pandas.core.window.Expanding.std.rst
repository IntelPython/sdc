.. _pandas.core.window.Expanding.std:

:orphan:

pandas.core.window.Expanding.std
********************************

Calculate expanding standard deviation.

Normalized by N-1 by default. This can be changed using the `ddof`
argument.

:param ddof:
    int, default 1
        Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        \*args, \*\*kwargs
        For NumPy compatibility. No additional arguments are used.

:return: Series or DataFrame
    Returns the same object type as the caller of the expanding calculation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

