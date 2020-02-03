.. _pandas.core.window.Expanding.quantile:

:orphan:

pandas.core.window.Expanding.quantile
*************************************

Calculate the expanding quantile.

:param quantile:
    float
        Quantile to compute. 0 <= quantile <= 1.

:param interpolation:
    {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        .. versionadded:: 0.23.0

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - linear: `i + (j - i) \* fraction`, where `fraction` is the
            fractional part of the index surrounded by `i` and `j`.
        - lower: `i`.
        - higher: `j`.
        - nearest: `i` or `j` whichever is nearest.
        - midpoint: (`i` + `j`) / 2.
            \*\*kwargs:
            For compatibility with other expanding methods. Has no effect on
            the result.

:return: Series or DataFrame
    Returned object type is determined by the caller of the expanding
    calculation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

