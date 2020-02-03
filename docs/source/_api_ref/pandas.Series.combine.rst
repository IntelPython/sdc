.. _pandas.Series.combine:

:orphan:

pandas.Series.combine
*********************

Combine the Series with a Series or scalar according to `func`.

Combine the Series and `other` using `func` to perform elementwise
selection for combined Series.
`fill_value` is assumed when value is missing at some index
from one of the two objects being combined.

:param other:
    Series or scalar
        The value(s) to be combined with the `Series`.

:param func:
    function
        Function that takes two scalars as inputs and returns an element.

:param fill_value:
    scalar, optional
        The value to assume when an index is missing from
        one Series or the other. The default specifies to use the
        appropriate NaN value for the underlying dtype of the Series.

:return: Series
    The result of combining the Series with the other object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

