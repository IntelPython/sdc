.. _pandas.Series.repeat:

:orphan:

pandas.Series.repeat
********************

Repeat elements of a Series.

Returns a new Series where each element of the current Series
is repeated consecutively a given number of times.

:param repeats:
    int or array of ints
        The number of repetitions for each element. This should be a
        non-negative integer. Repeating 0 times will return an empty
        Series.

:param axis:
    None
        Must be ``None``. Has no effect but is accepted for compatibility
        with numpy.

:return: Series
    Newly created Series with repeated elements.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

