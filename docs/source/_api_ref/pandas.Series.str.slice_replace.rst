.. _pandas.Series.str.slice_replace:

:orphan:

pandas.Series.str.slice_replace
*******************************

Replace a positional slice of a string with another value.

:param start:
    int, optional
        Left index position to use for the slice. If not specified (None),
        the slice is unbounded on the left, i.e. slice from the start
        of the string.

:param stop:
    int, optional
        Right index position to use for the slice. If not specified (None),
        the slice is unbounded on the right, i.e. slice until the
        end of the string.

:param repl:
    str, optional
        String for replacement. If not specified (None), the sliced region
        is replaced with an empty string.

:return: Series or Index
    Same type as the original object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

