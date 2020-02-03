.. _pandas.DataFrame.slice_shift:

:orphan:

pandas.DataFrame.slice_shift
****************************

Equivalent to `shift` without copying data. The shifted data will
not include the dropped periods and the shifted axis will be smaller
than the original.

:param periods:
    int
        Number of periods to move, can be positive or negative

:return: shifted : same type as caller



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

