.. _pandas.DataFrame.tshift:

:orphan:

pandas.DataFrame.tshift
***********************

Shift the time index, using the index's frequency if available.

:param periods:
    int
        Number of periods to move, can be positive or negative

:param freq:
    DateOffset, timedelta, or time rule string, default None
        Increment to use from the tseries module or time rule (e.g. 'EOM')

:param axis:
    int or basestring
        Corresponds to the axis that contains the Index

:return: shifted : Series/DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

