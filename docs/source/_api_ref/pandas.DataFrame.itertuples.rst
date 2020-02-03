.. _pandas.DataFrame.itertuples:

:orphan:

pandas.DataFrame.itertuples
***************************

Iterate over DataFrame rows as namedtuples.

:param index:
    bool, default True
        If True, return the index as the first element of the tuple.

:param name:
    str or None, default "Pandas"
        The name of the returned namedtuples or None to return regular
        tuples.

:return: iterator
    An object to iterate over namedtuples for each row in the
    DataFrame with the first field possibly being the index and
    following fields being the column values.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

