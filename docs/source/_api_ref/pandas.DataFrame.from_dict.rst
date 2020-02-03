.. _pandas.DataFrame.from_dict:

:orphan:

pandas.DataFrame.from_dict
**************************

Construct DataFrame from dict of array-like or dicts.

Creates DataFrame object from dictionary by columns or by index
allowing dtype specification.

:param data:
    dict

:param Of the form {field:
    array-like} or {field : dict}.

:param orient:
    {'columns', 'index'}, default 'columns'
        The "orientation" of the data. If the keys of the passed dict
        should be the columns of the resulting DataFrame, pass 'columns'
        (default). Otherwise if the keys should be rows, pass 'index'.

:param dtype:
    dtype, default None
        Data type to force, otherwise infer.

:param columns:
    list, default None
        Column labels to use when ``orient='index'``. Raises a ValueError
        if used with ``orient='columns'``.

        .. versionadded:: 0.23.0

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

