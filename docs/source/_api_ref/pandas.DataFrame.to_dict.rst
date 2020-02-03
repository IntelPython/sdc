.. _pandas.DataFrame.to_dict:

:orphan:

pandas.DataFrame.to_dict
************************

Convert the DataFrame to a dictionary.

The type of the key-value pairs can be customized with the parameters
(see below).

:param orient:
    str {'dict', 'list', 'series', 'split', 'records', 'index'}
        Determines the type of the values of the dictionary.

:param - 'dict' (default):
    dict like {column -> {index -> value}}

:param - 'list':
    dict like {column -> [values]}

:param - 'series':
    dict like {column -> Series(values)}

:param - 'split':
    dict like
        {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}

:param - 'records':
    list like
        [{column -> value}, ... , {column -> value}]

:param - 'index':
    dict like {index -> {column -> value}}

        Abbreviations are allowed. `s` indicates `series` and `sp`
        indicates `split`.

:param into:
    class, default dict
        The collections.abc.Mapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

        .. versionadded:: 0.21.0

:return: dict, list or collections.abc.Mapping
    Return a collections.abc.Mapping object representing the DataFrame.
    The resulting transformation depends on the `orient` parameter.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

