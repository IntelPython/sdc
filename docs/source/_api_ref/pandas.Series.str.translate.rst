.. _pandas.Series.str.translate:

:orphan:

pandas.Series.str.translate
***************************

Map all characters in the string through the given mapping table.
Equivalent to standard :meth:`str.translate`.

:param table:
    dict
        table is a mapping of Unicode ordinals to Unicode ordinals, strings, or
        None. Unmapped characters are left untouched.
        Characters mapped to None are deleted. :meth:`str.maketrans` is a
        helper function for making translation tables.

:return: Series or Index



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

