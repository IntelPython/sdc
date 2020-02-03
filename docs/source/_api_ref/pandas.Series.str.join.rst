.. _pandas.Series.str.join:

:orphan:

pandas.Series.str.join
**********************

Join lists contained as elements in the Series/Index with passed delimiter.

If the elements of a Series are lists themselves, join the content of these
lists using the delimiter passed to the function.
This function is an equivalent to :meth:`str.join`.

:param sep:
    str
        Delimiter to use between list entries.

:return: Series/Index: object
    The list entries concatenated by intervening occurrences of the
    delimiter.

:raises:
    AttributeError
        If the supplied Series contains neither strings nor lists.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

