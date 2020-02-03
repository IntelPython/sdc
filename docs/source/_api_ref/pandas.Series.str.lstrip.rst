.. _pandas.Series.str.lstrip:

:orphan:

pandas.Series.str.lstrip
************************

Remove leading and trailing characters.

Strip whitespaces (including newlines) or a set of specified characters
from each string in the Series/Index from left side.
Equivalent to :meth:`str.lstrip`.

:param to_strip:
    str or None, default None
        Specifying the set of characters to be removed.
        All combinations of this set of characters will be stripped.
        If None then whitespaces are removed.

:return: Series/Index of objects



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

