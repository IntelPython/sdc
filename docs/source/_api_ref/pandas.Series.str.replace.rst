.. _pandas.Series.str.replace:

:orphan:

pandas.Series.str.replace
*************************

Replace occurrences of pattern/regex in the Series/Index with
some other string. Equivalent to :meth:`str.replace` or
:func:`re.sub`.

:param pat:
    str or compiled regex
        String can be a character sequence or regular expression.

        .. versionadded:: 0.20.0

:param repl:
    str or callable
        Replacement string or a callable. The callable is passed the regex
        match object and must return a replacement string to be used.
        See :func:`re.sub`.

        .. versionadded:: 0.20.0

:param n:
    int, default -1 (all)
        Number of replacements to make from start.

:param case:
    bool, default None
        - If True, case sensitive (the default if `pat` is a string)
        - Set to False for case insensitive
        - Cannot be set if `pat` is a compiled regex

:param flags:
    int, default 0 (no flags)
        - re module flags, e.g. re.IGNORECASE
        - Cannot be set if `pat` is a compiled regex

:param regex:
    bool, default True
        - If True, assumes the passed-in pattern is a regular expression.
        - If False, treats the pattern as a literal string
        - Cannot be set to False if `pat` is a compiled regex or `repl` is
            a callable.

        .. versionadded:: 0.23.0

:return: Series or Index of object
    A copy of the object with all matching occurrences of `pat` replaced by
    `repl`.

:raises:
    ValueError
        - if `regex` is False and `repl` is a callable or `pat` is a compiled
            regex
        - if `pat` is a compiled regex and `case` or `flags` is set



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

