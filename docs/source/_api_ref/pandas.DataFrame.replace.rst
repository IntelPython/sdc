.. _pandas.DataFrame.replace:

:orphan:

pandas.DataFrame.replace
************************

Replace values given in `to_replace` with `value`.

Values of the DataFrame are replaced with other values dynamically.
This differs from updating with ``.loc`` or ``.iloc``, which require
you to specify a location to update with some value.

:param to_replace:
    str, regex, list, dict, Series, int, float, or None
        How to find the values that will be replaced.

        - numeric, str or regex:

        - numeric: numeric values equal to `to_replace` will be
            replaced with `value`
        - str: string exactly matching `to_replace` will be replaced
            with `value`
        - regex: regexs matching `to_replace` will be replaced with
            `value`

        - list of str, regex, or numeric:

        - First, if `to_replace` and `value` are both lists, they
            **must** be the same length.
        - Second, if ``regex=True`` then all of the strings in **both**
            lists will be interpreted as regexs otherwise they will match
            directly. This doesn't matter much for `value` since there
            are only a few possible substitution regexes you can use.
        - str, regex and numeric rules apply as above.

        - dict:

        - Dicts can be used to specify different replacement values
            for different existing values. For example,
            ``{'a': 'b', 'y': 'z'}`` replaces the value 'a' with 'b' and
            'y' with 'z'. To use a dict in this way the `value`
            parameter should be `None`.
        - For a DataFrame a dict can specify that different values
            should be replaced in different columns. For example,
            ``{'a': 1, 'b': 'z'}`` looks for the value 1 in column 'a'
            and the value 'z' in column 'b' and replaces these values
            with whatever is specified in `value`. The `value` parameter
            should not be ``None`` in this case. You can treat this as a
            special case of passing two lists except that you are
            specifying the column to search in.
        - For a DataFrame nested dictionaries, e.g.,
            ``{'a': {'b': np.nan}}``, are read as follows: look in column
            'a' for the value 'b' and replace it with NaN. The `value`
            parameter should be ``None`` to use a nested dict in this
            way. You can nest regular expressions as well. Note that
            column names (the top-level dictionary keys in a nested
            dictionary) **cannot** be regular expressions.

        - None:

        - This means that the `regex` argument must be a string,
            compiled regular expression, or list, dict, ndarray or
            Series of such elements. If `value` is also ``None`` then
            this **must** be a nested dictionary or Series.

        See the examples section for examples of each of these.

:param value:
    scalar, dict, list, str, regex, default None
        Value to replace any values matching `to_replace` with.
        For a DataFrame a dict of values can be used to specify which
        value to use for each column (columns not in the dict will not be
        filled). Regular expressions, strings and lists or dicts of such
        objects are also allowed.

:param inplace:
    bool, default False
        If True, in place. Note: this will modify any
        other views on this object (e.g. a column from a DataFrame).
        Returns the caller if this is True.

:param limit:
    int, default None
        Maximum size gap to forward or backward fill.

:param regex:
    bool or same types as `to_replace`, default False
        Whether to interpret `to_replace` and/or `value` as regular
        expressions. If this is ``True`` then `to_replace` *must* be a
        string. Alternatively, this could be a regular expression or a
        list, dict, or array of regular expressions in which case
        `to_replace` must be ``None``.

:param method:
    {'pad', 'ffill', 'bfill', `None`}
        The method to use when for replacement, when `to_replace` is a
        scalar, list or tuple and `value` is ``None``.

        .. versionchanged:: 0.23.0

:return: DataFrame
    Object after replacement.

:raises:
    AssertionError
        - If `regex` is not a ``bool`` and `to_replace` is not
            ``None``.
            TypeError
        - If `to_replace` is a ``dict`` and `value` is not a ``list``,
            ``dict``, ``ndarray``, or ``Series``
        - If `to_replace` is ``None`` and `regex` is not compilable
            into a regular expression or is a list, dict, ndarray, or
            Series.
        - When replacing multiple ``bool`` or ``datetime64`` objects and
            the arguments to `to_replace` does not match the type of the
            value being replaced
            ValueError
        - If a ``list`` or an ``ndarray`` is passed to `to_replace` and
            `value` but they are not the same length.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

