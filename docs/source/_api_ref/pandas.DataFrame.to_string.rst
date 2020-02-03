.. _pandas.DataFrame.to_string:

:orphan:

pandas.DataFrame.to_string
**************************

Render a DataFrame to a console-friendly tabular output.

:param buf:
    StringIO-like, optional
        Buffer to write to.

:param columns:
    sequence, optional, default None
        The subset of columns to write. Writes all columns by default.

:param col_space:
    int, optional
        The minimum width of each column.

:param header:
    bool, optional
        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.

:param index:
    bool, optional, default True
        Whether to print index (row) labels.

:param na_rep:
    str, optional, default 'NaN'
        String representation of NAN to use.

:param formatters:
    list or dict of one-param. functions, optional
        Formatter functions to apply to columns' elements by position or
        name.
        The result of each function must be a unicode string.
        List must be of length equal to the number of columns.

:param float_format:
    one-parameter function, optional, default None
        Formatter function to apply to columns' elements if they are
        floats. The result of this function must be a unicode string.

:param sparsify:
    bool, optional, default True
        Set to False for a DataFrame with a hierarchical index to print
        every multiindex key at each row.

:param index_names:
    bool, optional, default True
        Prints the names of the indexes.

:param justify:
    str, default None
        How to justify the column labels. If None uses the option from
        the print configuration (controlled by set_option), 'right' out
        of the box. Valid values are

        - left
        - right
        - center
        - justify
        - justify-all
        - start
        - end
        - inherit
        - match-parent
        - initial
        - unset.

:param max_rows:
    int, optional
        Maximum number of rows to display in the console.

:param min_rows:
    int, optional
        The number of rows to display in the console in a truncated repr
        (when number of rows is above `max_rows`).

:param max_cols:
    int, optional
        Maximum number of columns to display in the console.

:param show_dimensions:
    bool, default False
        Display DataFrame dimensions (number of rows by number of columns).

:param decimal:
    str, default '.'
        Character recognized as decimal separator, e.g. ',' in Europe.

        .. versionadded:: 0.18.0

:param line_width:
    int, optional
        Width to wrap a line in characters.

:return: str (or unicode, depending on data and options)
    String representation of the dataframe.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

