.. _pandas.Series.to_latex:

:orphan:

pandas.Series.to_latex
**********************

Render an object to a LaTeX tabular environment table.

Render an object to a tabular environment table. You can splice
this into a LaTeX document. Requires \usepackage{booktabs}.

.. versionchanged:: 0.20.2

:param buf:
    file descriptor or None
        Buffer to write to. If None, the output is returned as a string.

:param columns:
    list of label, optional
        The subset of columns to write. Writes all columns by default.

:param col_space:
    int, optional
        The minimum width of each column.

:param header:
    bool or list of str, default True
        Write out the column names. If a list of strings is given,
        it is assumed to be aliases for the column names.

:param index:
    bool, default True
        Write row names (index).

:param na_rep:
    str, default 'NaN'
        Missing data representation.

:param formatters:
    list of functions or dict of {str: function}, optional
        Formatter functions to apply to columns' elements by position or
        name. The result of each function must be a unicode string.
        List must be of length equal to the number of columns.

:param float_format:
    one-parameter function or str, optional, default None
        Formatter for floating point numbers. For example
        ``float_format="%%.2f"`` and ``float_format="{:0.2f}".format`` will
        both result in 0.1234 being formatted as 0.12.

:param sparsify:
    bool, optional
        Set to False for a DataFrame with a hierarchical index to print
        every multiindex key at each row. By default, the value will be
        read from the config module.

:param index_names:
    bool, default True
        Prints the names of the indexes.

:param bold_rows:
    bool, default False
        Make the row labels bold in the output.

:param column_format:
    str, optional
        The columns format as specified in `LaTeX table format
        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g. 'rcl' for 3
        columns. By default, 'l' will be used for all columns except
        columns of numbers, which default to 'r'.

:param longtable:
    bool, optional
        By default, the value will be read from the pandas config
        module. Use a longtable environment instead of tabular. Requires
        adding a \usepackage{longtable} to your LaTeX preamble.

:param escape:
    bool, optional
        By default, the value will be read from the pandas config
        module. When set to False prevents from escaping latex special
        characters in column names.

:param encoding:
    str, optional
        A string representing the encoding to use in the output file,
        defaults to 'utf-8'.

:param decimal:
    str, default '.'
        Character recognized as decimal separator, e.g. ',' in Europe.

        .. versionadded:: 0.18.0

:param multicolumn:
    bool, default True
        Use \multicolumn to enhance MultiIndex columns.
        The default will be read from the config module.

        .. versionadded:: 0.20.0

:param multicolumn_format:
    str, default 'l'
        The alignment for multicolumns, similar to `column_format`
        The default will be read from the config module.

        .. versionadded:: 0.20.0

:param multirow:
    bool, default False
        Use \multirow to enhance MultiIndex rows. Requires adding a
        \usepackage{multirow} to your LaTeX preamble. Will print
        centered labels (instead of top-aligned) across the contained
        rows, separating groups via clines. The default will be read
        from the pandas config module.

        .. versionadded:: 0.20.0

:return: str or None
    If buf is None, returns the resulting LateX format as a
    string. Otherwise returns None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

