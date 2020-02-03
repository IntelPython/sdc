.. _pandas.DataFrame.to_csv:

:orphan:

pandas.DataFrame.to_csv
***********************

Write object to a comma-separated values (csv) file.

.. versionchanged:: 0.24.0

:param path_or_buf:
    str or file handle, default None
        File path or object, if None is provided the result is returned as
        a string.  If a file object is passed it should be opened with
        `newline=''`, disabling universal newlines.

        .. versionchanged:: 0.24.0

        Was previously named "path" for Series.

:param sep:
    str, default ','
        String of length 1. Field delimiter for the output file.

:param na_rep:
    str, default ''
        Missing data representation.

:param float_format:
    str, default None
        Format string for floating point numbers.

:param columns:
    sequence, optional
        Columns to write.

:param header:
    bool or list of str, default True
        Write out the column names. If a list of strings is given it is
        assumed to be aliases for the column names.

        .. versionchanged:: 0.24.0

        Previously defaulted to False for Series.

:param index:
    bool, default True
        Write row names (index).

:param index_label:
    str or sequence, or False, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the object uses MultiIndex. If
        False do not print fields for index names. Use index_label=False
        for easier importing in R.

:param mode:
    str
        Python write mode, default 'w'.

:param encoding:
    str, optional
        A string representing the encoding to use in the output file,
        defaults to 'utf-8'.

:param compression:
    str, default 'infer'
        Compression mode among the following possible values: {'infer',
        'gzip', 'bz2', 'zip', 'xz', None}. If 'infer' and `path_or_buf`
        is path-like, then detect compression from the following
        extensions: '.gz', '.bz2', '.zip' or '.xz'. (otherwise no
        compression).

        .. versionchanged:: 0.24.0

        'infer' option added and set to default.

:param quoting:
    optional constant from csv module
        Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
        then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
        will treat them as non-numeric.

:param quotechar:
    str, default '\"'
        String of length 1. Character used to quote fields.

:param line_terminator:
    str, optional
        The newline character or character sequence to use in the output
        file. Defaults to `os.linesep`, which depends on the OS in which
        this method is called ('\n' for linux, '\r\n' for Windows, i.e.).

        .. versionchanged:: 0.24.0

:param chunksize:
    int or None
        Rows to write at a time.

:param date_format:
    str, default None
        Format string for datetime objects.

:param doublequote:
    bool, default True
        Control quoting of `quotechar` inside a field.

:param escapechar:
    str, default None
        String of length 1. Character used to escape `sep` and `quotechar`
        when appropriate.

:param decimal:
    str, default '.'
        Character recognized as decimal separator. E.g. use ',' for
        European data.

:return: None or str
    If path_or_buf is None, returns the resulting csv format as a
    string. Otherwise returns None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

