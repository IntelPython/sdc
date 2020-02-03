.. _pandas.DataFrame.to_json:

:orphan:

pandas.DataFrame.to_json
************************

Convert the object to a JSON string.

Note NaN's and None will be converted to null and datetime objects
will be converted to UNIX timestamps.

:param path_or_buf:
    string or file handle, optional
        File path or object. If not specified, the result is returned as
        a string.

:param orient:
    string
        Indication of expected JSON string format.

        - Series

        - default is 'index'
        - allowed values are: {'split','records','index','table'}

        - DataFrame

        - default is 'columns'
        - allowed values are:
            {'split','records','index','columns','values','table'}

        - The format of the JSON string

:param - 'split':
    dict like {'index' -> [index],
        'columns' -> [columns], 'data' -> [values]}

:param - 'records':
    list like
        [{column -> value}, ... , {column -> value}]

:param - 'index':
    dict like {index -> {column -> value}}

:param - 'columns':
    dict like {column -> {index -> value}}

:param - 'values':
    just the values array

:param - 'table':
    dict like {'schema': {schema}, 'data': {data}}
        describing the data, and the data component is
        like ``orient='records'``.

        .. versionchanged:: 0.20.0

:param date_format:
    {None, 'epoch', 'iso'}
        Type of date conversion. 'epoch' = epoch milliseconds,
        'iso' = ISO8601. The default depends on the `orient`. For
        ``orient='table'``, the default is 'iso'. For all other orients,
        the default is 'epoch'.

:param double_precision:
    int, default 10
        The number of decimal places to use when encoding
        floating point values.

:param force_ascii:
    bool, default True
        Force encoded string to be ASCII.

:param date_unit:
    string, default 'ms' (milliseconds)
        The time unit to encode to, governs timestamp and ISO8601
        precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
        microsecond, and nanosecond respectively.

:param default_handler:
    callable, default None
        Handler to call if object cannot otherwise be converted to a
        suitable format for JSON. Should receive a single argument which is
        the object to convert and return a serialisable object.

:param lines:
    bool, default False
        If 'orient' is 'records' write out line delimited json format. Will
        throw ValueError if incorrect 'orient' since others are not list
        like.

        .. versionadded:: 0.19.0

:param compression:
    {'infer', 'gzip', 'bz2', 'zip', 'xz', None}

        A string representing the compression to use in the output file,
        only used when the first argument is a filename. By default, the
        compression is inferred from the filename.

        .. versionadded:: 0.21.0

        'infer' option added and set to default

:param index:
    bool, default True
        Whether to include the index values in the JSON string. Not
        including the index (``index=False``) is only supported when
        orient is 'split' or 'table'.

        .. versionadded:: 0.23.0

:return: None or str
    If path_or_buf is None, returns the resulting json format as a
    string. Otherwise returns None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

