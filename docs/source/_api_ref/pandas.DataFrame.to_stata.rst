.. _pandas.DataFrame.to_stata:

:orphan:

pandas.DataFrame.to_stata
*************************

Export DataFrame object to Stata dta format.

Writes the DataFrame to a Stata dataset file.
"dta" files contain a Stata dataset.

:param fname:
    str, buffer or path object
        String, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() function. If using a buffer
        then the buffer will not be automatically closed after the file
        data has been written.

:param convert_dates:
    dict
        Dictionary mapping columns containing datetime types to stata
        internal format to use when writing the dates. Options are 'tc',
        'td', 'tm', 'tw', 'th', 'tq', 'ty'. Column can be either an integer
        or a name. Datetime columns that do not have a conversion type
        specified will be converted to 'tc'. Raises NotImplementedError if
        a datetime column has timezone information.

:param write_index:
    bool
        Write the index to Stata dataset.

:param encoding:
    str
        Default is latin-1. Unicode is not supported.

:param byteorder:
    str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`.

:param time_stamp:
    datetime
        A datetime to use as file creation date.  Default is the current
        time.

:param data_label:
    str, optional
        A label for the data set.  Must be 80 characters or smaller.

:param variable_labels:
    dict
        Dictionary containing columns as keys and variable labels as
        values. Each label must be 80 characters or smaller.

        .. versionadded:: 0.19.0

:param version:
    {114, 117}, default 114
        Version to use in the output dta file.  Version 114 can be used
        read by Stata 10 and later.  Version 117 can be read by Stata 13
        or later. Version 114 limits string variables to 244 characters or
        fewer while 117 allows strings with lengths up to 2,000,000
        characters.

        .. versionadded:: 0.23.0

:param convert_strl:
    list, optional
        List of column names to convert to string columns to Stata StrL
        format. Only available if version is 117.  Storing strings in the
        StrL format can produce smaller dta files if strings have more than
        8 characters and values are repeated.

        .. versionadded:: 0.23.0

:raises:
    NotImplementedError
        - If datetimes contain timezone information
        - Column dtype is not representable in Stata
            ValueError
        - Columns listed in convert_dates are neither datetime64[ns]
            or datetime.datetime
        - Column listed in convert_dates is not in DataFrame
        - Categorical label contains more than 32,000 characters

        .. versionadded:: 0.19.0



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

