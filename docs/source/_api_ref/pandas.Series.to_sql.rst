.. _pandas.Series.to_sql:

:orphan:

pandas.Series.to_sql
********************

Write records stored in a DataFrame to a SQL database.

Databases supported by SQLAlchemy  are supported. Tables can be
newly created, appended to, or overwritten.

:param name:
    string
        Name of SQL table.

:param con:
    sqlalchemy.engine.Engine or sqlite3.Connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. Legacy support is provided for sqlite3.Connection objects.

:param schema:
    string, optional
        Specify the schema (if database flavor supports this). If None, use
        default schema.

:param if_exists:
    {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

:param index:
    bool, default True
        Write DataFrame index as a column. Uses `index_label` as the column
        name in the table.

:param index_label:
    string or sequence, default None
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.

:param chunksize:
    int, optional
        Rows will be written in batches of this size at a time. By default,
        all rows will be written at once.

:param dtype:
    dict, optional
        Specifying the datatype for columns. The keys should be the column
        names and the values should be the SQLAlchemy types or strings for
        the sqlite3 legacy mode.

:param method:
    {None, 'multi', callable}, default None
        Controls the SQL insertion clause used:

:param \* None:
    Uses standard SQL ``INSERT`` clause (one per row).
        - 'multi': Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter)``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.

        .. versionadded:: 0.24.0

:raises:
    ValueError
        When the table already exists and `if_exists` is 'fail' (the
        default).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

