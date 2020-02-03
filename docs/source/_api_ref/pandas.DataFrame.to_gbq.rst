.. _pandas.DataFrame.to_gbq:

:orphan:

pandas.DataFrame.to_gbq
***********************

Write a DataFrame to a Google BigQuery table.

This function requires the `pandas-gbq package
<https://pandas-gbq.readthedocs.io>`__.

See the `How to authenticate with Google BigQuery
<https://pandas-gbq.readthedocs.io/en/latest/howto/authentication.html>`__
guide for authentication instructions.

:param destination_table:
    str
        Name of table to be written, in the form ``dataset.tablename``.

:param project_id:
    str, optional
        Google BigQuery Account project ID. Optional when available from
        the environment.

:param chunksize:
    int, optional
        Number of rows to be inserted in each chunk from the dataframe.
        Set to ``None`` to load the whole dataframe at once.

:param reauth:
    bool, default False
        Force Google BigQuery to re-authenticate the user. This is useful
        if multiple accounts are used.

:param if_exists:
    str, default 'fail'
        Behavior when the destination table exists. Value can be one of:

        ``'fail'``
        If table exists, do nothing.
        ``'replace'``
        If table exists, drop it, recreate it, and insert data.
        ``'append'``
        If table exists, insert data. Create if does not exist.

:param auth_local_webserver:
    bool, default False
        Use the `local webserver flow`_ instead of the `console flow`_
        when getting user credentials.

        .. _local webserver flow:

        .. _console flow:

        *New in version 0.2.0 of pandas-gbq*.

:param table_schema:
    list of dicts, optional
        List of BigQuery table fields to which according DataFrame
        columns conform to, e.g. ``[{'name': 'col1', 'type':
        'STRING'},...]``. If schema is not provided, it will be
        generated according to dtypes of DataFrame columns. See
        BigQuery API documentation on available names of a field.

        *New in version 0.3.1 of pandas-gbq*.

:param location:
    str, optional
        Location where the load job should run. See the `BigQuery locations
        documentation
        <https://cloud.google.com/bigquery/docs/dataset-locations>`__ for a
        list of available locations. The location must match that of the
        target dataset.

        *New in version 0.5.0 of pandas-gbq*.

:param progress_bar:
    bool, default True
        Use the library `tqdm` to show the progress bar for the upload,
        chunk by chunk.

        *New in version 0.5.0 of pandas-gbq*.

:param credentials:
    google.auth.credentials.Credentials, optional
        Credentials for accessing Google APIs. Use this parameter to
        override default credentials, such as to use Compute Engine
        :class:`google.auth.compute_engine.Credentials` or Service
        Account :class:`google.oauth2.service_account.Credentials`
        directly.

        *New in version 0.8.0 of pandas-gbq*.

        .. versionadded:: 0.24.0

:param verbose:
    bool, deprecated
        Deprecated in pandas-gbq version 0.4.0. Use the `logging module
        to adjust verbosity instead
        <https://pandas-gbq.readthedocs.io/en/latest/intro.html#logging>`__.

:param private_key:
    str, deprecated
        Deprecated in pandas-gbq version 0.8.0. Use the ``credentials``
        parameter and
        :func:`google.oauth2.service_account.Credentials.from_service_account_info`
        or
        :func:`google.oauth2.service_account.Credentials.from_service_account_file`
        instead.

        Service account private key in JSON format. Can be file path
        or string contents. This is useful for remote server
        authentication (eg. Jupyter/IPython notebook on remote host).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

