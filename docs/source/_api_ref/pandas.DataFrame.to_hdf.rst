.. _pandas.DataFrame.to_hdf:

:orphan:

pandas.DataFrame.to_hdf
***********************

Write the contained data to an HDF5 file using HDFStore.

Hierarchical Data Format (HDF) is self-describing, allowing an
application to interpret the structure and contents of a file with
no outside information. One HDF file can hold a mix of related objects
which can be accessed as a group or as individual objects.

In order to add another DataFrame or Series to an existing HDF file
please use append mode and a different a key.

For more information see the :ref:`user guide <io.hdf5>`.

:param path_or_buf:
    str or pandas.HDFStore
        File path or HDFStore object.

:param key:
    str
        Identifier for the group in the store.

:param mode:
    {'a', 'w', 'r+'}, default 'a'
        Mode to open file:

        - 'w': write, a new file is created (an existing file with
            the same name would be deleted).
        - 'a': append, an existing file is opened for reading and
            writing, and if the file does not exist it is created.
        - 'r+': similar to 'a', but the file must already exist.

:param format:
    {'fixed', 'table'}, default 'fixed'
        Possible values:

        - 'fixed': Fixed format. Fast writing/reading. Not-appendable,
            nor searchable.
        - 'table': Table format. Write as a PyTables Table structure
            which may perform worse but allow more flexible operations
            like searching / selecting subsets of the data.

:param append:
    bool, default False
        For Table formats, append the input data to the existing.

:param data_columns:
    list of columns or True, optional
       List of columns to create as indexed data columns for on-disk
       queries, or True to use all columns. By default only the axes
       of the object are indexed. See :ref:`io.hdf5-query-data-columns`.
       Applicable only to format='table'.

:param complevel:
    {0-9}, optional
        Specifies a compression level for data.
        A value of 0 disables compression.

:param complib:
    {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        As of v0.20.2 these additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
        'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.

:param fletcher32:
    bool, default False
        If applying compression use the fletcher32 checksum.

:param dropna:
    bool, default False
        If true, ALL nan rows will not be written to store.

:param errors:
    str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

