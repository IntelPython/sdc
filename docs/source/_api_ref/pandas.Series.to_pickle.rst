.. _pandas.Series.to_pickle:

:orphan:

pandas.Series.to_pickle
***********************

Pickle (serialize) object to file.

:param path:
    str
        File path where the pickled object will be stored.

:param compression:
    {'infer', 'gzip', 'bz2', 'zip', 'xz', None},         default 'infer'
        A string representing the compression to use in the output file. By
        default, infers from the file extension in specified path.

        .. versionadded:: 0.20.0

:param protocol:
    int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see  paragraph 12.1.2). The possible
        values are 0, 1, 2, 3, 4. A negative value for the protocol
        parameter is equivalent to setting its value to HIGHEST_PROTOCOL.

        .. versionadded:: 0.21.0



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

