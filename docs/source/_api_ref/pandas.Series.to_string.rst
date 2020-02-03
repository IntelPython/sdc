.. _pandas.Series.to_string:

:orphan:

pandas.Series.to_string
***********************

Render a string representation of the Series.

:param buf:
    StringIO-like, optional
        Buffer to write to.

:param na_rep:
    str, optional
        String representation of NaN to use, default 'NaN'.

:param float_format:
    one-parameter function, optional
        Formatter function to apply to columns' elements if they are
        floats, default None.

:param header:
    bool, default True
        Add the Series header (index name).

:param index:
    bool, optional
        Add index (row) labels, default True.

:param length:
    bool, default False
        Add the Series length.

:param dtype:
    bool, default False
        Add the Series dtype.

:param name:
    bool, default False
        Add the Series name if not None.

:param max_rows:
    int, optional
        Maximum number of rows to show before truncating. If None, show
        all.

:param min_rows:
    int, optional
        The number of rows to display in a truncated repr (when number
        of rows is above `max_rows`).

:return: str or None
    String representation of Series if ``buf=None``, otherwise None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

