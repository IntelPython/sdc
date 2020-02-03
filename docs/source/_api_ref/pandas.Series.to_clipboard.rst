.. _pandas.Series.to_clipboard:

:orphan:

pandas.Series.to_clipboard
**************************

Copy object to the system clipboard.

Write a text representation of object to the system clipboard.
This can be pasted into Excel, for example.

:param excel:
    bool, default True
        - True, use the provided separator, writing in a csv format for
            allowing easy pasting into excel.
        - False, write a string representation of the object to the
            clipboard.

:param sep:
    str, default ``'\t'``
        Field delimiter.
        \*\*kwargs
        These parameters will be passed to DataFrame.to_csv.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

