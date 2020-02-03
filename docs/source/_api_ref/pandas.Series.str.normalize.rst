.. _pandas.Series.str.normalize:

:orphan:

pandas.Series.str.normalize
***************************

Return the Unicode normal form for the strings in the Series/Index.
For more information on the forms, see the
:func:`unicodedata.normalize`.

:param form:
    {'NFC', 'NFKC', 'NFD', 'NFKD'}
        Unicode form

:return: normalized : Series/Index of objects



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

