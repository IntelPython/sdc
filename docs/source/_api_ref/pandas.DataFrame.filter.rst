.. _pandas.DataFrame.filter:

:orphan:

pandas.DataFrame.filter
***********************

Subset rows or columns of dataframe according to labels in
the specified index.

Note that this routine does not filter a dataframe on its
contents. The filter is applied to the labels of the index.

:param items:
    list-like
        Keep labels from axis which are in items.

:param like:
    string
        Keep labels from axis for which "like in label == True".

:param regex:
    string (regular expression)
        Keep labels from axis for which re.search(regex, label) == True.

:param axis:
    int or string axis name
        The axis to filter on.  By default this is the info axis,
        'index' for Series, 'columns' for DataFrame.

:return: same type as input object



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

