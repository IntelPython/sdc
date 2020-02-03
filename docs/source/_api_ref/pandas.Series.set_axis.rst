.. _pandas.Series.set_axis:

:orphan:

pandas.Series.set_axis
**********************

Assign desired index to given axis.

Indexes for column or row labels can be changed by assigning
a list-like or Index.

.. versionchanged:: 0.21.0

   The signature is now `labels` and `axis`, consistent with
   the rest of pandas API. Previously, the `axis` and `labels`
   arguments were respectively the first and second positional
   arguments.

:param labels:
    list-like, Index
        The values for the new index.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to update. The value 0 identifies the rows, and 1
        identifies the columns.

:param inplace:
    bool, default None
        Whether to return a new %(klass)s instance.

.. warning::
        ``inplace=None`` currently falls back to to True, but in a
        future version, will default to False. Use inplace=True
        explicitly rather than relying on the default.

:return: renamed : %(klass)s or None
    An object of same type as caller if inplace=False, None otherwise.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

