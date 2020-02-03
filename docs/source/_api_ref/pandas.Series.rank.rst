.. _pandas.Series.rank:

:orphan:

pandas.Series.rank
******************

Compute numerical data ranks (1 through n) along axis.

By default, equal values are assigned a rank that is the average of the
ranks of those values.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Index to direct ranking.

:param method:
    {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        How to rank the group of records that have the same value
        (i.e. ties):

        - average: average rank of the group
        - min: lowest rank in the group
        - max: highest rank in the group
        - first: ranks assigned in order they appear in the array
        - dense: like 'min', but rank always increases by 1 between groups

:param numeric_only:
    bool, optional
        For DataFrame objects, rank only numeric columns if set to True.

:param na_option:
    {'keep', 'top', 'bottom'}, default 'keep'
        How to rank NaN values:

        - keep: assign NaN rank to NaN values
        - top: assign smallest rank to NaN values if ascending
        - bottom: assign highest rank to NaN values if ascending

:param ascending:
    bool, default True
        Whether or not the elements should be ranked in ascending order.

:param pct:
    bool, default False
        Whether or not to display the returned rankings in percentile
        form.

:return: same type as caller
    Return a Series or DataFrame with data ranks as values.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

