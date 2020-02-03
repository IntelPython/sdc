.. _pandas.DataFrame.sample:

:orphan:

pandas.DataFrame.sample
***********************

Return a random sample of items from an axis of object.

You can use `random_state` for reproducibility.

:param n:
    int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.

:param frac:
    float, optional
        Fraction of axis items to return. Cannot be used with `n`.

:param replace:
    bool, default False
        Sample with or without replacement.

:param weights:
    str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        Infinite values not allowed.

:param random_state:
    int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState
        object.

:param axis:
    int or string, optional
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames).

:return: Series or DataFrame
    A new object of same type as caller containing `n` items randomly
    sampled from the caller object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

