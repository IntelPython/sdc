.. _pandas.Series.to_dict:

:orphan:

pandas.Series.to_dict
*********************

Convert Series to {label -> value} dict or dict-like object.

:param into:
    class, default dict
        The collections.abc.Mapping subclass to use as the return
        object. Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

        .. versionadded:: 0.21.0

:return: collections.abc.Mapping
    Key-value representation of Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

