.. _pandas.Series.rename:

:orphan:

pandas.Series.rename
********************

Alter Series index labels or name.

Function / dict values must be unique (1-to-1). Labels not contained in
a dict / Series will be left as-is. Extra labels listed don't throw an
error.

Alternatively, change ``Series.name`` with a scalar value.

See the :ref:`user guide <basics.rename>` for more.

:param index:
    scalar, hashable sequence, dict-like or function, optional
        dict-like or functions are transformations to apply to
        the index.
        Scalar or hashable sequence-like will alter the ``Series.name``
        attribute.

:param copy:
    bool, default True
        Whether to copy underlying data.

:param inplace:
    bool, default False
        Whether to return a new Series. If True then value of copy is
        ignored.

:param level:
    int or level name, default None
        In case of a MultiIndex, only rename labels in the specified
        level.

:return: Series
    Series with index labels or name altered.

Limitations
-----------
- Parameter level is currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_rename.py
   :language: python
   :lines: 27-
   :caption: Alter Series index labels or name.
   :name: ex_series_rename

.. command-output:: python ./series/series_rename.py
   :cwd: ../../../examples

