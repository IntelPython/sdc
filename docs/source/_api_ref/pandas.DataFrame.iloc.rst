.. _pandas.DataFrame.iloc:

:orphan:

pandas.DataFrame.iloc
*********************

Purely integer-location based indexing for selection by position.

``.iloc[]`` is primarily integer position based (from ``0`` to
``length-1`` of the axis), but may also be used with a boolean
array.

Allowed inputs are:

- An integer, e.g. ``5``.
- A list or array of integers, e.g. ``[4, 3, 0]``.
- A slice object with ints, e.g. ``1:7``.
- A boolean array.
- A ``callable`` function with one argument (the calling Series or
    DataFrame) and that returns valid output for indexing (one of the above).
    This is useful in method chains, when you don't have a reference to the
    calling object, but would like to base your selection on some value.

``.iloc`` will raise ``IndexError`` if a requested indexer is
out-of-bounds, except *slice* indexers which allow out-of-bounds
indexing (this conforms with python/numpy *slice* semantics).

See more at :ref:`Selection by Position <indexing.integer>`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

