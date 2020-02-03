.. _pandas.Series.iloc:

:orphan:

pandas.Series.iloc
******************

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

Examples
--------
.. literalinclude:: ../../../examples/series_iloc/series_iloc_value.py
   :language: python
   :lines: 27-
   :caption: With a scalar integer.
   :name: ex_series_iloc

.. command-output:: python ./series_iloc/series_iloc_value.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_iloc/series_iloc_slice.py
   :language: python
   :lines: 27-
   :caption: With a slice object.
   :name: ex_series_iloc

.. command-output:: python ./series_iloc/series_iloc_slice.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.iat <pandas.DataFrame.iat>`
        Fast integer location scalar accessor.

    :ref:`DataFrame.loc <pandas.DataFrame.loc>`
        Purely label-location based indexer for selection by label.

    :ref:`Series.iloc <pandas.Series.iloc>`
        Purely integer-location based indexing for selection by position.

