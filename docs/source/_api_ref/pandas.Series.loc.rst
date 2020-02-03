.. _pandas.Series.loc:

:orphan:

pandas.Series.loc
*****************

Access a group of rows and columns by label(s) or a boolean array.

``.loc[]`` is primarily label based, but may also be used with a
boolean array.

Allowed inputs are:

- A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
    interpreted as a *label* of the index, and **never** as an
    integer position along the index).
- A list or array of labels, e.g. ``['a', 'b', 'c']``.
- A slice object with labels, e.g. ``'a':'f'``.

.. warning:: Note that contrary to usual python slices, **both** the
      start and the stop are included

- A boolean array of the same length as the axis being sliced,
    e.g. ``[True, False, True]``.
- A ``callable`` function with one argument (the calling Series or
    DataFrame) and that returns valid output for indexing (one of the above)

See more at :ref:`Selection by Label <indexing.label>`

:raises:
    KeyError:
        when any items are not found

Limitations
-----------
- Loc returns Series
- Loc slice and callable with String is not implemented
- Loc slice without start is not supported
- Loc callable returns float Series

Examples
--------
.. literalinclude:: ../../../examples/series_loc/series_loc_single_result.py
   :language: python
   :lines: 27-
   :caption: With a scalar integer. Returns single value.
   :name: ex_series_loc

.. command-output:: python ./series_loc/series_loc_single_result.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_loc/series_loc_multiple_result.py
   :language: python
   :lines: 27-
   :caption: With a scalar integer. Returns multiple value.
   :name: ex_series_loc

.. command-output:: python ./series_loc/series_loc_multiple_result.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_loc/series_loc_slice.py
   :language: python
   :lines: 27-
   :caption: With a slice object. Returns multiple value.
   :name: ex_series_loc

.. command-output:: python ./series_loc/series_loc_slice.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.at <pandas.DataFrame.at>`
        Access a single value for a row/column label pair.

    :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
        Access group of rows and columns by integer position(s).

    :ref:`DataFrame.xs <pandas.DataFrame.xs>`
        Returns a cross-section (row(s) or column(s)) from the Series/DataFrame.

    :ref:`Series.loc <pandas.Series.loc>`
        Access group of values using labels.

