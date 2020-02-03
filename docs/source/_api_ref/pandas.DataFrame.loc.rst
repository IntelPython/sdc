.. _pandas.DataFrame.loc:

:orphan:

pandas.DataFrame.loc
********************

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

