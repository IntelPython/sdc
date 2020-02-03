.. _pandas.Series.map:

:orphan:

pandas.Series.map
*****************

Map values of Series according to input correspondence.

Used for substituting each value in a Series with another value,
that may be derived from a function, a ``dict`` or
a :class:`Series`.

:param arg:
    function, dict, or Series
        Mapping correspondence.

:param na_action:
    {None, 'ignore'}, default None
        If 'ignore', propagate NaN values, without passing them to the
        mapping correspondence.

:return: Series
    Same index as caller.

Limitations
-----------
String data types are not supported by Intel Scalable Dataframe Compiler.
`arg` could be function or dict and could not be Series. The function should return scalar type.
`na_action` is unsupported by Intel Scalable Dataframe Compiler.

Examples
--------
.. literalinclude:: ../../../examples/series/series_map.py
   :language: python
   :lines: 36-
   :caption: `map()` accepts a function.
   :name: ex_series_map

.. command-output:: python ./series/series_map.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.map <pandas.Series.apply>`
        For applying more complex functions on a Series.
    :ref:`DataFrame.apply <pandas.DataFrame.apply>`
        Apply a function row-/column-wise.
    :ref:`DataFrame.applymap <pandas.DataFrame.applymap>`
        Apply a function elementwise on a whole DataFrame.

