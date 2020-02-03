.. _pandas.Series.mod:

:orphan:

pandas.Series.mod
*****************

Return Modulo of series and other, element-wise (binary operator `mod`).

Equivalent to ``series % other``, but with support to substitute a fill_value for
missing data in one of the inputs.

:param other:
    Series or scalar value

:param fill_value:
    None or float value, default None (NaN)
        Fill existing missing (NaN) values, and any new element needed for
        successful Series alignment, with this value before computation.
        If data in both corresponding Series locations is missing
        the result will be missing.

:param level:
    int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.

:return: Series
    The result of the operation.

Limitations
-----------
- Parameters level, fill_value are currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_mod.py
   :language: python
   :lines: 27-
   :caption: Return Modulo of series and other, element-wise (binary operator mod).
   :name: ex_series_mod

.. command-output:: python ./series/series_mod.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

