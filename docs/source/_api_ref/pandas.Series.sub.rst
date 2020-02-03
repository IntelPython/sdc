.. _pandas.Series.sub:

:orphan:

pandas.Series.sub
*****************

Return Subtraction of series and other, element-wise (binary operator `sub`).

Equivalent to ``series - other``, but with support to substitute a fill_value for
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
.. literalinclude:: ../../../examples/series/series_sub.py
   :language: python
   :lines: 27-
   :caption: Return Subtraction of series and other, element-wise (binary operator sub).
   :name: ex_series_sub

.. command-output:: python ./series/series_sub.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.rsub <pandas.Series.rsub>`

