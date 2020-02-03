.. _pandas.Series.le:

:orphan:

pandas.Series.le
****************

Return Less than or equal to of series and other, element-wise (binary operator `le`).

Equivalent to ``series <= other``, but with support to substitute a fill_value for
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

Examples
--------
.. literalinclude:: ../../../examples/series/series_le.py
   :language: python
   :lines: 27-
   :caption: Element-wise less than or equal of one Series by another (binary operator le)
   :name: ex_series_le

.. command-output:: python ./series/series_le.py
   :cwd: ../../../examples

.. note::
    Parameters level, fill_value, axis are currently unsupported by Intel Scalable Dataframe Compiler

