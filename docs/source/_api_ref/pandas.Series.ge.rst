.. _pandas.Series.ge:

:orphan:

pandas.Series.ge
****************

Return Greater than or equal to of series and other, element-wise (binary operator `ge`).

Equivalent to ``series >= other``, but with support to substitute a fill_value for
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
.. literalinclude:: ../../../examples/series/series_ge.py
   :language: python
   :lines: 27-
   :caption: Element-wise greater than or equal of one Series by another (binary operator ge)
   :name: ex_series_ge

.. command-output:: python ./series/series_ge.py
   :cwd: ../../../examples

.. note::
    Parameters level, fill_value, axis are currently unsupported by Intel Scalable Dataframe Compiler

