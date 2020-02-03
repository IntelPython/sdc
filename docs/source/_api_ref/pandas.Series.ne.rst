.. _pandas.Series.ne:

:orphan:

pandas.Series.ne
****************

Return Not equal to of series and other, element-wise (binary operator `ne`).

Equivalent to ``series != other``, but with support to substitute a fill_value for
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
.. literalinclude:: ../../../examples/series/series_ne.py
   :language: python
   :lines: 27-
   :caption: Element-wise not equal of one Series by another (binary operator ne)
   :name: ex_series_ne

.. command-output:: python ./series/series_ne.py
   :cwd: ../../../examples

.. note::
    Parameters level, fill_value, axis are currently unsupported by Intel Scalable Dataframe Compiler

