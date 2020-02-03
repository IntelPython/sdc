.. _pandas.Series.abs:

:orphan:

pandas.Series.abs
*****************

Return a Series/DataFrame with absolute numeric value of each element.

This function only applies to elements that are all numeric.

:return: abs
    Series/DataFrame containing the absolute value of each element.

Examples
--------
.. literalinclude:: ../../../examples/series/series_abs.py
   :language: python
   :lines: 27-
   :caption: Getting the absolute value of each element in Series
   :name: ex_series_abs

.. command-output:: python ./series/series_abs.py
   :cwd: ../../../examples

.. seealso::

    `numpy.absolute <https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html>`_
        Calculate the absolute value element-wise.

