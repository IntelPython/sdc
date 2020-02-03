.. _pandas.Series.str.rjust:

:orphan:

pandas.Series.str.rjust
***********************

Filling left side of strings in the Series/Index with an
additional character. Equivalent to :meth:`str.rjust`.

:param width:
    int
        Minimum width of resulting string; additional characters will be filled
        with ``fillchar``

:param fillchar:
    str
        Additional character for filling, default is whitespace

:return: filled : Series/Index of objects

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_rjust.py
   :language: python
   :lines: 27-
   :caption: Filling left side of strings in the Series with an additional character
   :name: ex_series_str_rjust

.. command-output:: python ./series/str/series_str_rjust.py
   :cwd: ../../../examples

.. todo:: Add support of 32-bit Unicode for `str.rjust()`

