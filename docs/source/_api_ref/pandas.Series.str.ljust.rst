.. _pandas.Series.str.ljust:

:orphan:

pandas.Series.str.ljust
***********************

Filling right side of strings in the Series/Index with an
additional character. Equivalent to :meth:`str.ljust`.

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
.. literalinclude:: ../../../examples/series/str/series_str_ljust.py
   :language: python
   :lines: 27-
   :caption: Filling right side of strings in the Series with an additional character
   :name: ex_series_str_ljust

.. command-output:: python ./series/str/series_str_ljust.py
   :cwd: ../../../examples

.. todo:: Add support of 32-bit Unicode for `str.ljust()`

