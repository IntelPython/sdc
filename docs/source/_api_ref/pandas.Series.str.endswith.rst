.. _pandas.Series.str.endswith:

:orphan:

pandas.Series.str.endswith
**************************

Test if the end of each string element matches a pattern.

Equivalent to :meth:`str.endswith`.

:param pat:
    str
        Character sequence. Regular expressions are not accepted.

:param na:
    object, default NaN
        Object shown if element tested is not a string.

:return: Series or Index of bool
    A Series of booleans indicating whether the given pattern matches
    the end of each string element.

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_endswith.py
   :language: python
   :lines: 27-
   :caption: Test if the end of each string element matches a string
   :name: ex_series_str_endswith

.. command-output:: python ./series/str/series_str_endswith.py
   :cwd: ../../../examples

.. todo::
    - Add support of matching the end of each string by a pattern
    - Add support of parameter ``na``

.. seealso::
    `str.endswith <https://docs.python.org/3/library/stdtypes.html#str.endswith>`_
        Python standard library string method.
    :ref:`Series.str.startswith <pandas.Series.str.startswith>`
        Same as endswith, but tests the start of string.
    :ref:`Series.str.contains <pandas.Series.str.contains>`
        Tests if string element contains a pattern.

