.. _pandas.Series.str.startswith:

:orphan:

pandas.Series.str.startswith
****************************

Test if the start of each string element matches a pattern.

Equivalent to :meth:`str.startswith`.

:param pat:
    str
        Character sequence. Regular expressions are not accepted.

:param na:
    object, default NaN
        Object shown if element tested is not a string.

:return: Series or Index of bool
    A Series of booleans indicating whether the given pattern matches
    the start of each string element.

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_startswith.py
   :language: python
   :lines: 27-
   :caption: Test if the start of each string element matches a string
   :name: ex_series_str_startswith

.. command-output:: python ./series/str/series_str_startswith.py
   :cwd: ../../../examples

.. todo::
    - Add support of matching the start of each string by a pattern
    - Add support of parameter ``na``

.. seealso::
    `str.startswith <https://docs.python.org/3/library/stdtypes.html#str.startswith>`_
        Python standard library string method.
    :ref:`Series.str.endswith <pandas.Series.str.endswith>`
        Same as startswith, but tests the end of string.
    :ref:`Series.str.contains <pandas.Series.str.contains>`
        Tests if string element contains a pattern.

