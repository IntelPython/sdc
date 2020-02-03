.. _pandas.Series.str.zfill:

:orphan:

pandas.Series.str.zfill
***********************

Pad strings in the Series/Index by prepending '0' characters.

Strings in the Series/Index are padded with '0' characters on the
left of the string to reach a total string length  `width`. Strings
in the Series/Index with length greater or equal to `width` are
unchanged.

:param width:
    int
        Minimum length of resulting string; strings with length less
        than `width` be prepended with '0' characters.

:return: Series/Index of objects

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_zfill.py
   :language: python
   :lines: 27-
   :caption: Pad strings in the Series by prepending '0' characters
   :name: ex_series_str_zfill

.. command-output:: python ./series/str/series_str_zfill.py
   :cwd: ../../../examples

.. todo:: Add support of 32-bit Unicode for `str.zfill()`

.. seealso::
    :ref:`Series.str.rjust <pandas.Series.str.rjust>`
        Fills the left side of strings with an arbitrary character.
    :ref:`Series.str.ljust <pandas.Series.str.ljust>`
        Fills the right side of strings with an arbitrary character.
    :ref:`Series.str.pad <pandas.Series.str.pad>`
        Fills the specified sides of strings with an arbitrary character.
    :ref:`Series.str.center <pandas.Series.str.center>`
        Fills boths sides of strings with an arbitrary character.

