.. _pandas.Series.str.casefold:

:orphan:

pandas.Series.str.casefold
**************************

Convert strings in the Series/Index to be casefolded.

.. versionadded:: 0.25.0

Equivalent to :meth:`str.casefold`.

:return: Series/Index of objects

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_casefold.py
   :language: python
   :lines: 27-
   :caption: Convert strings in the Series/Index to be casefolded.
   :name: ex_series_str_casefold

.. command-output:: python ./series/str/series_str_casefold.py
   :cwd: ../../../examples

.. seealso::

                    :ref:`Series.str.lower <pandas.Series.str.lower>`
                        Converts all characters to lowercase.
                    :ref:`Series.str.upper <pandas.Series.str.upper>`
                        Converts all characters to uppercase.
                    :ref:`Series.str.title <pandas.Series.str.title>`
                        Converts first character of each word to uppercase and remaining to lowercase.
                    :ref:`Series.str.capitalize <pandas.Series.str.capitalize>`
                        Converts first character to uppercase and remaining to lowercase.
                    :ref:`Series.str.swapcase <pandas.Series.str.swapcase>`
                        Converts uppercase to lowercase and lowercase to uppercase.
                    :ref:`Series.str.casefold <pandas.Series.str.casefold>`
                        Removes all case distinctions in the string.

