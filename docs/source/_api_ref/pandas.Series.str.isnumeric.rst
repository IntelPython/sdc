.. _pandas.Series.str.isnumeric:

:orphan:

pandas.Series.str.isnumeric
***************************

Check whether all characters in each string are numeric.

This is equivalent to running the Python string method
:meth:`str.isnumeric` for each element of the Series/Index. If a string
has zero characters, ``False`` is returned for that check.

:return: Series or Index of bool
    Series or Index of boolean values with the same length as the original
    Series/Index.

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_isnumeric.py
   :language: python
   :lines: 27-
   :caption: Check whether all characters in each string are numeric.
   :name: ex_series_str_isnumeric

.. command-output:: python ./series/str/series_str_isnumeric.py
   :cwd: ../../../examples

.. seealso::

                :ref:`Series.str.isalpha <pandas.Series.str.isalpha>`
                    Check whether all characters are alphabetic.
                :ref:`Series.str.isnumeric <pandas.Series.str.isnumeric>`
                    Check whether all characters are numeric.
                :ref:`Series.str.isalnum <pandas.Series.str.isalnum>`
                    Check whether all characters are alphanumeric.
                :ref:`Series.str.isdigit <pandas.Series.str.isdigit>`
                    Check whether all characters are digits.
                :ref:`Series.str.isdecimal <pandas.Series.str.isdecimal>`
                    Check whether all characters are decimal.
                :ref:`Series.str.isspace <pandas.Series.str.isspace>`
                    Check whether all characters are whitespace.
                :ref:`Series.str.islower <pandas.Series.str.islower>`
                    Check whether all characters are lowercase.
                :ref:`Series.str.isupper <pandas.Series.str.isupper>`
                    Check whether all characters are uppercase.
                :ref:`Series.str.istitle <pandas.Series.str.istitle>`
                    Check whether all characters are titlecase.

