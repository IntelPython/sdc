.. _pandas.Series.str.len:

:orphan:

pandas.Series.str.len
*********************

Compute the length of each element in the Series/Index. The element may be
a sequence (such as a string, tuple or list) or a collection
(such as a dictionary).

:return: Series or Index of int
    A Series or Index of integer values indicating the length of each
    element in the Series or Index.

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_len.py
   :language: python
   :lines: 27-
   :caption: Compute the length of each element in the Series
   :name: ex_series_str_len

.. command-output:: python ./series/str/series_str_len.py
   :cwd: ../../../examples

.. seealso::
    `str.len`
        Python built-in function returning the length of an object.
    :ref:`Series.size <pandas.Series.size>`
        Returns the length of the Series.

