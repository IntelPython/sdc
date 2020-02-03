.. _pandas.Series.str.find:

:orphan:

pandas.Series.str.find
**********************

Return lowest indexes in each strings in the Series/Index
where the substring is fully contained between [start:end].
Return -1 on failure. Equivalent to standard :meth:`str.find`.

:param sub:
    str
        Substring being searched

:param start:
    int
        Left edge index

:param end:
    int
        Right edge index

:return: found : Series/Index of integer values

Limitations
-----------
Series elements are expected to be Unicode strings. Elements cannot be NaN.

Examples
--------
.. literalinclude:: ../../../examples/series/str/series_str_find.py
   :language: python
   :lines: 27-
   :caption: Return lowest indexes in each strings in the Series
   :name: ex_series_str_find

.. command-output:: python ./series/str/series_str_find.py
   :cwd: ../../../examples

.. todo:: Add support of parameters ``start`` and ``end``

.. seealso::
    :ref:`Series.str.rfind <pandas.Series.str.rfind>`
        Return highest indexes in each strings.

