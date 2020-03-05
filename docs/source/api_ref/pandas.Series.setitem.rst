.. _pandas.Series.setitem:

:orphan:

pandas.Series.setitem
***************

Limitations
-----------
   - Not supported for idx as a string slice, e.g. S['a':'f'] = value
   - Not supported for string series
   - Not supported for a case of setting value for non existing index
   - Not supported for cases when setting causes change of the Series dtype

Examples
--------
.. literalinclude:: ../../../examples/series_setitem_int.py
   :language: python
   :lines: 27-
   :caption: Setting Pandas Series elements
   :name: ex_series_setitem

.. command-output:: python ./series_setitem_int.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_setitem_slice.py
   :language: python
   :lines: 27-
   :caption: Setting Pandas Series elements by slice
   :name: ex_series_setitem

.. command-output:: python ./series_setitem_slice.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_setitem_series.py
   :language: python
   :lines: 27-
   :caption: Setting Pandas Series elements by series
   :name: ex_series_setitem

.. command-output:: python ./series_setitem_series.py
   :cwd: ../../../examples
