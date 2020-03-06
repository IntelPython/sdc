.. _pandas.Series.getitem:

:orphan:

pandas.Series.getitem
***************

Evaluation obj[key]

Limitations
-----------
Supported ``key`` can be one of the following:
    - Integer scalar, e.g. :obj:`series[0]`
    - An array or a list, e.g. :obj:`series[0,2,5]`
    - A list of booleans, e.g. :obj:`series[True,False]`
    - A slice, e.g. :obj:`series[2:5]`
    - Another series

Examples
--------
.. literalinclude:: ../../../examples/series/series_getitem/series_getitem_scalar_single_result.py
   :language: python
   :lines: 32-
   :caption: Getting Pandas Series elements. Returns single value.
   :name: ex_series_getitem

.. command-output:: python ./series/series_getitem/series_getitem_scalar_single_result.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_getitem/series_getitem_scalar_multiple_result.py
   :language: python
   :lines: 34-
   :caption: Getting Pandas Series elements. Returns multiple value.
   :name: ex_series_getitem

.. command-output:: python ./series/series_getitem/series_getitem_scalar_multiple_result.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_getitem/series_getitem_slice.py
   :language: python
   :lines: 35-
   :caption: Getting Pandas Series elements by slice.
   :name: ex_series_getitem

.. command-output:: python ./series/series_getitem/series_getitem_slice.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_getitem/series_getitem_bool_array.py
   :language: python
   :lines: 37-
   :caption: Getting Pandas Series elements by array of booleans.
   :name: ex_series_getitem

.. command-output:: python ./series/series_getitem/series_getitem_bool_array.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_getitem/series_getitem_series.py
   :language: python
   :lines: 36-
   :caption: Getting Pandas Series elements by another Series.
   :name: ex_series_getitem

.. command-output:: python ./series/series_getitem/series_getitem_series.py
   :cwd: ../../../examples

.. todo:: Fix SDC behavior and add the expected output of the > python ./series_getitem.py to the docstring
