.. _pandas.DataFrame.getitem:

:orphan:

pandas.DataFrame.getitem
***************

Evaluation obj[key]

Limitations
-----------
Supported ``key`` can be one of the following:
    - String literal, e.g. :obj:`df['A']`
    - A slice, e.g. :obj:`df[2:5]`
    - A tuple of string, e.g. :obj:`df[('A', 'B')]`
    - An array of booleans, e.g. :obj:`df[True,False]`
    - A series of booleans, e.g. :obj:`df(series([True,False]))`
Column through getting attribute.

Examples
--------
.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_key.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame column through getting attribute.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_key.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_column.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame elements. Key is string.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_column.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_slice.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame elements. Key is slice of integer.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_slice.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_tuple.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame elements. Key is tuple of string.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_tuple.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_array.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame elements. Key is array of boolean.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_array.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_series.py
   :language: python
   :lines: 27-
   :caption: Getting Pandas DataFrame elements. Key is series of boolean.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_series.py
   :cwd: ../../../examples
