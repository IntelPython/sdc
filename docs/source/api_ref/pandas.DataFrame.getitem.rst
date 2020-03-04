.. _pandas.DataFrame.getitem:

:orphan:

pandas.DataFrame.getitem
***************

Evaluation of `obj[key]`

Limitations
-----------
Supported ``key`` can be one of the following:
    - String literal, e.g. :obj:`df['A']`
    - A slice, e.g. :obj:`df[2:5]`
    - A tuple of string, e.g. :obj:`df[('A', 'B')]`
    - An array of booleans, e.g. :obj:`df[True,False]`
    - A series of booleans, e.g. :obj:`df(series([True,False]))`
Supported getting a column through getting attribute.

Examples
--------
.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_attr.py
   :language: python
   :lines: 37-
   :caption: Getting Pandas DataFrame column through getting attribute.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_attr.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem.py
   :language: python
   :lines: 37-
   :caption: Getting Pandas DataFrame column where key is a string.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_slice.py
   :language: python
   :lines: 34-
   :caption: Getting slice of Pandas DataFrame.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_slice.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_tuple.py
   :language: python
   :lines: 37-
   :caption: Getting Pandas DataFrame elements where key is a tuple of strings.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_tuple.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_array.py
   :language: python
   :lines: 34-
   :caption: Getting Pandas DataFrame elements where key is an array of booleans.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_array.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/dataframe/df_getitem/df_getitem_series.py
   :language: python
   :lines: 34-
   :caption: Getting Pandas DataFrame elements where key is series of booleans.
   :name: ex_dataframe_getitem

.. command-output:: python ./dataframe/df_getitem/df_getitem_series.py
   :cwd: ../../../examples
