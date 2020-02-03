.. _pandas.Series.values:

:orphan:

pandas.Series.values
********************

Return Series as ndarray or ndarray-like depending on the dtype.

.. warning::
   We recommend using :attr:`Series.array` or
   :meth:`Series.to_numpy`, depending on whether you need
   a reference to the underlying data or a NumPy array.

:return: numpy.ndarray or ndarray-like

Examples
--------
.. literalinclude:: ../../../examples/series/series_values.py
   :language: python
   :lines: 27-
   :caption: Return Series as ndarray or ndarray-like depending on the dtype.
   :name: ex_series_values

.. command-output:: python ./series/series_values.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.array <pandas.Series.array>`
        Reference to the underlying data.

    :ref:`Series.to_numpy <pandas.Series.to_numpy>`
        A NumPy array representing the underlying data.

