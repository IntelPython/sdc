.. _pandas.Series.iat:

:orphan:

pandas.Series.iat
*****************

Access a single value for a row/column pair by integer position.

Similar to ``iloc``, in that both provide integer-based lookups. Use
``iat`` if you only need to get or set a single value in a DataFrame
or Series.

:raises:
    IndexError
        When integer position is out of bounds

Examples
--------
.. literalinclude:: ../../../examples/series_iat.py
   :language: python
   :lines: 27-
   :caption: Get value at specified index position.
   :name: ex_series_iat

.. command-output:: python ./series_iat.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.at <pandas.DataFrame.at>`
        Access a single value for a row/column label pair.

    :ref:`DataFrame.loc <pandas.DataFrame.loc>`
        Purely label-location based indexer for selection by label.

    :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
        Access group of rows and columns by integer position(s).

