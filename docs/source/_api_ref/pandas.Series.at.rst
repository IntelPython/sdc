.. _pandas.Series.at:

:orphan:

pandas.Series.at
****************

Access a single value for a row/column label pair.

Similar to ``loc``, in that both provide label-based lookups. Use
``at`` if you only need to get or set a single value in a DataFrame
or Series.

:raises:
    KeyError
        When label does not exist in DataFrame

Examples
--------
.. literalinclude:: ../../../examples/series_at/series_at_single_result.py
   :language: python
   :lines: 27-
   :caption: With a scalar integer. Returns single value.
   :name: ex_series_at

.. command-output:: python ./series_at/series_at_single_result.py
   :cwd: ../../../examples

.. literalinclude:: ../../../examples/series_at/series_at_multiple_result.py
   :language: python
   :lines: 27-
   :caption: With a scalar integer. Returns multiple value.
   :name: ex_series_at

.. command-output:: python ./series_at/series_at_multiple_result.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.iat <pandas.DataFrame.iat>`
        Access a single value for a row/column pair by integer position.

    :ref:`DataFrame.loc <pandas.DataFrame.loc>`
        Access a group of rows and columns by label(s).

    :ref:`Series.at <pandas.Series.at>`
        Access a single value using a label.

