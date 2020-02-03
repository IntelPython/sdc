.. _pandas.DataFrame.append:

:orphan:

pandas.DataFrame.append
***********************

Append rows of `other` to the end of caller, returning a new object.

Columns in `other` that are not in the caller are added as new columns.

:param other:
    DataFrame or Series/dict-like object, or list of these
        The data to append.

:param ignore_index:
    boolean, default False
        If True, do not use the index labels.

:param verify_integrity:
    boolean, default False
        If True, raise ValueError on creating index with duplicates.

:param sort:
    boolean, default None
        Sort columns if the columns of `self` and `other` are not aligned.
        The default sorting is deprecated and will change to not-sorting
        in a future version of pandas. Explicitly pass ``sort=True`` to
        silence the warning and sort. Explicitly pass ``sort=False`` to
        silence the warning and not sort.

        .. versionadded:: 0.23.0

:return: DataFrame

Examples
--------
.. literalinclude:: ../../../examples/dataframe/dataframe_append.py
    :language: python
    :lines: 37-
    :caption: Appending rows of other to the end of caller, returning a new object. Columns in other that are not
              in the caller are added as new columns.
    :name: ex_dataframe_append

.. command-output:: python ./dataframe/dataframe_append.py
    :cwd: ../../../examples

.. note::
    Parameter ignore_index, verify_integrity, sort are currently unsupported
    by Intel Scalable Dataframe Compiler
    Currently only pandas.DataFrame is supported as "other" parameter

.. seealso::
    `pandas.concat <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html>`_
        General function to concatenate DataFrame or Series objects.

