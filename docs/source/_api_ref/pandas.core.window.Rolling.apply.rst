.. _pandas.core.window.Rolling.apply:

:orphan:

pandas.core.window.Rolling.apply
********************************

The rolling function's apply function.

:param func:
    function
        Must produce a single value from an ndarray input if ``raw=True``
        or a single value from a Series if ``raw=False``.

:param raw:
    bool, default None

:param \* ``False``:
    passes each row or column as a Series to the
        function.

:param \* ``True`` or ``None``:
    the passed function will receive ndarray
        objects instead.
        If you are just applying a NumPy reduction function this will
        achieve much better performance.

        The `raw` parameter is required and will show a FutureWarning if
        not passed. In the future `raw` will default to False.

        .. versionadded:: 0.23.0

        Arguments and keyword arguments to be passed into func.

:return: Series or DataFrame
    Return type is determined by the caller.

Limitations
-----------
Supported ``raw`` only can be `None` or `True`. Parameters ``args``, ``kwargs`` unsupported.
Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

Examples
--------
.. literalinclude:: ../../../examples/series/rolling/series_rolling_apply.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling apply.
   :name: ex_series_rolling_apply

.. command-output:: python ./series/rolling/series_rolling_apply.py
   :cwd: ../../../examples

.. seealso::
    :ref:`Series.rolling <pandas.Series.rolling>`
        Calling object with a Series.
    :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
        Calling object with a DataFrame.
    :ref:`Series.apply <pandas.Series.apply>`
        Similar method for Series.
    :ref:`DataFrame.apply <pandas.DataFrame.apply>`
        Similar method for DataFrame.

