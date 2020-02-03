.. _pandas.core.window.Expanding.apply:

:orphan:

pandas.core.window.Expanding.apply
**********************************

The expanding function's apply function.

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

