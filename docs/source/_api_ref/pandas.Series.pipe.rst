.. _pandas.Series.pipe:

:orphan:

pandas.Series.pipe
******************

Apply func(self, \\*args, \\*\\*kwargs).

:param func:
    function
        function to apply to the Series/DataFrame.
        ``args``, and ``kwargs`` are passed into ``func``.
        Alternatively a ``(callable, data_keyword)`` tuple where
        ``data_keyword`` is a string indicating the keyword of
        ``callable`` that expects the Series/DataFrame.

:param args:
    iterable, optional
        positional arguments passed into ``func``.

:param kwargs:
    mapping, optional
        a dictionary of keyword arguments passed into ``func``.

:return: object : the return type of ``func``.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

