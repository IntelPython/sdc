.. _pandas.DataFrame.rolling:

:orphan:

pandas.DataFrame.rolling
************************

Provide rolling window calculations.

.. versionadded:: 0.18.0

:param window:
    int, or offset
        Size of the moving window. This is the number of observations used for
        calculating the statistic. Each window will be a fixed size.

        If its an offset then this will be the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes. This is
        new in 0.19.0

:param min_periods:
    int, default None
        Minimum number of observations in window required to have a value
        (otherwise result is NA). For a window that is specified by an offset,
        `min_periods` will default to 1. Otherwise, `min_periods` will default
        to the size of the window.

:param center:
    bool, default False
        Set the labels at the center of the window.

:param win_type:
    str, default None
        Provide a window type. If ``None``, all points are evenly weighted.
        See the notes below for further information.

:param on:
    str, optional
        For a DataFrame, a datetime-like column on which to calculate the rolling
        window, rather than the DataFrame's index. Provided integer column is
        ignored and excluded from result since an integer index is not used to
        calculate the rolling window.

:param axis:
    int or str, default 0

:param closed:
    str, default None
        Make the interval closed on the 'right', 'left', 'both' or
        'neither' endpoints.
        For offset-based windows, it defaults to 'right'.
        For fixed windows, defaults to 'both'. Remaining cases not implemented
        for fixed windows.

        .. versionadded:: 0.20.0

:return: a Window or Rolling sub-classed for the particular operation

Examples
--------
.. literalinclude:: ../../../examples/dataframe/rolling/dataframe_rolling_min.py
   :language: python
   :lines: 27-
   :caption: Calculate the rolling minimum.
   :name: ex_dataframe_rolling

.. command-output:: python ./dataframe/rolling/dataframe_rolling_min.py
   :cwd: ../../../examples

.. todo:: Add support of parameters ``center``, ``win_type``, ``on``, ``axis`` and ``closed``

.. seealso::
    :ref:`expanding <pandas.DataFrame.expanding>`
        Provides expanding transformations.
    :ref:`ewm <pandas.DataFrame.ewm>`
        Provides exponential weighted functions.

