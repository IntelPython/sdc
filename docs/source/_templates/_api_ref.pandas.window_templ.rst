.. _api_ref.pandas.window:
.. include:: ./../ext_links.txt

Rolling Windows
===============
This section covers a collection of moving windows operations on series and dataframes.

Standard Moving Window objects are returned by ``.rolling`` calls:
:func:`pandas.DataFrame.rolling`, :func:`pandas.Series.rolling`, etc.

Expanding Moving Window objects are returned by ``.expanding`` calls:
:func:`pandas.DataFrame.expanding`, :func:`pandas.Series.expanding`, etc.

Exponentially-Weighted Moving Window objects are returned by ``.ewm`` calls:
:func:`pandas.DataFrame.ewm`, :func:`pandas.Series.ewm`, etc.

Standard Moving Window Functions
--------------------------------
.. currentmodule:: pandas.core.window

.. sdc_toctree
Rolling.count
Rolling.sum
Rolling.mean
Rolling.median
Rolling.var
Rolling.std
Rolling.min
Rolling.max
Rolling.corr
Rolling.cov
Rolling.skew
Rolling.kurt
Rolling.apply
Rolling.aggregate
Rolling.quantile

.. currentmodule:: pandas.core.window

.. sdc_toctree
Window.mean
Window.sum

..Window.var
..Window.std

.. _api_ref.pandas.functions_expanding:

Standard Expanding Window Functions
-----------------------------------
.. currentmodule:: pandas.core.window

.. sdc_toctree
Expanding.count
Expanding.sum
Expanding.mean
Expanding.median
Expanding.var
Expanding.std
Expanding.min
Expanding.max
Expanding.corr
Expanding.cov
Expanding.skew
Expanding.kurt
Expanding.apply
Expanding.aggregate
Expanding.quantile

Exponentially-Weighted Moving Window Functions
----------------------------------------------
.. currentmodule:: pandas.core.window

.. sdc_toctree
EWM.mean
EWM.std
EWM.var
EWM.corr
EWM.cov
