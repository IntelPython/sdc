.. _pandas.Series.to_xarray:

:orphan:

pandas.Series.to_xarray
***********************

Return an xarray object from the pandas object.

:return: xarray.DataArray or xarray.Dataset
    Data in the pandas structure converted to Dataset if the object is
    a DataFrame, or a DataArray if the object is a Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

