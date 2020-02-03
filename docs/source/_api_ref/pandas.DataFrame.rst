.. _pandas.DataFrame:

:orphan:

pandas.DataFrame
****************

Two-dimensional size-mutable, potentially heterogeneous tabular data
structure with labeled axes (rows and columns). Arithmetic operations
align on both row and column labels. Can be thought of as a dict-like
container for Series objects. The primary pandas data structure.

:param data:
    ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, or list-like objects

        .. versionchanged :: 0.23.0

        Python 3.6 and later.

        .. versionchanged :: 0.25.0

        for Python 3.6 and later.

:param index:
    Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided

:param columns:
    Index or array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided

:param dtype:
    dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer

:param copy:
    boolean, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

