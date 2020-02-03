.. _pandas.Series.to_excel:

:orphan:

pandas.Series.to_excel
**********************

Write object to an Excel sheet.

To write a single object to an Excel .xlsx file it is only necessary to
specify a target file name. To write to multiple sheets it is necessary to
create an `ExcelWriter` object with a target file name, and specify a sheet
in the file to write to.

Multiple sheets may be written to by specifying unique `sheet_name`.
With all data written to the file it is necessary to save the changes.
Note that creating an `ExcelWriter` object with a file name that already
exists will result in the contents of the existing file being erased.

:param excel_writer:
    str or ExcelWriter object
        File path or existing ExcelWriter.

:param sheet_name:
    str, default 'Sheet1'
        Name of sheet which will contain DataFrame.

:param na_rep:
    str, default ''
        Missing data representation.

:param float_format:
    str, optional
        Format string for floating point numbers. For example
        ``float_format="%.2f"`` will format 0.1234 to 0.12.

:param columns:
    sequence or list of str, optional
        Columns to write.

:param header:
    bool or list of str, default True
        Write out the column names. If a list of string is given it is
        assumed to be aliases for the column names.

:param index:
    bool, default True
        Write row names (index).

:param index_label:
    str or sequence, optional
        Column label for index column(s) if desired. If not specified, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.

:param startrow:
    int, default 0
        Upper left cell row to dump data frame.

:param startcol:
    int, default 0
        Upper left cell column to dump data frame.

:param engine:
    str, optional
        Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
        via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``, and
        ``io.excel.xlsm.writer``.

:param merge_cells:
    bool, default True
        Write MultiIndex and Hierarchical Rows as merged cells.

:param encoding:
    str, optional
        Encoding of the resulting excel file. Only necessary for xlwt,
        other writers support unicode natively.

:param inf_rep:
    str, default 'inf'
        Representation for infinity (there is no native representation for
        infinity in Excel).

:param verbose:
    bool, default True
        Display more information in the error logs.

:param freeze_panes:
    tuple of int (length 2), optional
        Specifies the one-based bottommost row and rightmost column that
        is to be frozen.

        .. versionadded:: 0.20.0.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

