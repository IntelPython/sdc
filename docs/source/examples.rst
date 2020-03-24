.. _examples:
.. include:: ./ext_links.txt

List of examples
================

    .. literalinclude:: ../../examples/basic_workflow.py
       :language: python
       :lines: 26-
       :name: ex_basic_workflow
        
    .. command-output:: python ./basic_workflow.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/basic_workflow_batch.py
       :language: python
       :lines: 26-
       :name: ex_basic_workflow_batch
        
    .. command-output:: python ./basic_workflow_batch.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/basic_workflow_parallel.py
       :language: python
       :lines: 26-
       :name: ex_basic_workflow_parallel
        
    .. command-output:: python ./basic_workflow_parallel.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/dataframe/dataframe_append.py
        :language: python
        :lines: 37-
        :caption: Appending rows of other to the end of caller, returning a new object. Columns in other that are not
                  in the caller are added as new columns.
        :name: ex_dataframe_append

    .. command-output:: python ./dataframe/dataframe_append.py
        :cwd: ../../examples

    .. literalinclude:: ../../examples/dataframe/dataframe_copy.py
       :language: python
       :lines: 36-
       :caption: Make a copy of this object’s indices and data.
       :name: ex_dataframe_copy

    .. command-output:: python ./dataframe/dataframe_copy.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_count.py
       :language: python
       :lines: 33-
       :caption: Count non-NA cells for each column or row.
       :name: ex_dataframe_count

    .. command-output:: python ./dataframe/dataframe_count.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_drop.py
        :language: python
        :lines: 37-
        :caption: Drop specified columns from DataFrame.
        :name: ex_dataframe_drop

    .. command-output:: python ./dataframe/dataframe_drop.py
        :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_head.py
       :language: python
       :lines: 37-
       :caption: Return the first n rows.
       :name: ex_dataframe_head

    .. command-output:: python ./dataframe/dataframe_head.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_iat.py
       :language: python
       :lines: 28-
       :caption: Get value at specified index position.
       :name: ex_dataframe_iat

    .. command-output:: python ./dataframe/dataframe_iat.py
       :cwd: ../../examples


       .. literalinclude:: ../../examples/dataframe/dataframe_index.py
          :language: python
          :lines: 27-
          :caption: The index (row labels) of the DataFrame.
          :name: ex_dataframe_index

       .. command-output:: python ./dataframe/dataframe_index.py
           :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_isna.py
       :language: python
       :lines: 35-
       :caption: Detect missing values.
       :name: ex_dataframe_isna

    .. command-output:: python ./dataframe/dataframe_isna.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_max.py
       :language: python
       :lines: 35-
       :caption: Return the maximum of the values for the columns.
       :name: ex_dataframe_max

    .. command-output:: python ./dataframe/dataframe_max.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_mean.py
       :language: python
       :lines: 35-
       :caption: Return the mean of the values for the columns.
       :name: ex_dataframe_mean

    .. command-output:: python ./dataframe/dataframe_mean.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_median.py
       :language: python
       :lines: 35-
       :caption: Return the median of the values for the columns.
       :name: ex_dataframe_median

    .. command-output:: python ./dataframe/dataframe_median.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_min.py
       :language: python
       :lines: 35-
       :caption: Return the minimum of the values for the columns.
       :name: ex_dataframe_min

    .. command-output:: python ./dataframe/dataframe_min.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_pct_change.py
        :language: python
        :lines: 36-
        :caption: Percentage change between the current and a prior element.
        :name: ex_dataframe_pct_change

    .. command-output:: python ./dataframe/dataframe_pct_change.py
        :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_prod.py
       :language: python
       :lines: 35-
       :caption: Return the product of the values for the columns.
       :name: ex_dataframe_prod

    .. command-output:: python ./dataframe/dataframe_prod.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/dataframe/dataframe_rolling_sum.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_dataframe_rolling_sum
        
    .. command-output:: python ./dataframe/dataframe_rolling_sum.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/dataframe/dataframe_sum.py
       :language: python
       :lines: 35-
       :caption: Return the sum of the values for the columns.
       :name: ex_dataframe_sum

    .. command-output:: python ./dataframe/dataframe_sum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_values.py
      :language: python
      :lines: 27-
      :caption: The values data of the DataFrame.
      :name: ex_dataframe_values

    .. command-output:: python ./dataframe/dataframe_values.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/dataframe_var.py
       :language: python
       :lines: 35-
       :caption: Return unbiased variance over requested axis.
       :name: ex_dataframe_var

    .. command-output:: python ./dataframe/dataframe_var.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem
        
    .. command-output:: python ./dataframe/getitem/df_getitem.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem_array.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem_array
        
    .. command-output:: python ./dataframe/getitem/df_getitem_array.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem_attr.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem_attr
        
    .. command-output:: python ./dataframe/getitem/df_getitem_attr.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem_series.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem_series
        
    .. command-output:: python ./dataframe/getitem/df_getitem_series.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem_slice.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem_slice
        
    .. command-output:: python ./dataframe/getitem/df_getitem_slice.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/getitem/df_getitem_tuple.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_getitem_df_getitem_tuple
        
    .. command-output:: python ./dataframe/getitem/df_getitem_tuple.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_apply.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_apply
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_apply.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_corr.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_corr
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_corr.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_count.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_count
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_count.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_cov.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_cov
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_cov.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_kurt.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_kurt
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_kurt.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_max.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_max
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_max.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_mean.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_mean
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_mean.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_median.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_median
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_median.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_min.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_min
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_min.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_quantile.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_quantile
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_quantile.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_skew.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_skew
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_skew.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_std.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_std
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_std.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_sum.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_sum
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_sum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/rolling/dataframe_rolling_var.py
       :language: python
       :lines: 26-
       :name: ex_dataframe_rolling_dataframe_rolling_var
        
    .. command-output:: python ./dataframe/rolling/dataframe_rolling_var.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/dataframe/setitem/df_set_existing_column.py
       :language: python
       :lines: 37-
       :caption: Setting data to existing column of the DataFrame.
       :name: ex_dataframe_set_existing_column

    .. command-output:: python ./dataframe/setitem/df_set_existing_column.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/dataframe/setitem/df_set_new_column.py
       :language: python
       :lines: 37-
       :caption: Setting new column to the DataFrame.
       :name: ex_dataframe_set_new_column

    .. command-output:: python ./dataframe/setitem/df_set_new_column.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/rolling/series_rolling_apply.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_apply
        
    .. command-output:: python ./series/rolling/series_rolling_apply.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_corr.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_corr
        
    .. command-output:: python ./series/rolling/series_rolling_corr.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_count.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_count
        
    .. command-output:: python ./series/rolling/series_rolling_count.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_cov.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_cov
        
    .. command-output:: python ./series/rolling/series_rolling_cov.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_kurt.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_kurt
        
    .. command-output:: python ./series/rolling/series_rolling_kurt.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_max.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_max
        
    .. command-output:: python ./series/rolling/series_rolling_max.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_mean.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_mean
        
    .. command-output:: python ./series/rolling/series_rolling_mean.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_median.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_median
        
    .. command-output:: python ./series/rolling/series_rolling_median.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_min.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_min
        
    .. command-output:: python ./series/rolling/series_rolling_min.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_quantile.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_quantile
        
    .. command-output:: python ./series/rolling/series_rolling_quantile.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_skew.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_skew
        
    .. command-output:: python ./series/rolling/series_rolling_skew.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_std.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_std
        
    .. command-output:: python ./series/rolling/series_rolling_std.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_sum.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_sum
        
    .. command-output:: python ./series/rolling/series_rolling_sum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/rolling/series_rolling_var.py
       :language: python
       :lines: 26-
       :name: ex_series_rolling_series_rolling_var
        
    .. command-output:: python ./series/rolling/series_rolling_var.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/series_abs.py
       :language: python
       :lines: 27-
       :caption: Getting the absolute value of each element in Series
       :name: ex_series_abs

    .. command-output:: python ./series/series_abs.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_add.py
       :language: python
       :lines: 27-
       :caption: Getting the addition of Series and other
       :name: ex_series_add

    .. command-output:: python ./series/series_add.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_append.py
       :language: python
       :lines: 37-
       :caption: Concatenate two or more Series.
       :name: ex_series_append

    .. command-output:: python ./series/series_append.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/series_apply.py
       :language: python
       :lines: 26-
       :name: ex_series_series_apply
        
    .. command-output:: python ./series/series_apply.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_apply_lambda.py
       :language: python
       :lines: 26-
       :name: ex_series_series_apply_lambda
        
    .. command-output:: python ./series/series_apply_lambda.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_apply_log.py
       :language: python
       :lines: 26-
       :name: ex_series_series_apply_log
        
    .. command-output:: python ./series/series_apply_log.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/series_argsort.py
       :language: python
       :lines: 27-
       :caption: Override ndarray.argsort.
       :name: ex_series_argsort

    .. command-output:: python ./series/series_argsort.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_astype.py
       :language: python
       :lines: 36-
       :caption: Cast a pandas object to a specified dtype dtype.
       :name: ex_series_astype

    .. command-output:: python ./series/series_astype.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_at/series_at_multiple_result.py
       :language: python
       :lines: 27-
       :caption: With a scalar integer. Returns multiple value.
       :name: ex_series_at

    .. command-output:: python ./series/series_at/series_at_multiple_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_at/series_at_single_result.py
       :language: python
       :lines: 27-
       :caption: With a scalar integer. Returns single value.
       :name: ex_series_at

    .. command-output:: python ./series/series_at/series_at_single_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_copy.py
       :language: python
       :lines: 27-
       :caption: Make a copy of this object’s indices and data.
       :name: ex_series_copy

    .. command-output:: python ./series/series_copy.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_corr.py
       :language: python
       :lines: 27-
       :caption: Compute correlation with other Series, excluding missing values.
       :name: ex_series_corr

    .. command-output:: python ./series/series_corr.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_count.py
       :language: python
       :lines: 27-
       :caption: Counting non-NaN values in Series
       :name: ex_series_count

    .. command-output:: python ./series/series_count.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_cov.py
       :language: python
       :lines: 27-
       :caption: Compute covariance with Series, excluding missing values.
       :name: ex_series_cov

    .. command-output:: python ./series/series_cov.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_cumsum.py
       :language: python
       :lines: 27-
       :caption: Returns cumulative sum over Series.
       :name: ex_series_cumsum

    .. command-output:: python ./series/series_cumsum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_describe.py
       :language: python
       :lines: 39-
       :caption: Generate descriptive statistics.
       :name: ex_series_describe

    .. command-output:: python ./series/series_describe.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_div.py
       :language: python
       :lines: 27-
       :caption: Element-wise division of one Series by another (binary operator div)
       :name: ex_series_div

    .. command-output:: python ./series/series_div.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_dropna.py
       :language: python
       :lines: 34-
       :caption: Return a new Series with missing values removed.
       :name: ex_series_dropna

    .. command-output:: python ./series/series_dropna.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_eq.py
       :language: python
       :lines: 27-
       :caption: Element-wise equal of one Series by another (binary operator eq)
       :name: ex_series_eq

    .. command-output:: python ./series/series_mod.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_fillna.py
       :language: python
       :lines: 35-
       :caption: Fill NA/NaN values using the specified method.
       :name: ex_series_fillna

    .. command-output:: python ./series/series_fillna.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_floordiv.py
       :language: python
       :lines: 27-
       :caption: Return Integer division of series and other, element-wise (binary operator floordiv).
       :name: ex_series_floordiv

    .. command-output:: python ./series/series_floordiv.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_ge.py
       :language: python
       :lines: 27-
       :caption: Element-wise greater than or equal of one Series by another (binary operator ge)
       :name: ex_series_ge

    .. command-output:: python ./series/series_ge.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_getitem/series_getitem_bool_array.py
       :language: python
       :lines: 37-
       :caption: Getting Pandas Series elements by array of booleans.
       :name: ex_series_getitem

    .. command-output:: python ./series/series_getitem/series_getitem_bool_array.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_getitem/series_getitem_scalar_multiple_result.py
       :language: python
       :lines: 34-
       :caption: Getting Pandas Series elements. Returns multiple value.
       :name: ex_series_getitem

    .. command-output:: python ./series/series_getitem/series_getitem_scalar_multiple_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_getitem/series_getitem_scalar_single_result.py
       :language: python
       :lines: 32-
       :caption: Getting Pandas Series elements. Returns single value.
       :name: ex_series_getitem

    .. command-output:: python ./series/series_getitem/series_getitem_scalar_single_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_getitem/series_getitem_series.py
       :language: python
       :lines: 36-
       :caption: Getting Pandas Series elements by another Series.
       :name: ex_series_getitem

    .. command-output:: python ./series/series_getitem/series_getitem_series.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_getitem/series_getitem_slice.py
       :language: python
       :lines: 35-
       :caption: Getting Pandas Series elements by slice.
       :name: ex_series_getitem

    .. command-output:: python ./series/series_getitem/series_getitem_slice.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_groupby.py
       :language: python
       :lines: 33-
       :caption: Return the mean of the values grouped by numpy array.
       :name: ex_series_groupby

    .. command-output:: python ./series/series_groupby.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_gt.py
       :language: python
       :lines: 27-
       :caption: Element-wise greater than of one Series by another (binary operator gt)
       :name: ex_series_gt

    .. command-output:: python ./series/series_gt.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_head.py
       :language: python
       :lines: 34-
       :caption: Getting the first n rows.
       :name: ex_series_head

    .. command-output:: python ./series/series_head.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_iat.py
       :language: python
       :lines: 27-
       :caption: Get value at specified index position.
       :name: ex_series_iat

    .. command-output:: python ./series/series_iat.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_idxmax.py
       :language: python
       :lines: 27-
       :caption: Getting the row label of the maximum value.
       :name: ex_series_idxmax

    .. command-output:: python ./series/series_idxmax.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_idxmin.py
       :language: python
       :lines: 27-
       :caption: Getting the row label of the minimum value.
       :name: ex_series_idxmin

    .. command-output:: python ./series/series_idxmin.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_iloc/series_iloc_slice.py
       :language: python
       :lines: 33-
       :caption: With a slice object.
       :name: ex_series_iloc

    .. command-output:: python ./series/series_iloc/series_iloc_slice.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_iloc/series_iloc_value.py
       :language: python
       :lines: 27-
       :caption: With a scalar integer.
       :name: ex_series_iloc

    .. command-output:: python ./series/series_iloc/series_iloc_value.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_index.py
       :language: python
       :lines: 27-
       :caption: The index (axis labels) of the Series.
       :name: ex_series_index

    .. command-output:: python ./series/series_index.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_isin.py
       :language: python
       :lines: 27-
       :caption: Check whether values are contained in Series.
       :name: ex_series_isin

    .. command-output:: python ./series/series_isin.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_isna.py
       :language: python
       :lines: 27-
       :caption: Detect missing values.
       :name: ex_series_isna

    .. command-output:: python ./series/series_isna.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_isnull.py
       :language: python
       :lines: 27-
       :caption: Detect missing values.
       :name: ex_series_isnull

    .. command-output:: python ./series/series_isnull.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_le.py
       :language: python
       :lines: 27-
       :caption: Element-wise less than or equal of one Series by another (binary operator le)
       :name: ex_series_le

    .. command-output:: python ./series/series_le.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_loc/series_loc_multiple_result.py
       :language: python
       :lines: 34-
       :caption: With a scalar integer. Returns multiple value.
       :name: ex_series_loc

    .. command-output:: python ./series/series_loc/series_loc_multiple_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_loc/series_loc_single_result.py
       :language: python
       :lines: 32-
       :caption: With a scalar integer. Returns single value.
       :name: ex_series_loc

    .. command-output:: python ./series/series_loc/series_loc_single_result.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_loc/series_loc_slice.py
       :language: python
       :lines: 34-
       :caption: With a slice object. Returns multiple value.
       :name: ex_series_loc

    .. command-output:: python ./series/series_loc/series_loc_slice.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_lt.py
       :language: python
       :lines: 27-
       :caption: Element-wise less than of one Series by another (binary operator lt)
       :name: ex_series_lt

    .. command-output:: python ./series/series_lt.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/series_map.py
       :language: python
       :lines: 26-
       :name: ex_series_series_map
        
    .. command-output:: python ./series/series_map.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/series_max.py
       :language: python
       :lines: 27-
       :caption: Getting the maximum value of Series elements
       :name: ex_series_max

    .. command-output:: python ./series/series_max.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_mean.py
       :language: python
       :lines: 27-
       :caption: Return the mean of the values.
       :name: ex_series_mean

    .. command-output:: python ./series/series_mean.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_median.py
       :language: python
       :lines: 27-
       :caption: Return the median of the values for the requested axis.
       :name: ex_series_median

    .. command-output:: python ./series/series_median.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_min.py
       :language: python
       :lines: 27-
       :caption: Getting the minimum value of Series elements
       :name: ex_series_min

    .. command-output:: python ./series/series_min.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_mod.py
       :language: python
       :lines: 27-
       :caption: Return Modulo of series and other, element-wise (binary operator mod).
       :name: ex_series_mod

    .. command-output:: python ./series/series_mod.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_mul.py
       :language: python
       :lines: 27-
       :caption: Element-wise multiplication of two Series
       :name: ex_series_mul

    .. command-output:: python ./series/series_mul.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_ndim.py
       :language: python
       :lines: 27-
       :caption: Number of dimensions of the underlying data, by definition 1.
       :name: ex_series_ndim

    .. command-output:: python ./series/series_ndim.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_ne.py
       :language: python
       :lines: 27-
       :caption: Element-wise not equal of one Series by another (binary operator ne)
       :name: ex_series_ne

    .. command-output:: python ./series/series_ne.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_nlargest.py
       :language: python
       :lines: 27-
       :caption: Returns the largest n elements.
       :name: ex_series_nlargest

    .. command-output:: python ./series/series_nlargest.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_notna.py
       :language: python
       :lines: 27-
       :caption: Detect existing (non-missing) values.
       :name: ex_series_notna

    .. command-output:: python ./series/series_notna.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_nsmallest.py
       :language: python
       :lines: 27-
       :caption: Returns the smallest n elements.
       :name: ex_series_nsmallest

    .. command-output:: python ./series/series_nsmallest.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_nunique.py
       :language: python
       :lines: 27-
       :caption: Return number of unique elements in the object.
       :name: ex_series_nunique

    .. command-output:: python ./series/series_nunique.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_pct_change.py
       :language: python
       :lines: 36-
       :caption: Percentage change between the current and a prior element.
       :name: ex_series_pct_change

    .. command-output:: python ./series/series_pct_change.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_pow.py
       :language: python
       :lines: 27-
       :caption: Element-wise power of one Series by another (binary operator pow)
       :name: ex_series_pow

    .. command-output:: python ./series/series_pow.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_prod.py
       :language: python
       :lines: 27-
       :caption: Return the product of the values.
       :name: ex_series_prod

    .. command-output:: python ./series/series_prod.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_quantile.py
       :language: python
       :lines: 27-
       :caption: Computing quantile for the Series
       :name: ex_series_quantile

    .. command-output:: python ./series/series_quantile.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_rename.py
       :language: python
       :lines: 36-
       :caption: Alter Series index labels or name.
       :name: ex_series_rename

    .. command-output:: python ./series/series_rename.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_setitem_int.py
       :language: python
       :lines: 27-
       :caption: Setting Pandas Series elements
       :name: ex_series_setitem

    .. command-output:: python ./series/series_setitem_int.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_setitem_series.py
       :language: python
       :lines: 27-
       :caption: Setting Pandas Series elements by series
       :name: ex_series_setitem

    .. command-output:: python ./series/series_setitem_series.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_setitem_slice.py
       :language: python
       :lines: 27-
       :caption: Setting Pandas Series elements by slice
       :name: ex_series_setitem

    .. command-output:: python ./series/series_setitem_slice.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_shape.py
       :language: python
       :lines: 27-
       :caption: Return a tuple of the shape of the underlying data.
       :name: ex_series_shape

    .. command-output:: python ./series/series_shape.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_shift.py
       :language: python
       :lines: 36-
       :caption: Shift index by desired number of periods with an optional time freq.
       :name: ex_series_shift

    .. command-output:: python ./series/series_shift.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_size.py
       :language: python
       :lines: 27-
       :caption: Return the number of elements in the underlying data.
       :name: ex_series_size

    .. command-output:: python ./series/series_size.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_sort_values.py
       :language: python
       :lines: 36-
       :caption: Sort by the values.
       :name: ex_series_sort_values

    .. command-output:: python ./series/series_sort_values.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_std.py
       :language: python
       :lines: 27-
       :caption: Returns sample standard deviation over Series.
       :name: ex_series_std

    .. command-output:: python ./series/series_std.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_sub.py
       :language: python
       :lines: 27-
       :caption: Return Subtraction of series and other, element-wise (binary operator sub).
       :name: ex_series_sub

    .. command-output:: python ./series/series_sub.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_sum.py
       :language: python
       :lines: 27-
       :caption: Return the sum of the values for the requested axis.
       :name: ex_series_sum

    .. command-output:: python ./series/series_sum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_T.py
       :language: python
       :lines: 27-
       :caption: Return the transpose, which is by definition self.
       :name: ex_series_T

    .. command-output:: python ./series/series_T.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_take.py
       :language: python
       :lines: 27-
       :caption: Return the elements in the given positional indices along an axis.
       :name: ex_series_take

    .. command-output:: python ./series/series_take.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_truediv.py
       :language: python
       :lines: 27-
       :caption: Element-wise division of one Series by another (binary operator truediv)
       :name: ex_series_truediv

    .. command-output:: python ./series/series_truediv.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_unique.py
       :language: python
       :lines: 27-
       :caption: Getting unique values in Series
       :name: ex_series_unique

    .. command-output:: python ./series/series_unique.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_values.py
       :language: python
       :lines: 27-
       :caption: Return Series as ndarray or ndarray-like depending on the dtype.
       :name: ex_series_values

    .. command-output:: python ./series/series_values.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_value_counts.py
       :language: python
       :lines: 35-
       :caption: Getting the number of values excluding NaNs
       :name: ex_series_value_counts

    .. command-output:: python ./series/series_value_counts.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/series_var.py
       :language: python
       :lines: 27-
       :caption: Returns unbiased variance over Series.
       :name: ex_series_var

    .. command-output:: python ./series/series_var.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/str/series_str_capitalize.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_capitalize
        
    .. command-output:: python ./series/str/series_str_capitalize.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_casefold.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_casefold
        
    .. command-output:: python ./series/str/series_str_casefold.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/str/series_str_center.py
       :language: python
       :lines: 27-
       :caption: Filling left and right side of strings in the Series with an additional character
       :name: ex_series_str_center

    .. command-output:: python ./series/str/series_str_center.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_endswith.py
       :language: python
       :lines: 27-
       :caption: Test if the end of each string element matches a string
       :name: ex_series_str_endswith

    .. command-output:: python ./series/str/series_str_endswith.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_find.py
       :language: python
       :lines: 27-
       :caption: Return lowest indexes in each strings in the Series
       :name: ex_series_str_find

    .. command-output:: python ./series/str/series_str_find.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/str/series_str_isalnum.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isalnum
        
    .. command-output:: python ./series/str/series_str_isalnum.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isalpha.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isalpha
        
    .. command-output:: python ./series/str/series_str_isalpha.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isdecimal.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isdecimal
        
    .. command-output:: python ./series/str/series_str_isdecimal.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isdigit.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isdigit
        
    .. command-output:: python ./series/str/series_str_isdigit.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_islower.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_islower
        
    .. command-output:: python ./series/str/series_str_islower.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isnumeric.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isnumeric
        
    .. command-output:: python ./series/str/series_str_isnumeric.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isspace.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isspace
        
    .. command-output:: python ./series/str/series_str_isspace.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_istitle.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_istitle
        
    .. command-output:: python ./series/str/series_str_istitle.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_isupper.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_isupper
        
    .. command-output:: python ./series/str/series_str_isupper.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/str/series_str_len.py
       :language: python
       :lines: 27-
       :caption: Compute the length of each element in the Series
       :name: ex_series_str_len

    .. command-output:: python ./series/str/series_str_len.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_ljust.py
       :language: python
       :lines: 27-
       :caption: Filling right side of strings in the Series with an additional character
       :name: ex_series_str_ljust

    .. command-output:: python ./series/str/series_str_ljust.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/str/series_str_lstrip.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_lstrip
        
    .. command-output:: python ./series/str/series_str_lstrip.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/str/series_str_rjust.py
       :language: python
       :lines: 27-
       :caption: Filling left side of strings in the Series with an additional character
       :name: ex_series_str_rjust

    .. command-output:: python ./series/str/series_str_rjust.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/str/series_str_rstrip.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_rstrip
        
    .. command-output:: python ./series/str/series_str_rstrip.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/str/series_str_startswith.py
       :language: python
       :lines: 27-
       :caption: Test if the start of each string element matches a string
       :name: ex_series_str_startswith

    .. command-output:: python ./series/str/series_str_startswith.py
       :cwd: ../../examples



    .. literalinclude:: ../../examples/series/str/series_str_strip.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_strip
        
    .. command-output:: python ./series/str/series_str_strip.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_swapcase.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_swapcase
        
    .. command-output:: python ./series/str/series_str_swapcase.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_title.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_title
        
    .. command-output:: python ./series/str/series_str_title.py
       :cwd: ../../examples


    .. literalinclude:: ../../examples/series/str/series_str_upper.py
       :language: python
       :lines: 26-
       :name: ex_series_str_series_str_upper
        
    .. command-output:: python ./series/str/series_str_upper.py
       :cwd: ../../examples

    .. literalinclude:: ../../examples/series/str/series_str_zfill.py
       :language: python
       :lines: 27-
       :caption: Pad strings in the Series by prepending '0' characters
       :name: ex_series_str_zfill

    .. command-output:: python ./series/str/series_str_zfill.py
       :cwd: ../../examples


