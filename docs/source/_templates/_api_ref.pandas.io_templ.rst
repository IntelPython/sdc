.. _api_ref.pandas.io:
.. include:: ./../ext_links.txt

Input-Output
============
.. currentmodule:: pandas

This section include `Pandas*`_ functions for input data of a specific format into memory and for output in-memory
data to external storage format.


Pickling
~~~~~~~~

.. sdc_toctree
read_pickle

Flat Files
~~~~~~~~~~

.. sdc_toctree
read_table
read_csv
read_fwf

Clipboard
~~~~~~~~~

.. sdc_toctree
read_clipboard

Excel
~~~~~

.. sdc_toctree
read_excel
ExcelFile.parse
ExcelWriter

JSON
~~~~

.. sdc_toctree
read_json

.. currentmodule:: pandas.io.json

.. sdc_toctree
json_normalize
build_table_schema

.. currentmodule:: pandas

HTML
~~~~

.. sdc_toctree
read_html

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. sdc_toctree
read_hdf
HDFStore.put
HDFStore.append
HDFStore.get
HDFStore.select
HDFStore.info
HDFStore.keys
HDFStore.groups
HDFStore.walk

Feather
~~~~~~~

.. sdc_toctree
read_feather

Parquet
~~~~~~~

.. sdc_toctree
read_parquet

SAS
~~~

.. sdc_toctree
read_sas

SPSS
~~~~

.. sdc_toctree
read_spss

SQL
~~~

.. sdc_toctree
read_sql_table
read_sql_query
read_sql

Google BigQuery
~~~~~~~~~~~~~~~

.. sdc_toctree
read_gbq

STATA
~~~~~

.. sdc_toctree
read_stata

.. currentmodule:: pandas.io.stata

.. sdc_toctree
StataReader.data_label
StataReader.value_labels
StataReader.variable_labels
StataWriter.write_file
