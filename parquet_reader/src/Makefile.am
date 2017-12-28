lib_LTLIBRARIES = libhpat_parquet_reader.la
libhpat_parquet_reader_la_SOURCES = hpat_parquet_reader.cpp
AM_CPPFLAGS = -I$(prefix)/include -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
AM_LDFLAGS = -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -L${prefix}/lib -lparquet -larrow
