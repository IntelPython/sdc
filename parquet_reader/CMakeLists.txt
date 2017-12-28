cmake_minimum_required(VERSION 3.2)

project(hpat_parquet_reader)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

include_directories(${PREFIX}/include)
include_directories(${PREFIX}/lib)
link_directories(${PREFIX}/lib)

set(HEADER_FILES ${PREFIX}/include/parquet/api/reader.h
                ${PREFIX}/include/parquet/arrow/reader.h
                ${PREFIX}/include/arrow/table.h
                ${PREFIX}/include/arrow/io/hdfs.h)

add_library(${PROJECT_NAME} SHARED hpat_parquet_reader.cpp ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} arrow)
target_link_libraries(${PROJECT_NAME} parquet)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/)
