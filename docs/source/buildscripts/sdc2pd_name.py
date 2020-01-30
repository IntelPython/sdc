# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import os
import glob
import sdc  # TODO: Rename hpat module name to sdc

# *****************************************************************************************************
# ***                                     PARSER CONFIGURATION                                      ***
# *****************************************************************************************************

# Exclude these *.py files from parsing that changes SDC internal function names to Pandas-like
exclude_files_list = [
    '__init__.py',
    'hpat_pandas_dataframe_types.py',
    'hpat_pandas_seriesgroupby_types.py',
]


# This dictionary is used to substitute SDC internal lower class name with respective Pandas mixed-case name
CLASSNAME_SUBSTITUTOR = {
    'dataframe': 'DataFrame',
    'series': 'Series',
    'seriesgroupby': 'SeriesGroupBy',
    'timestamp': 'Timestamp',
    'timedelta': 'Timedelta',
    'period': 'Period',
    'interval': 'Interval',
    'index': 'Index',
    'rangeindex': 'RangeIndex',
    'categoricalindex': 'CategoricalIndex',
    'intervalindex': 'IntervalIndex',
    'multiindex': 'MultiIndex',
    'datetimeindex': 'DatetimeIndex',
    'timedeltaindex': 'TimedeltaIndex',
    'periodindex': 'PeriodIndex',
    'dateoffset': 'DateOffset',
    'businessday': 'BusinessDay',
    'businesshour': 'BusinessHour',
    'custombusinessday': 'CustomBusinessDay',
    'custombusinesshour': 'CustomBusinessHour',
    'monthoffset': 'MonthOffset',
    'monthend': 'MonthEnd',
    'monthbegin': 'MonthBegin',
    'businessmonthend': 'BusinessMonthEnd',
    'businessmonthbegin': 'BusinessMonthBegin',
    'custombusinessmonthend': 'CustomBusinessMonthEnd',
    'custombusinessmonthbegin': 'CustomBusinessMonthBegin',
    'semimonthoffset': 'SemiMonthOffset',
    'semimonthend': 'SemiMonthEnd',
    'semimonthbegin': 'SemiMonthBegin',
    'week': 'Week',
    'weekofmonth': 'WeekOfMonth',
    'lastweekofmonth': 'LastWeekOfMonth',
    'quarteroffset': 'QuarterOffset',
    'bquarterend': 'BQuarterEnd',
    'bquarterbegin': 'BQuarterBegin',
    'quarterend': 'QuarterEnd',
    'quarterbegin': 'QuarterBegin',
    'yearoffset': 'YearOffset',
    'byearend': 'BYearEnd',
    'byearbegin': 'BYearBegin',
    'yearend': 'YearEnd',
    'yearbegin': 'YearBegin',
    'fy5253': 'FY5253',
    'fy5253quarter': 'FY5253Quarter',
    'easter': 'Easter',
    'tick': 'Tick',
    'day': 'Day',
    'hour': 'Hour',
    'minute': 'Minute',
    'second': 'Second',
    'milli': 'Milli',
    'micro': 'Micro',
    'nano': 'Nano',
    'bday': 'BDay',
    'bmonthend': 'BMonthEnd',
    'bmonthbegin': 'BMonthBegin',
    'cbmonthend': 'CBMonthEnd',
    'cbmonthbegin': 'CBMonthBegin',
    'cday': 'CDay',
    'rolling': 'Rolling',
    'expanding': 'Expanding',
    'ewm': 'EWM',
    'groupby': 'GroupBy',
    'dataframegroupby': 'DataFrameGroupBy',
    'resampler': 'Resampler',
}


# This is main parsing functions that changes the content of the SDC internal names to Pandas-like
# Note that input fname is the absolute path to the file being parsed
# The parser assumes SDC filename has the structure:
#    sdc_pandas_<classname>_functions
# The parser assumes for a given file of above structure internal names follow the naming scheme:
#    sdc_pandas_<classname>_<function name>
def parse_file(fname):

    # Constructing the SDC function basename
    basename = os.path.basename(fname)  # Extracting file name from the full path
    print('Parsing ' + basename + '...')
    split_basename = basename.split('_')  # Splitting to get the file name structure
    func_name_def_sdc_base = 'def sdc_' + split_basename[1] + \
                    '_' + split_basename[2] + '_'  # Construct the function basename
    func_name_def_hpat_base = 'def hpat_' + split_basename[1] + \
                    '_' + split_basename[2] + '_'  # Construct the function basename
    pd_func_name_def_base = 'def ' + split_basename[1] + '.' + CLASSNAME_SUBSTITUTOR[split_basename[2]] + '.'

    # Loading file content
    with open(fname, 'r') as fn:
        content = fn.read()

        content = content.replace(func_name_def_sdc_base, pd_func_name_def_base)
        content = content.replace(func_name_def_hpat_base, pd_func_name_def_base)
        print(content)


#  Get path to SDC module installed
sdc_path = os.path.dirname(sdc.__file__)  # TODO: Change hpat to sdc
sdc_datatypes_path = os.path.join(sdc_path, "datatypes")
sdc_datatypes_pathname_mask = os.path.join(sdc_datatypes_path, "*.py")

#  Parse all Python files from the SDC installation directory
print('Parsing *.py files in ' + sdc_datatypes_path + ':')

pyfiles_list = glob.glob(sdc_datatypes_pathname_mask)  # Get all *.py file names in the SDC package dir

# Exclude files from exclude_files list
exclude_files_list = [os.path.join(sdc_datatypes_path, fname) for fname in exclude_files_list]  # Abs. path name

pyfiles_list = [fname for fname in pyfiles_list if fname not in exclude_files_list] # File list with exclusions

# ************************************************ PARSING SDC FILES ***********************************
for fname in pyfiles_list:
    parse_file(fname)
