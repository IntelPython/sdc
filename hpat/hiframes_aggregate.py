from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config
from numba.ir_utils import visit_vars_inner, replace_vars_inner
from numba.typing import signature
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_arr_ext import string_array_type


class Aggregate(ir.Stmt):
    def __init__(self, df_out, df_in, key_name, df_out_vars, df_in_vars,
                                                                key_arr, loc):
        # name of output dataframe (just for printing purposes)
        self.df_out = df_out
        # name of input dataframe (just for printing purposes)
        self.df_in = df_in
        # key name (for printing)
        self.key_name = key_name

        self.df_out_vars = df_out_vars
        self.df_in_vars = df_in_vars
        self.key_arr = key_arr

        self.loc = loc

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        return "aggregate: {} = {} [key: {}] ".format(df_out_str, df_in_str,
                                                    self.key_name)
