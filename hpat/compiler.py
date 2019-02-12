from __future__ import print_function, division, absolute_import

# from .pio import PIO
from .distributed import DistributedPass
from .hiframes.hiframes import HiFrames
from .hiframes.hiframes_typed import HiFramesTyped
import numba
import numba.compiler
from numba import ir_utils, ir, postproc
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import guard, get_definition
from numba.inline_closurecall import inline_closure_call, InlineClosureCallPass
from hpat import config
if config._has_h5py:
    from hpat import pio

# this is for previous version of pipeline manipulation (numba hpat_req <0.38)
# def stage_io_pass(pipeline):
#     """
#     Convert IO calls
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     if config._has_h5py:
#         io_pass = pio.PIO(pipeline.func_ir, pipeline.locals)
#         io_pass.run()
#
#
# def stage_distributed_pass(pipeline):
#     """
#     parallelize for distributed-memory
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     dist_pass = DistributedPass(pipeline.func_ir, pipeline.typingctx,
#                                 pipeline.type_annotation.typemap, pipeline.type_annotation.calltypes)
#     dist_pass.run()
#
#
# def stage_df_pass(pipeline):
#     """
#     Convert DataFrame calls
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     df_pass = HiFrames(pipeline.func_ir, pipeline.typingctx,
#                        pipeline.args, pipeline.locals)
#     df_pass.run()
#
#
# def stage_df_typed_pass(pipeline):
#     """
#     Convert HiFrames after typing
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     df_pass = HiFramesTyped(pipeline.func_ir, pipeline.typingctx,
#                             pipeline.type_annotation.typemap, pipeline.type_annotation.calltypes)
#     df_pass.run()
#
#
# def stage_inline_pass(pipeline):
#     """
#     Inline function calls (to enable distributed pass analysis)
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     inline_calls(pipeline.func_ir)
#
#
# def stage_repeat_inline_closure(pipeline):
#     assert pipeline.func_ir
#     inline_pass = InlineClosureCallPass(
#         pipeline.func_ir, pipeline.flags.auto_parallel)
#     inline_pass.run()
#     post_proc = postproc.PostProcessor(pipeline.func_ir)
#     post_proc.run()
#
#
# def add_hpat_stages(pipeline_manager, pipeline):
#     pp = pipeline_manager.pipeline_stages['nopython']
#     new_pp = []
#     for (func, desc) in pp:
#         if desc == 'nopython frontend':
#             # before type inference: add inline calls pass,
#             # untyped hiframes pass, hdf5 io
#             # also repeat inline closure pass to inline df stencils
#             new_pp.append(
#                 (lambda: stage_inline_pass(pipeline), "inline funcs"))
#             new_pp.append((lambda: stage_df_pass(
#                 pipeline), "convert DataFrames"))
#             new_pp.append((lambda: stage_io_pass(
#                 pipeline), "replace IO calls"))
#             new_pp.append((lambda: stage_repeat_inline_closure(
#                 pipeline), "repeat inline closure"))
#         # need to handle string array exprs before nopython rewrites converts
#         # them to arrayexpr.
#         # since generic_rewrites has the same description, we check func name
#         if desc == 'nopython rewrites' and 'generic_rewrites' not in str(func):
#             new_pp.append((lambda: stage_df_typed_pass(
#                 pipeline), "typed hiframes pass"))
#         if desc == 'nopython mode backend':
#             # distributed pass after parfor pass and before lowering
#             new_pp.append((lambda: stage_distributed_pass(
#                 pipeline), "convert to distributed"))
#         new_pp.append((func, desc))
#     pipeline_manager.pipeline_stages['nopython'] = new_pp


def inline_calls(func_ir):
    work_list = list(func_ir.blocks.items())
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Assign):
                lhs = instr.target
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == 'call':
                    func_def = guard(get_definition, func_ir, expr.func)
                    if isinstance(func_def, ir.Global) and isinstance(func_def.value, CPUDispatcher):
                        py_func = func_def.value.py_func
                        new_blocks = inline_closure_call(func_ir,
                                                         py_func.__globals__,
                                                         block, i, py_func, work_list=work_list)
                        # for block in new_blocks:
                        #     work_list.append(block)
                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break

    # sometimes type inference fails after inlining since blocks are inserted
    # at the end and there are agg constraints (categorical_split case)
    # CFG simplification fixes this case
    func_ir.blocks = ir_utils.simplify_CFG(func_ir.blocks)


class HPATPipeline(numba.compiler.BasePipeline):
    """HPAT compiler pipeline
    """
    def define_pipelines(self, pm):
        name = 'hpat'
        pm.create_pipeline(name)
        self.add_preprocessing_stage(pm)
        self.add_with_handling_stage(pm)
        self.add_pre_typing_stage(pm)
        pm.add_stage(self.stage_inline_pass, "inline funcs")
        pm.add_stage(self.stage_df_pass, "convert DataFrames")
        # pm.add_stage(self.stage_io_pass, "replace IO calls")
        # repeat inline closure pass to inline df stencils
        pm.add_stage(self.stage_repeat_inline_closure, "repeat inline closure")
        self.add_typing_stage(pm)
        # breakup optimization stage since df_typed needs to run before
        # rewrites
        # e.g. need to handle string array exprs before nopython rewrites
        # converts them to arrayexpr.
        # self.add_optimization_stage(pm)
        # hiframes typed pass should be before pre_parfor since variable types
        # need updating, and A.call to np.call transformation is invalid for
        # Series (e.g. S.var is not the same as np.var(S))
        pm.add_stage(self.stage_df_typed_pass, "typed hiframes pass")
        pm.add_stage(self.stage_pre_parfor_pass, "Preprocessing for parfors")
        if not self.flags.no_rewrites:
            pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        if self.flags.auto_parallel.enabled:
            pm.add_stage(self.stage_parfor_pass, "convert to parfors")
        pm.add_stage(self.stage_distributed_pass, "convert to distributed")
        pm.add_stage(self.stage_ir_legalization,
                "ensure IR is legal prior to lowering")
        self.add_lowering_stage(pm)
        self.add_cleanup_stage(pm)

    def stage_inline_pass(self):
        """
        Inline function calls (to enable distributed pass analysis)
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        inline_calls(self.func_ir)


    def stage_df_pass(self):
        """
        Convert DataFrame calls
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        df_pass = HiFrames(self.func_ir, self.typingctx,
                           self.args, self.locals, self.metadata)
        df_pass.run()


    def stage_io_pass(self):
        """
        Convert IO calls
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        if config._has_h5py:
            io_pass = pio.PIO(self.func_ir, self.locals)
            io_pass.run()


    def stage_repeat_inline_closure(self):
        assert self.func_ir
        inline_pass = InlineClosureCallPass(
            self.func_ir, self.flags.auto_parallel)
        inline_pass.run()
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()


    def stage_distributed_pass(self):
        """
        parallelize for distributed-memory
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        dist_pass = DistributedPass(
            self.func_ir, self.typingctx, self.targetctx,
            self.type_annotation.typemap, self.type_annotation.calltypes,
            self.metadata)
        dist_pass.run()


    def stage_df_typed_pass(self):
        """
        Convert HiFrames after typing
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        df_pass = HiFramesTyped(self.func_ir, self.typingctx,
                                self.type_annotation.typemap,
                                self.type_annotation.calltypes,
                                self.return_type)
        ret_typ = df_pass.run()
        # XXX update return type since it can be replaced with UnBoxSeries
        # to handle boxing
        if ret_typ is not None:
            self.return_type = ret_typ

class HPATPipelineSeq(HPATPipeline):
    """HPAT pipeline without the distributed pass (used in rolling kernels)
    """
    def define_pipelines(self, pm):
        name = 'hpat_seq'
        pm.create_pipeline(name)
        self.add_preprocessing_stage(pm)
        self.add_with_handling_stage(pm)
        self.add_pre_typing_stage(pm)
        pm.add_stage(self.stage_inline_pass, "inline funcs")
        pm.add_stage(self.stage_df_pass, "convert DataFrames")
        pm.add_stage(self.stage_repeat_inline_closure, "repeat inline closure")
        self.add_typing_stage(pm)
        pm.add_stage(self.stage_df_typed_pass, "typed hiframes pass")
        pm.add_stage(self.stage_pre_parfor_pass, "Preprocessing for parfors")
        if not self.flags.no_rewrites:
            pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        if self.flags.auto_parallel.enabled:
            pm.add_stage(self.stage_parfor_pass, "convert to parfors")
        # pm.add_stage(self.stage_distributed_pass, "convert to distributed")
        pm.add_stage(self.stage_lower_parfor_seq, "parfor seq lower")
        pm.add_stage(self.stage_ir_legalization,
                "ensure IR is legal prior to lowering")
        self.add_lowering_stage(pm)
        self.add_cleanup_stage(pm)

    def stage_lower_parfor_seq(self):
        numba.parfor.lower_parfor_sequential(
                self.typingctx, self.func_ir, self.typemap, self.calltypes)
