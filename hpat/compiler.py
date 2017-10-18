from __future__ import print_function, division, absolute_import

# from .pio import PIO
from .distributed import DistributedPass
from .hiframes import HiFrames
import numba
import numba.compiler
from numba import ir_utils, ir
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import guard, get_definition
from numba.inline_closurecall import inline_closure_call

# def stage_io_pass(pipeline):
#     """
#     Convert IO calls
#     """
#     # Ensure we have an IR and type information.
#     assert pipeline.func_ir
#     io_pass = PIO(pipeline.func_ir, pipeline.locals)
#     io_pass.run()

def stage_distributed_pass(pipeline):
    """
    parallelize for distributed-memory
    """
    # Ensure we have an IR and type information.
    assert pipeline.func_ir
    dist_pass = DistributedPass(pipeline.func_ir, pipeline.typingctx,
        pipeline.type_annotation.typemap, pipeline.type_annotation.calltypes)
    dist_pass.run()

def stage_df_pass(pipeline):
    """
    Convert DataFrame calls
    """
    # Ensure we have an IR and type information.
    assert pipeline.func_ir
    df_pass = HiFrames(pipeline.func_ir, pipeline.typingctx, pipeline.args, pipeline.locals)
    df_pass.run()

def stage_inline_pass(pipeline):
    """
    Inline function calls (to enable distributed pass analysis)
    """
    # Ensure we have an IR and type information.
    assert pipeline.func_ir
    inline_calls(pipeline.func_ir)

def add_hpat_stages(pipeline_manager, pipeline):
    pp = pipeline_manager.pipeline_stages['nopython']
    new_pp = []
    for (func,desc) in pp:
        if desc=='nopython frontend':
            new_pp.append((lambda:stage_inline_pass(pipeline), "inline funcs"))
            new_pp.append((lambda:stage_df_pass(pipeline), "convert DataFrames"))
            # run io pass in df pass to enable type inference
            #new_pp.append((lambda:stage_io_pass(pipeline), "replace IO calls"))
        if desc=='nopython mode backend':
            new_pp.append((lambda:stage_distributed_pass(pipeline), "convert to distributed"))
        new_pp.append((func,desc))
    pipeline_manager.pipeline_stages['nopython'] = new_pp

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
                                        func_ir.func_id.func.__globals__,
                                        block, i, py_func)
                        for block in new_blocks:
                            work_list.append(block)
                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break
