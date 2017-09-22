from __future__ import print_function, division, absolute_import

# from .pio import PIO
from .distributed import DistributedPass
from .hiframes import HiFrames
import numba
import numba.compiler
from numba import ir_utils, ir
from numba.targets.registry import CPUDispatcher
from numba.ir_utils import (mk_unique_var, add_offset_to_labels,
                            get_name_var_table, replace_vars)

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
    call_table, _ = ir_utils.get_call_table(func_ir.blocks)
    for label, block in func_ir.blocks.items():
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Assign):
                rhs = stmt.value
                if isinstance(rhs, ir.Expr) and rhs.op=='call':
                    func = rhs.func.name
                    if (func in call_table and call_table[func]
                            and isinstance(call_table[func][0], CPUDispatcher)):
                        py_func = call_table[func][0].py_func
                        inline_calls_inner(func_ir, block, stmt, i, py_func)
                        return  # inline_calls_inner will call back recursively

def inline_calls_inner(func_ir, block, stmt, i, py_func):
    call_expr = stmt.value
    scope = block.scope
    callee_ir = numba.compiler.run_frontend(py_func)

    # relabel callee_ir by adding an offset
    max_label = max(func_ir.blocks.keys())
    callee_blocks = add_offset_to_labels(callee_ir.blocks, max_label+1)
    callee_ir.blocks = callee_blocks
    min_label = min(callee_blocks.keys())
    max_label = max(callee_blocks.keys())

    #  init _max_label global in ir_utils before using next_label()
    ir_utils._max_label = max_label

    # rename all variables in callee blocks
    var_table = get_name_var_table(callee_ir.blocks)
    new_var_dict = {}
    for name, var in var_table.items():
        new_var = scope.define(mk_unique_var(var.name), loc=var.loc)
        new_var_dict[name] = new_var
    replace_vars(callee_ir.blocks, new_var_dict)

    # replace callee arguments
    args = list(call_expr.args)
    # TODO: replace defaults (add to args)
    _replace_args(callee_ir.blocks, args)

    # split caller blocks into two
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = ir_utils.next_label()
    func_ir.blocks[new_label] = new_block
    block.body = block.body[:i]
    block.body.append(ir.Jump(min_label, stmt.loc))

    # replace Return with assignment to LHS
    _replace_returns(callee_ir.blocks, stmt.target, new_label)

    # insert all new blocks
    for label, bl in callee_ir.blocks.items():
        func_ir.blocks[label] = bl

    # run inline_calls recursively to transform other calls
    inline_calls(func_ir)
    return

def _replace_args(blocks, args):
    """
    Replace ir.Arg(...) with real arguments from call site
    """
    for label, block in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                idx = stmt.value.index
                assert(idx < len(args))
                stmt.value = args[idx]

def _replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for label, block in blocks.items():
        casts = []
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Return):
                assert(i + 1 == len(block.body))
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))
                # remove cast of the returned value
                for cast in casts:
                    if cast.target.name == stmt.value.name:
                        cast.value = cast.value.value
            elif (isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == 'cast'):
                casts.append(stmt)
