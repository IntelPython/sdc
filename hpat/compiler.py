from __future__ import print_function, division, absolute_import

from .pio import PIO

def stage_io_pass(pipeline):
    """
    Convert IO calls
    """
    # Ensure we have an IR and type information.
    assert pipeline.func_ir
    io_pass = PIO(pipeline.func_ir, pipeline.locals)
    io_pass.run()

def add_hpat_stages(pipeline_manager, pipeline):
    pp = pipeline_manager.pipeline_stages['nopython']
    new_pp = []
    for (func,desc) in pp:
        if desc=='nopython frontend':
            new_pp.append((lambda:stage_io_pass(pipeline), "replace IO calls"))
        new_pp.append((func,desc))
    pipeline_manager.pipeline_stages['nopython'] = new_pp
