import hpat

def count_array_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.REP for v in vals])

def count_parfor_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.REP for v in vals])

def count_parfor_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD for v in vals])

def count_array_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD for v in vals])

def count_parfor_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])

def count_array_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v==Distribution.OneD_Var for v in vals])

def dist_IR_contains(*args):
    return sum([(s in hpat.distributed.fir_text) for s in args])
