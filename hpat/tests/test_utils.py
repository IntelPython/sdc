import hpat


def count_array_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_REPs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.REP for v in vals])


def count_parfor_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_array_OneDs():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD for v in vals])


def count_parfor_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.parfor_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def count_array_OneD_Vars():
    from hpat.distributed import Distribution
    vals = hpat.distributed.dist_analysis.array_dists.values()
    return sum([v == Distribution.OneD_Var for v in vals])


def dist_IR_contains(*args):
    return sum([(s in hpat.distributed.fir_text) for s in args])


@hpat.jit
def get_rank():
    return hpat.distributed_api.get_rank()


@hpat.jit
def get_start_end(n):
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    start = hpat.distributed_api.get_start(n, n_pes, rank)
    end = hpat.distributed_api.get_end(n, n_pes, rank)
    return start, end

# Decorator to repeat test execution
def repeat_test(test):
    def repeat_test_wrapper(*args, **kwargs):
        import os

        repeat_number = 2
        if 'REPEAT_TEST_NUMBER' in os.environ:
            repeat_number = int(os.environ['REPEAT_TEST_NUMBER'])

        print('Repeat this test {} times'.format(repeat_number))
        for _ in range(repeat_number):
            test(*args, **kwargs)

    return repeat_test_wrapper
