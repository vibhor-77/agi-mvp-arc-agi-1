"""
domains/arc — ARC-AGI grid transformation domain.

Quick start
-----------
    from domains.arc.domain    import ARCDomain, ARCTask
    from domains.arc.benchmark import get_benchmark
    from domains.arc.runner    import run_benchmark

    tasks   = get_benchmark()
    results = run_benchmark(tasks)
"""
from .domain    import ARCDomain, ARCTask, grid_cell_accuracy, is_exact_match
from .benchmark import build_benchmark, get_benchmark
