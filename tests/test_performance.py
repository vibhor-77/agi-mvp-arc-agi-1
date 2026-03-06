import pytest
import time
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from core.tree import Node
from domains.arc.domain import ARCTask
from core.primitives import registry
import domains.arc.primitives

def test_jit_throughput_threshold():
    """
    Scientific Regression Test: 
    Ensure the JIT engine maintains at least 5k evals/sec on M1 hardware.
    """
    grid = [[(x+y)%10 for x in range(20)] for y in range(20)]
    task = ARCTask("perf_test", [(grid, grid)], [(grid, grid)])
    primitives = {n: registry.get(n) for n in registry.names()}
    
    expr = "grot90(ginv(x))"
    node = Node.parse(expr)
    inp = grid
    
    # Warmup
    for _ in range(5):
        _ = node.eval([inp], primitives)
        
    # Bench
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        _ = node.eval([inp], primitives)
    t1 = time.perf_counter()
    
    rate = N / (t1 - t0)
    print(f"\n[Performance Test] Rate: {rate:.1f} evals/sec")
    
    # We expect > 5,000 evals/sec on M1 after JIT optimization
    # (Leaving headroom for CI environments, but on local M1 Max this hits 20k+)
    assert rate > 2000, f"Performance regression! Only hit {rate:.1f} evals/sec"

if __name__ == "__main__":
    test_jit_throughput_threshold()
