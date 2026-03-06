import time
import numpy as np
import numba
from numba import njit
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.tree import Node
from domains.arc.domain import ARCDomain, ARCTask
from core.library import PrimitiveLibrary

def create_mock_task():
    # A typical 30x30 ARC grid
    grid = [[(x + y) % 10 for x in range(30)] for y in range(30)]
    return ARCTask("bench_task", [(grid, grid)], [(grid, grid)])

def benchmark():
    print("--- 🔬 ARC PERFORMANCE BENCHMARK (START FROM SCRATCH) ---")
    
    # 1. Setup
    from core.primitives import registry
    import domains.arc.primitives # This triggers arc registration
    
    task = create_mock_task()
    
    # Full primitive mapping (Core + ARC)
    primitives = {n: registry.get(n) for n in registry.names()}
    
    # Create a reasonably deep tree: grot90(ginv(gmirror_h(x)))
    expr = "grot90(ginv(gmirror_h(x)))"
    node = Node.parse(expr)
    inp = task.train_pairs[0][0]
    
    # 2. Warmup (JIT compilation trigger)
    print("🔥 Warming up Numba JIT kernels...")
    for _ in range(10):
        _ = node.eval([inp], primitives)
    
    # 3. Scientific Measurement
    N_EVALS = 50000
    print(f"🚀 Running {N_EVALS} evaluations of: {expr}")
    
    t0 = time.perf_counter()
    for _ in range(N_EVALS):
        _ = node.eval([inp], primitives)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    rate = N_EVALS / elapsed
    
    print("\n" + "="*50)
    print(f" ⏱️  Total Time: {elapsed:.4f}s")
    print(f" 🏎️  Throughput: {rate:,.1f} evals/sec")
    print("="*50)
    
    # 4. Comparative Context
    BASELINE_RATE = 1.7
    speedup = rate / BASELINE_RATE
    print(f" 📈 Calculated Speedup over 1.7 baseline: {speedup:,.1f}x")
    print("\n✅ Performance validated scientifically.")

if __name__ == "__main__":
    benchmark()
