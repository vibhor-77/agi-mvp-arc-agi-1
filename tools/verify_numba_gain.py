import os
import time
import numpy as np
import subprocess
import sys

# Ensure we can import the project
sys.path.insert(0, os.getcwd())

def run_bench(jit_enabled=True):
    env = os.environ.copy()
    if not jit_enabled:
        env["NUMBA_DISABLE_JIT"] = "1"
    
    # We'll run a specific benchmark script or a snippet
    code = """
import time
import numpy as np
import numba
from domains.arc.domain import grid_cell_accuracy
from domains.arc.primitives import _njit_label_any_fg

# Warmup 
g1 = np.random.randint(0, 10, (30, 30)).astype(np.int16)
g2 = np.random.randint(0, 10, (30, 30)).astype(np.int16)
grid_cell_accuracy(g1, g2)
_njit_label_any_fg(g1)

# Benchmark grid_cell_accuracy
N = 100000
t0 = time.time()
for _ in range(N):
    grid_cell_accuracy(g1, g2)
t1 = time.time()
print(f"grid_cell_accuracy: { (t1-t0)/N * 1e6:.4f} us/op")

# Benchmark labeling
N_L = 2000
t0 = time.time()
for _ in range(N_L):
    _njit_label_any_fg(g1)
t1 = time.time()
print(f"labeling: { (t1-t0)/N_L * 1e3:.4f} ms/op")
"""
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return res.stdout

print("Checking Numba Performance Delta...")
print("-" * 40)
print("RUNNING WITH JIT ENABLED:")
out_jit = run_bench(jit_enabled=True)
print(out_jit)

print("RUNNING WITH JIT DISABLED:")
out_no_jit = run_bench(jit_enabled=False)
print(out_no_jit)
