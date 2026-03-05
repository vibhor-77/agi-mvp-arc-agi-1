#!/usr/bin/env python3
import sys
import os
import time
import json

# Setup Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.primitives import registry

def test_finalists():
    tasks = load_tasks_from_dir("arc_data/data/training")[:20]
    active_ops = registry.names(domain="arc") or []
    
    # Just the 2 primary breadth-heavy candidates
    finalists = [
        (10, 40, 25),
        (20, 20, 25)
    ]
    
    results = []
    for b, o, g in finalists:
        print(f"\n🚀 Testing Candidate: Beam {b}, Offspring {o}, Gen {g}")
        cfg = BenchmarkConfig(
            beam_size=b, offspring=o, generations=g,
            task_workers=3, workers=1, baseline_only=True, verbose=True, seed=42
        )
        t0 = time.time()
        report = evaluate_tasks(tasks=tasks, op_subset=active_ops, cfg=cfg, label=f"Finalist_{b}_{o}_{g}")
        elapsed = time.time() - t0
        total_evals = sum(r.n_evals for r in report.results)
        
        res = {
            "params": {"beam": b, "offspring": o, "generations": g},
            "metrics": {
                "n_solved": report.n_solved,
                "n_evals": total_evals,
                "roc_math": (report.n_solved/20.0 * 1e6) / (total_evals + 1),
                "time_s": elapsed
            }
        }
        print("\n🏆 CONFIG RESULT:")
        print(json.dumps(res, indent=2))
        results.append(res)
    
    with open("docs/finalist_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_finalists()
