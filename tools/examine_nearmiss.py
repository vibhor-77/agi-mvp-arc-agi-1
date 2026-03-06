import sys
import os
from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.primitives import registry
from core.library import PrimitiveLibrary

def examine(task_id, data_dir):
    tasks = load_tasks_from_dir(data_dir)
    tasks = [t for t in tasks if t.name == task_id]
    if not tasks:
        print(f"Task {task_id} not found.")
        return
    
    lib = PrimitiveLibrary("models/arc_soft_v1.json")
    lib.load()
    
    cfg = BenchmarkConfig(
        beam_size=20, generations=20, task_workers=1, timeout_s=30
    )
    
    ops = registry.names(domain="arc")
    report = evaluate_tasks(
        tasks, ops, cfg, label=f"Examine {task_id}",
        learned_ops=lib.learned_ops,
        transition_matrix=lib.transition_matrix
    )
    
    tr = report.results[0]
    print(f"\n--- [ TASK {task_id} ] ---")
    print(f"Solved: {tr.solved}")
    print(f"Acc: {tr.test_acc:.4f}")
    if tr.best_tree:
        print(f"Expression: {tr.best_tree}")

if __name__ == "__main__":
    examine(sys.argv[1], sys.argv[2])
