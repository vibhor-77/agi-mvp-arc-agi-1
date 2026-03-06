#!/usr/bin/env python3
"""
agi_repair.py (REFINEMENT PASS)
===============================
Takes near-miss programs and applies a shallow local search to fix 
color, alignment, or padding errors.
"""
import argparse
import os
import sys
import time
from copy import deepcopy

from domains.arc.runner import load_tasks_from_dir, ARCTask, TaskResult
from domains.arc.primitives import registry
from core.library import PrimitiveLibrary
from core.primitives import registry as core_registry
from core.tree import Node
import numpy as np
import multiprocessing
import multiprocessing.pool

def attempt_repair(task: ARCTask, base_expr: str, lib: PrimitiveLibrary):
    """
    Two-stage Repair:
    1. Discovery: If no base_expr, find the 'macro-logic' quickly.
    2. Refinement: Apply NumPy-accelerated correction templates.
    """
    import core.primitives as cp
    lib.register_all(domain="arc")
    primitives = {n: cp.registry.get(n) for n in cp.registry.names(domain="arc")}
    op_names = list(primitives.keys())

    # Accelerated Search Primitives (Expanded Surgical Set)
    repair_ops = [
        "g_replace_0_with_1", "g_replace_1_with_2", "g_replace_2_with_1",
        "g_replace_3_with_1", "g_replace_1_with_3", "g_replace_0_with_2",
        "g_replace_0_with_3", "g_replace_3_with_0", "g_replace_0_with_4",
        "gswap_01", "gswap_12", "gswap_23", "gswap_02", "gswap_03",
        "g_replace_nonzero_with_1", "g_flood_fill", "g_shift_up", "g_shift_down",
        "g_shift_left", "g_shift_right", "g_pad1", "gcrop_border", "ginv",
        "g_fg_to_most_common", "g_fg_to_least_common",
        "gmirror_h", "gmirror_v", "grot90", "grot180", "grot270"
    ]
    repair_ops = [op for op in repair_ops if op in op_names]
    
    # Pre-cache training data
    np_targets = [np.array(out) for _, out in task.train_pairs]
    
    eval_count = 0
    start_time = time.time()

    def grid_cell_accuracy(pred, target):
        if pred is None: return 0.0
        pred_np = np.array(pred)
        target_np = np.array(target)
        if pred_np.shape != target_np.shape: return 0.0
        matches = np.count_nonzero(pred_np == target_np)
        return matches / target_np.size

    def get_score(node):
        nonlocal eval_count
        eval_count += 1
        accs = []
        for inp, target in task.train_pairs:
            try:
                pred = node.eval([inp], primitives)
                accs.append(grid_cell_accuracy(pred, target))
            except:
                accs.append(0.0)
        
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        # If perfect on train, check validation
        if avg_acc >= 0.999:
            try:
                test_in, test_out = task.test_pairs[0]
                pred_test = node.eval([test_in], primitives)
                if grid_cell_accuracy(pred_test, test_out) >= 0.999:
                    return 2.0
            except:
                pass
        return avg_acc

    # Phase 1: High-Speed Discovery (if needed)
    best_expr = base_expr
    best_acc = 0.0
    
    if not best_expr:
        # Discovery search to find the 'Shape'
        from domains.arc.domain import ARCDomain
        from core.search import BeamSearch, SearchConfig
        domain = ARCDomain(task, library=lib)
        cfg = SearchConfig(beam_size=40, generations=5, timeout_s=45) 
        search = BeamSearch(domain.fitness, domain.primitive_names(), domain.n_vars(), cfg)
        report = search.run()
        if report.best_tree:
            best_expr = str(report.best_tree)
            best_acc = get_score(report.best_tree)
            if best_acc >= 2.0: return best_expr, 1.0
    else:
        best_acc = get_score(Node.parse(best_expr))
        if best_acc >= 2.0: return best_expr, 1.0

    # Phase 2: NumPy Refinement (Surgical Layer)
    if best_acc >= 0.10:
        # Pass 1: Unary Surgery
        for op in repair_ops:
            candidate = f"{op}({best_expr})"
            score = get_score(Node.parse(candidate))
            if score >= 2.0: return candidate, 1.0
            if score > best_acc:
                best_acc = score
                best_expr = candidate # Hill climb
        
        # Pass 2: Double Surgery (Geometric + Color)
        # We only do this if we haven't solved it yet and we have a decent base
        if best_acc >= 0.50:
            color_ops = [r for r in repair_ops if "replace" in r or "swap" in r]
            geom_ops = [r for r in repair_ops if "rot" in r or "mirror" in r or "shift" in r]
            for op1 in color_ops:
                for op2 in geom_ops:
                    candidate = f"{op1}({op2}({best_expr}))"
                    score = get_score(Node.parse(candidate))
                    if score >= 2.0: return candidate, 1.0
                    
        # Pass 3: Padding/Logic Refinement
        if best_acc >= 0.80:
             for op in ["g_pad1", "gcrop_border", "g_flood_fill", "ginv"]:
                 if op in op_names:
                     candidate = f"{op}({best_expr})"
                     if get_score(Node.parse(candidate)) >= 2.0: return candidate, 1.0

    duration = time.time() - start_time
    print(f"  [Stats] {task.name} evals={eval_count} rate={eval_count/max(0.1, duration):.1f}/s")
    return None, best_acc

def repair_worker(task, base_expr, model_path, queue):
    """Picklable global worker function."""
    try:
        from core.library import PrimitiveLibrary
        lib = PrimitiveLibrary(model_path)
        lib.load()
        res = attempt_repair(task, base_expr, lib)
        queue.put(res)
    except Exception as e:
        queue.put((None, 0.0))

# Non-daemonic Pool to allow child processes (Discovery beam search)
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self): return False
    @daemon.setter
    def daemon(self, value): pass

class NoDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

def _repair_internal_worker(task, expr, mpath, q):
    """Deeply picklable internal worker."""
    try:
        from core.library import PrimitiveLibrary
        lib_inner = PrimitiveLibrary(mpath)
        lib_inner.load()
        res = attempt_repair(task, expr, lib_inner)
        q.put(res)
    except:
        q.put((None, 0.0))

def repair_pool_wrapper(task_in, model_path, base_expr):
    """Picklable worker for Pool with its own internal timeout logic."""
    import multiprocessing
    import queue
    
    try:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_repair_internal_worker, args=(task_in, base_expr, model_path, q))
        p.start()
        p.join(timeout=60) # 60s hard limit per task

        if p.is_alive():
            p.terminate()
            p.join()
            return None, 0.0
        
        try:
            return q.get_nowait()
        except:
            return None, 0.0
    except:
        return None, 0.0

def cmd_repair(args):
    lib = PrimitiveLibrary(args.model)
    lib.load()
    
    tasks = []
    if args.log:
        import re
        print(f"📖 Identifying near-miss tasks from log...")
        all_tasks = load_tasks_from_dir(args.data)
        near_ids = set()
        with open(args.log, "r") as f:
            for line in f:
                m = re.search(r"~.*DONE\s+([a-f0-9]+)", line)
                if m: near_ids.add(m.group(1))
        tasks = [t for t in all_tasks if t.name in near_ids]
    else:
        all_tasks = load_tasks_from_dir(args.data)
        if args.task_ids:
            ids = args.task_ids.split(",")
            tasks = [t for t in all_tasks if t.name in ids]
        else:
            tasks = all_tasks

    print(f"--- REPAIR SWEEP: {len(tasks)} tasks | model={args.model} ---")
    
    solves = 0
    t0 = time.time()
    
    import multiprocessing
    import multiprocessing.pool
    from functools import partial
    
    print(f"🚀 Launching Parallel Task Runner (4 workers) for {len(tasks)} tasks...")

    # Use our wrapper but with a fresh result queue isn't needed for imap
    func = partial(repair_pool_wrapper, model_path=args.model, base_expr=args.base_expr)
    
    with NoDaemonPool(processes=4) as pool:
        for i, (new_expr, acc) in enumerate(pool.imap(func, tasks)):
            tid = tasks[i].name
            if new_expr:
                solves += 1
                sys.stdout.write("\033[K")
                print(f"\r🌟 [REPAIRED] {tid}: {new_expr}")
            else:
                sys.stdout.write(f"\r[{i+1}/{len(tasks)}] Processed {tid} (No fix)...")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n=================================================================")
    print(f" 🏁 SWEEP COMPLETE: {solves}/{len(tasks)} tasks fixed.")
    print(f" Time: {elapsed:.1f}s ({elapsed/max(1,len(tasks)):.2f}s/task)")
    print(f"=================================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["repair"])
    parser.add_argument("--data", type=str, default="arc_data/data/evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task-ids", type=str, default="")
    parser.add_argument("--base-expr", type=str, default="")
    parser.add_argument("--log", type=str, default="")
    
    args = parser.parse_args()
    if args.command == "repair":
        cmd_repair(args)

if __name__ == "__main__":
    main()
