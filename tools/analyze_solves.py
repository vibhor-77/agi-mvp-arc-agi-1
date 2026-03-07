import json
import os
import argparse

def get_expr_depth(expr):
    """Simple estimate of expression depth based on parentheses nesting."""
    depth = 0
    max_depth = 0
    for char in expr:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth -= 1
    return max_depth

def analyze_reports(reports_dir="reports"):
    jsons = [f for f in os.listdir(reports_dir) if f.endswith(".json") and f.startswith("eval_")]
    
    all_solves = []
    
    for j in jsons:
        with open(os.path.join(reports_dir, j), "r") as f:
            try:
                data = json.load(f)
                results = data.get("results", [])
                for r in results:
                    if r.get("solved"):
                        all_solves.append({
                            "task": r.get("task"),
                            "expr": r.get("found_expr"),
                            "nodes": r.get("n_nodes"),
                            "depth": get_expr_depth(r.get("found_expr", ""))
                        })
            except:
                continue
                
    # Deduplicate by task (keep shallowest)
    best_solves = {}
    for s in all_solves:
        tid = s["task"]
        if tid not in best_solves or s["depth"] < best_solves[tid]["depth"]:
            best_solves[tid] = s
            
    print("\n🔬 SUCCESSFUL TRANSFORMATION COMPLEXITY ANALYSIS")
    print("-" * 80)
    print(f"| {'Task ID':10} | {'Depth':5} | {'Nodes':5} | {'Expression':40} |")
    print("-" * 80)
    
    depths = []
    for tid, s in sorted(best_solves.items()):
        print(f"| {tid:10} | {s['depth']:5} | {s['nodes']:5} | {s['expr'][:50]:40} |")
        depths.append(s['depth'])
        
    if depths:
        print("-" * 80)
        print(f"Mean Depth: {sum(depths)/len(depths):.2f}")
        print(f"Max Depth:  {max(depths)}")
    else:
        print("No successful solves found in JSON reports.")

if __name__ == "__main__":
    analyze_reports()
