from core.tree import Node, make_node, make_leaf_var
from core.library import PrimitiveLibrary

def test_compression():
    lib = PrimitiveLibrary("tmp/test_lib.json")
    
    # 1. Define lib_op_1 as grot90(v0)
    op1_node = make_node("grot90", [make_leaf_var(0)])
    lib._add_to_library(op1_node)
    print(f"Learned lib_op_1: {lib.learned_ops['lib_op_1']['expr']}")
    
    # 2. Add an op that uses grot90(grot90(v0))
    # It should be compressed to lib_op_1(lib_op_1(v0))
    op2_node = make_node("grot90", [make_node("grot90", [make_leaf_var(0)])])
    lib._add_to_library(op2_node)
    
    op2_expr = lib.learned_ops["lib_op_2"]["expr"]
    print(f"Learned lib_op_2: {op2_expr}")
    
    if "lib_op_1" in op2_expr:
        print("✅ Hierarchical Compression SUCCESS")
    else:
        print("❌ Hierarchical Compression FAILED")

if __name__ == "__main__":
    test_compression()
