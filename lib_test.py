from core.library import PrimitiveLibrary
from core.tree import Node
import traceback

def test_save_load():
    try:
        lib = PrimitiveLibrary("test_lib.json")
        node1 = Node.parse("g_if(x, gmap_fill(x), grot90(x))")
        lib._add_to_library(node1)
        
        lib.save()
        
        lib2 = PrimitiveLibrary("test_lib.json")
        lib2.load()
        
        print("Loaded ops:")
        for name, meta in lib2.learned_ops.items():
            print(f"{name}: {meta['expr']} (arity {meta['arity']})")
            print(f"  Parsed Node: {meta['node']}")
            
    except Exception as e:
        traceback.print_exc()

test_save_load()
