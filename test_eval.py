from core.tree import Node
from core.primitives import registry
import domains.arc.primitives

def test_if():
    print("Testing g_if...")
    prims = {n: registry.get(n) for n in registry.names()}
    
    # Truthy eval
    truthy_grid = [[1, 2], [0, 0]]
    node1 = Node.parse("g_if(x, 10, 20)")
    assert node1.eval([truthy_grid], prims) == 10
    
    # Falsy eval
    falsy_grid = [[0, 0], [0, 0]]
    assert node1.eval([falsy_grid], prims) == 20
    print("g_if works!")

def test_while():
    # To test while loop, we need a grid operation that converges to zeros
    # e.g gkeep_color1 ... wait, what primitive gradually edits grid to 0?
    # Let's use `gmap_rot90` to see if it loops. But rot90 never hits 0.
    # What about a custom primitive for testing?
    
    def dec_first(g):
        import copy
        out = copy.deepcopy(g)
        if out[0][0] > 0:
            out[0][0] -= 1
        return out
        
    registry.register("test_dec", dec_first, domain="arc", arity=1)
    prims = {n: registry.get(n) for n in registry.names()}
    
    grid = [[5]]
    # While grid has non-zero elements, decrement it
    node = Node.parse("g_while(x, test_dec(x))")
    
    res = node.eval([grid], prims)
    print(f"While Loop Initial: [[5]], Result: {res}")
    
test_if()
test_while()
