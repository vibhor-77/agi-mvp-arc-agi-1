from core.tree import Node
def test_parse():
    n = Node.parse("g_overlay(gmap_rot90(x), x)")
    assert str(n) == "g_overlay(gmap_rot90(x), x)"
    assert n.op == "g_overlay"
    assert n.children[0].op == "gmap_rot90"
    assert n.children[0].children[0].var_idx == 0
    assert n.children[1].var_idx == 0

    n2 = Node.parse("x")
    assert str(n2) == "x"
    assert n2.var_idx == 0

    n3 = Node.parse("1.5")
    assert str(n3) == "1.5"
    assert n3.const == 1.5
