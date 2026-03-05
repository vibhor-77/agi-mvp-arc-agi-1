from core.tree import Node
try:
    print(Node.parse("g_while(x, 10)"))
except Exception as e:
    import traceback
    traceback.print_exc()
