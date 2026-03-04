import unittest
import os
import json
from core.library import PrimitiveLibrary
from core.tree import Node
from core.primitives import registry

class TestPrimitiveLibrary(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_library.json"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        # ensure some nodes are registered for the test to work
        registry.register("op_a", lambda x: x, domain="test_lib", overwrite=True)
        registry.register("op_b", lambda x: x, domain="test_lib", overwrite=True)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def test_library_initialization(self):
        lib = PrimitiveLibrary(self.test_file)
        self.assertEqual(lib.learned_ops, {})
        self.assertEqual(lib.transition_matrix, {})
        
    def test_library_extract_from_tasks(self):
        lib = PrimitiveLibrary(self.test_file)
        # Create some identical subtrees across different tasks to force compression
        # Tree 1: op_a(op_b(X0))
        t1 = Node("op_a", [Node("op_b", [Node("X0")])])
        # Tree 2: op_a(op_b(X0))
        t2 = Node("op_a", [Node("op_b", [Node("X0")])])
        
        trees = {"task1": t1, "task2": t2}
        lib.extract_from_tasks(trees, min_size=2, min_tasks=2)
        
        # Should have found op_a(op_b(_)) as a reusable subtree
        self.assertGreater(len(lib.learned_ops), 0)
        op_name = list(lib.learned_ops.keys())[0]
        self.assertIn("lib_op_", op_name)
        
    def test_library_save_and_load(self):
        lib = PrimitiveLibrary(self.test_file)
        node = Node("op_a", [Node("X0")])
        lib.learned_ops = {"lib_op_99": {"expr": "op_a(X0)", "node": node, "size": 2, "arity": 1}}
        lib.transition_matrix = {"op_a": {"X0": 1.0}}
        lib.save()
        
        self.assertTrue(os.path.exists(self.test_file))
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
            
        self.assertIn("library", data)
        self.assertIn("lib_op_99", data["library"])
        self.assertIn("op_a", data["transitions"])

    def test_library_registration(self):
        lib = PrimitiveLibrary(self.test_file)
        node = Node("op_a", [Node("X0")])
        lib.learned_ops = {"lib_op_99": {"expr": "op_a(X0)", "node": node, "size": 2, "arity": 1}}
        lib.register_all(domain="test_lib")
        self.assertIn("lib_op_99", registry.names(domain="test_lib"))

if __name__ == "__main__":
    unittest.main()
