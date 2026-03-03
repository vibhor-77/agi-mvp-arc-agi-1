"""
core — beam search engine and primitive registry.

Public API
----------
    from core.primitives import registry, PrimitiveRegistry
    from core.tree      import Node, random_tree, mutate, crossover
    from core.search    import BeamSearch, SearchConfig, SearchResult
    from core.domain    import Domain
"""
from .primitives import registry, PrimitiveRegistry
from .tree       import Node, random_tree, mutate, crossover, make_node, make_leaf_var, make_leaf_const
from .search     import BeamSearch, SearchConfig, SearchResult
from .domain     import Domain

__all__ = [
    "registry", "PrimitiveRegistry",
    "Node", "random_tree", "mutate", "crossover",
    "make_node", "make_leaf_var", "make_leaf_const",
    "BeamSearch", "SearchConfig", "SearchResult",
    "Domain",
]
