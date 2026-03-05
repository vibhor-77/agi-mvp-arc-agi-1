from core.search import BeamSearch, SearchConfig
from core.tree import Node
from domains.arc.primitives import registry

def fitness(n):
    return 10.0

cfg = SearchConfig(workers=4, generations=1, beam_size=10)
searcher = BeamSearch(fitness, list(registry.names()), config=cfg, n_vars=1)
searcher.run()
print("Success running with workers=4!")
