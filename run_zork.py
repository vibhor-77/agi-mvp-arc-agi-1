#!/usr/env/bin python3
from domains.zork.domain import ZorkDomain
from core.search import SearchConfig

def main():
    print("="*50)
    print(" ZORK SYMBOLIC NLP MVP ")
    print("="*50)
    
    domain = ZorkDomain()
    print(f"Primitives available ({len(domain.primitive_names())}):")
    print(", ".join(domain.primitive_names()))
    print("-" * 50)
    
    cfg = SearchConfig(
        beam_size=200,
        offspring=100,
        generations=50,
        workers=4,
        verbose=True,
        log_interval=2,
        converge_threshold=-float("inf") # Disable early stop for negative score
    )
    
    result = domain.solve(config=cfg)
    domain.on_result(result)

if __name__ == "__main__":
    main()
