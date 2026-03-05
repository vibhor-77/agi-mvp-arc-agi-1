from core.domain import Domain
from core.tree import Node
from core.search import SearchConfig, SearchResult
from core.primitives import registry

# Register the primitives module
import domains.zork.primitives 
from domains.zork.env import ZorkEnv

class ZorkDomain(Domain):
    """
    Symbolic policy for a Zork-style text adventure.
    The AST maps a state feature vector to an action string.
    """

    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self._ops = registry.names(domain="zork")
        self._primitives = {n: registry.get(n) for n in self._ops}

    def primitive_names(self) -> list[str]:
        return self._ops

    def n_vars(self) -> int:
        # [room_idx, has_key, is_door_locked]
        return 3

    def fitness(self, tree: Node) -> float:
        # Reinitialize environment to start fresh per evaluation
        env = ZorkEnv()
        total_reward = 0
        state_features = env.state.as_features()
        
        try:
            for _ in range(self.max_steps):
                if env.state.done:
                    break
                    
                # The tree returns an action string given the state features
                action_str = str(tree.eval([state_features], self._primitives))
                
                _, state_features, score, done = env.step(action_str)
                total_reward = score 
                
            # If done, give extra bonus for finishing in fewer steps
            if env.state.done:
                total_reward += 10.0 / (_ + 1)
        except Exception:
            # Tree failed to evaluate to a valid action
            return float('inf')

        # Lower is better, so negate reward. Add small complexity penalty.
        return -total_reward + 0.05 * tree.size()
        
    def description(self) -> str:
        return "Zork Text Adventure (Symbolic NLP MVP)"
        
    def solve(self, config: SearchConfig | None = None, **kwargs) -> SearchResult:
        from core.search import BeamSearch
        op_arities = {name: registry.arity(name) for name in self.primitive_names()}
        
        searcher = BeamSearch(
            fitness_fn=self.fitness,
            op_list=self.primitive_names(),
            n_vars=1,  # Passing the entire state vector as a single input
            config=config or SearchConfig(),
            op_arities=op_arities,
        )
        return searcher.run()
        
    def on_result(self, result: SearchResult) -> None:
        print(f"  Best policy     : {result.best_tree}")
        print(f"  Best score      : {-result.best_fitness + 0.05 * result.best_tree.size():.2f}")
        print(f"  Tree size       : {result.best_tree.size()} nodes")

