from core.primitives import registry

# Property extracts
def z_room(x: list[float]) -> float: return x[0]
def z_has_key(x: list[float]) -> float: return x[1]
def z_is_locked(x: list[float]) -> float: return x[2]

# Boolean checks returning 1.0 or 0.0
def z_is_cave(room_idx: float) -> float: return 1.0 if room_idx == 1.0 else 0.0
def z_is_forest(room_idx: float) -> float: return 1.0 if room_idx == 0.0 else 0.0
def z_is_true(val: float) -> float: return 1.0 if val > 0.5 else 0.0
def z_is_false(val: float) -> float: return 1.0 if val <= 0.5 else 0.0

# Conditionals
def z_if(cond: float, action1: str, action2: str) -> str:
    return action1 if cond > 0.5 else action2

# Actions (ignoring args, returning strings)
def z_act_north(x: list[float]) -> str: return "go north"
def z_act_south(x: list[float]) -> str: return "go south"
def z_act_take_key(x: list[float]) -> str: return "take key"
def z_act_unlock(x: list[float]) -> str: return "unlock door"

# Registration
registry.register("z_room", z_room, domain="zork", arity=1)
registry.register("z_has_key", z_has_key, domain="zork", arity=1)
registry.register("z_is_locked", z_is_locked, domain="zork", arity=1)
registry.register("z_is_cave", z_is_cave, domain="zork", arity=1)
registry.register("z_is_forest", z_is_forest, domain="zork", arity=1)
registry.register("z_is_true", z_is_true, domain="zork", arity=1)
registry.register("z_is_false", z_is_false, domain="zork", arity=1)

registry.register("z_if", z_if, domain="zork", arity=3)

registry.register("z_act_north", z_act_north, domain="zork", arity=1)
registry.register("z_act_south", z_act_south, domain="zork", arity=1)
registry.register("z_act_take_key", z_act_take_key, domain="zork", arity=1)
registry.register("z_act_unlock", z_act_unlock, domain="zork", arity=1)
