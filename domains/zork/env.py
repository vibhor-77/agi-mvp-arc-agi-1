from dataclasses import dataclass, field

@dataclass
class ZorkState:
    room: str = "forest"
    inventory: set[str] = field(default_factory=set)
    door_locked: bool = True
    score: int = 0
    done: bool = False
    
    def as_features(self) -> list[float]:
        # [room_idx, has_key, is_door_locked]
        room_map = {"forest": 0.0, "cave": 1.0, "castle": 2.0}
        has_key = 1.0 if "key" in self.inventory else 0.0
        return [room_map.get(self.room, 0.0), has_key, float(self.door_locked)]

class ZorkEnv:
    def __init__(self):
        self.state = ZorkState()
        
    def step(self, action: str) -> tuple[str, list[float], int, bool]:
        if self.state.done:
            return "Game over.", self.state.as_features(), self.state.score, True
            
        observation = ""
        if action == "go north":
            if self.state.room == "forest":
                self.state.room = "cave"
                observation = "You enter a dark cave. You see a key."
                # Removed +1 score exploit here
            elif self.state.room == "cave":
                if self.state.door_locked:
                    observation = "The castle door is locked."
                else:
                    self.state.room = "castle"
                    observation = "You enter the castle! You win!"
                    self.state.score += 20
                    self.state.done = True
            else:
                observation = "Can't go north here."
                
        elif action == "go south":
            if self.state.room == "cave":
                self.state.room = "forest"
                observation = "You are in a forest."
            else:
                observation = "Can't go south here."
                
        elif action == "take key":
            if self.state.room == "cave" and "key" not in self.state.inventory:
                self.state.inventory.add("key")
                observation = "You took the key."
                self.state.score += 5
            else:
                observation = "No key here."
                
        elif action == "unlock door":
            if self.state.room == "cave" and self.state.door_locked:
                if "key" in self.state.inventory:
                    self.state.door_locked = False
                    observation = "Door unlocked!"
                    self.state.score += 5
                else:
                    observation = "You don't have a key."
            else:
                observation = "Nothing to unlock."
        else:
            observation = "Unknown action."
            
        return observation, self.state.as_features(), self.state.score, self.state.done
