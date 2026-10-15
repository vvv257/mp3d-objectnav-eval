from collections.abc import Mapping
from .base import BaseAdapter


class MotionActionAdapter(BaseAdapter):
    def __init__(self, agent, rotate_field="degrees", move_field="meters", observation_adapter=None):
        super().__init__(agent, observation_adapter)
        if not isinstance(rotate_field, str) or not isinstance(move_field, str) or not rotate_field or not move_field or rotate_field == move_field:
            raise ValueError("Motion fields must be distinct nonempty strings")
        self.rotate_field, self.move_field = rotate_field, move_field

    def convert_action(self, action):
        if isinstance(action, Mapping):
            if set(action) == {self.rotate_field}:
                return {"type": "rotate", "degrees": action[self.rotate_field]}
            if set(action) == {self.move_field}:
                return {"type": "move", "meters": action[self.move_field]}
        return action
