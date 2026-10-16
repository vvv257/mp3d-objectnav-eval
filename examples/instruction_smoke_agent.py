from typing import Any, Mapping


class InstructionSmokeAgent:
    """Minimal structural Agent that proves language observations are delivered."""

    def __init__(self) -> None:
        self.steps = 0
        self.last_instruction = ""

    def reset(self, episode: Any) -> None:
        self.steps = 0
        self.last_instruction = ""

    def act(self, observation: Mapping[str, Any]) -> str:
        instruction = observation.get("instruction")
        text = instruction.get("text") if isinstance(instruction, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("InstructionSmokeAgent requires non-empty instruction.text")

        self.last_instruction = text
        self.steps += 1
        return "stop" if self.steps > 2 else "move_forward"
