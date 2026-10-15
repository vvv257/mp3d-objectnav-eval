from .converters import convert_command


class ActionExecutor:
    """Stateless per-command execution: no pending actions cross episode boundaries."""

    def __init__(self, backend, protocol):
        self.backend = backend
        self.protocol = protocol

    def convert(self, command):
        return convert_command(command, self.protocol, getattr(self.backend, "action_parameters", {}))

    def execute(self, conversion):
        for action in conversion.actions:
            if self.backend.episode_over:
                return
            self.protocol.validate_action(action)
            yield self.backend.step(action)
