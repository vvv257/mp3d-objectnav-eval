class AdapterConversionError(ValueError):
    def __init__(self, stage, original):
        super().__init__(str(original))
        self.stage = stage


class BaseAdapter:
    """Explicit wrapper; only receives the evaluator's permitted observations."""

    def __init__(self, agent, observation_adapter=None):
        self.agent = agent
        self.observation_adapter = observation_adapter

    def reset(self, episode):
        self.agent.reset(episode)

    def convert_action(self, action):
        return action

    def act(self, observation):
        if self.observation_adapter is not None:
            try:
                observation = self.observation_adapter.convert(observation)
            except Exception as exc:
                raise AdapterConversionError("adapter.observation", exc) from exc
        action = self.agent.act(observation)
        try:
            return self.convert_action(action)
        except Exception as exc:
            raise AdapterConversionError("adapter.action", exc) from exc
