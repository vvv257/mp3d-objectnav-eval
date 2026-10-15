import math


class ObservationAdapter:
    """Explicit per-sensor transforms. No environment reference or sensor queries."""

    def __init__(self, rules):
        if not isinstance(rules, dict):
            raise ValueError("Observation rules must be a mapping")
        permitted = {"rename", "scale", "offset", "dtype", "transpose", "resize", "clip"}
        for rule in rules.values():
            if not isinstance(rule, dict) or set(rule) - permitted:
                raise ValueError("Unknown observation conversion rule")
            for key in ("scale", "offset"):
                if key in rule and (isinstance(rule[key], bool) or not isinstance(rule[key], (int, float)) or not math.isfinite(rule[key])):
                    raise ValueError("Observation scale/offset must be finite")
        self.rules = rules

    def convert(self, observation):
        result = {}
        for source, value in observation.items():
            rule = self.rules.get(source, {})
            target = rule.get("rename", source)
            if not isinstance(target, str) or target in result:
                raise ValueError("Observation rename must be unique strings")
            if set(rule) - {"rename"}:
                import numpy as np
                value = np.array(value, copy=True)
                if "resize" in rule:
                    size = rule["resize"]
                    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size) or value.ndim < 2:
                        raise ValueError("resize expects positive [height, width] on an image")
                    h, w = size
                    rows = np.arange(h) * value.shape[0] // h
                    cols = np.arange(w) * value.shape[1] // w
                    value = value[rows[:, None], cols[None, :]]
                if "scale" in rule or "offset" in rule:
                    value = value * rule.get("scale", 1) + rule.get("offset", 0)
                if "clip" in rule:
                    low, high = rule["clip"]
                    if not math.isfinite(low) or not math.isfinite(high) or low > high:
                        raise ValueError("Invalid observation clip range")
                    value = np.clip(value, low, high)
                if "transpose" in rule:
                    value = value.transpose(rule["transpose"])
                if "dtype" in rule:
                    dtype = np.dtype(rule["dtype"])
                    if dtype.kind not in "biuf":
                        raise ValueError("Observation dtype must be numeric")
                    value = value.astype(dtype)
            result[target] = value
        return result
