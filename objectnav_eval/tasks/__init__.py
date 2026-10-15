from .base import TaskProtocol
from .objectnav import objectnav_protocol
from .vln_ce import vln_ce_episode_uid, vln_ce_r2r_protocol


_BUILTIN_PROTOCOLS = {
    "mp3d_objectnav_v1": objectnav_protocol,
    "vln_ce_r2r_v1": vln_ce_r2r_protocol,
}


def is_builtin_protocol(name):
    """Report whether `name` selects a built-in protocol instead of a YAML path."""
    return str(name) in _BUILTIN_PROTOCOLS


def load_protocol(path=None):
    if path is None:
        return objectnav_protocol()
    if is_builtin_protocol(path):
        return _BUILTIN_PROTOCOLS[str(path)]()
    from objectnav_eval.config import load_yaml
    return TaskProtocol(**load_yaml(path))


__all__ = [
    "TaskProtocol",
    "is_builtin_protocol",
    "load_protocol",
    "objectnav_protocol",
    "vln_ce_episode_uid",
    "vln_ce_r2r_protocol",
]
