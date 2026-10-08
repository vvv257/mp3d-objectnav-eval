"""MP3D ObjectNav evaluation toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .agent import ObjectNavAgent
from .evaluator import EvaluationConfig, Evaluator

try:
    __version__ = version("mp3d-objectnav-eval")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["ObjectNavAgent", "EvaluationConfig", "Evaluator", "__version__"]
