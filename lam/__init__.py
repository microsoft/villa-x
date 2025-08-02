import os

if "_DEBUG" not in os.environ:
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

from .model import IgorModel

try:
    from .actor_model import ActorModel, ActorModelConfig
    __all__ = ["IgorModel", "ActorModel", "ActorModelConfig"]
except ImportError:
    # huggingface_hub might not be installed
    __all__ = ["IgorModel"]
