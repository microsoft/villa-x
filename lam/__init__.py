import os

if "_DEBUG" not in os.environ:
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

from .model import IgorModel

__all__ = ["IgorModel"]
