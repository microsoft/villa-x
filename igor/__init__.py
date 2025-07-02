import os

if "_DEBUG" not in os.environ:
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

from .models.lam.model import IgorModel

__all__ = ["IgorModel"]
