def set_seed(
    seed: int,
    *,
    include_torch: bool = True,
    include_numpy: bool = True,
    include_tensorflow: bool = False,
):
    import random

    random.seed(seed)
    if include_numpy:
        import numpy as np

        np.random.seed(seed)
    if include_torch:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    if include_tensorflow:
        import os

        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
