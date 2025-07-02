"""
Launcher for all experiments.

"""

import datetime
import logging
import math
import os
import random
import sys
import time
import warnings

import hydra
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf, open_dict

logging.getLogger("httpx").setLevel(logging.ERROR)  # suppress urllib3 warnings
if "_DEBUG" not in os.environ:
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow warnings
    try:
        from absl import logging

        logging.set_verbosity(logging.ERROR)
    except ImportError:
        pass
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    multi_gpu = torch.cuda.device_count() > 1 or cfg.get("n_nodes", 1) > 1
    if multi_gpu:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        from torch.distributed import destroy_process_group, init_process_group

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))

        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    with open_dict(cfg):
        cfg.gpu_id = gpu_id
        cfg.multi_gpu = multi_gpu

    # seeding
    seed = cfg.get("seed", 42) + int(os.getenv("RANK") or 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()
    try:
        cls = hydra.utils.get_class(cfg._target_)
        runner = cls(cfg)
        if multi_gpu:
            if dist.is_initialized():
                dist.barrier()
        runner.run()
        end_time = time.time()
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.exception(e)
        raise e

    if multi_gpu:
        destroy_process_group()


@hydra.main(
    version_base="1.2",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="dummy.yaml",
)  # defaults
def main(cfg: OmegaConf):
    _main(cfg)


if __name__ == "__main__":
    main()
