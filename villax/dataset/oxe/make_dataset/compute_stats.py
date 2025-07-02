import copy
import json
import sys
from os.path import join as pjoin
from pathlib import Path
from typing import TYPE_CHECKING

import tensorflow as tf
import torch
import tqdm
from loguru import logger

from ...utils.metric import DescribeMetric
from ..schema import OpenXConfig, OXEDatasetStatistics
from .standardize import make_standardized_dataset

if TYPE_CHECKING:
    from ..schema import OpenXLoadConfig


def find_cached_statistics(save_dir: str, uid: str, use_normalization: bool = True):
    if not save_dir:
        return None

    path = Path(save_dir) / f"dataset_statistics_{uid}.json"
    if not path.exists() and not use_normalization:
        # find any cached statistics is sufficient, format: dataset_statistics_{32chars}.json
        for path in Path(save_dir).glob("dataset_statistics_*.json"):
            if len(path.stem) == len("dataset_statistics_") + 32:
                logger.debug(f"Loading existing dataset statistics from {path}.")
                json_data = json.loads(path.read_text())
                json_data.pop("load_config", None)
                json_data.pop("config", None)
                return OXEDatasetStatistics.model_validate(json_data)

    if path.exists():
        try:
            logger.debug(f"Loading existing dataset statistics from {path}.")
            json_data = json.loads(path.read_text())
            json_data.pop("load_config", None)
            json_data.pop("config", None)
            return OXEDatasetStatistics.model_validate(json_data)
        except Exception:
            logger.warning(f"Loading existing dataset statistics fails from {path}.")
    else:
        logger.debug(f"Not Found existing dataset statistics from {path}.")
    return None


def write_dataset_statistics(save_dir: str, metadata: OXEDatasetStatistics, uid: str):
    if not save_dir:
        return

    path = pjoin(save_dir, f"dataset_statistics_{uid}.json")
    try:
        with open(path, "w") as f:
            f.write(metadata.model_dump_json(indent=4))
    except Exception as e:
        logger.warning(f"Could not write dataset statistics to {path}. Re: {e}")


def unique_stats_id(
    load_config: "OpenXLoadConfig", config: "OpenXConfig", n_samples: int | None = None
) -> str:
    import hashlib

    deps = [
        1.0,  # TODO should be replaced with config.load_ratio but tfds don't support all[:k%] split
        config.gripper_encoding,
    ]
    if n_samples:
        deps.append(n_samples)
    uid = hashlib.md5("".join([str(i) for i in deps]).encode("utf-8")).hexdigest()
    return uid


def compute_stats(
    load_config: "OpenXLoadConfig",
    config: "OpenXConfig",
    force: bool = False,
    n_samples: int | None = None,
):
    uid = unique_stats_id(load_config, config, n_samples)
    meta_dir = pjoin(load_config.base_dir, config.name)
    if not force:
        metadata = find_cached_statistics(
            meta_dir, uid, use_normalization=load_config.use_normalization
        )
        if metadata is not None:
            metadata.config = config
            return metadata

    dataset = make_standardized_dataset(config=config, shuffle=True, split="all")

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")
    sdm, adm, ldm = None, None, DescribeMetric(dim=1)
    if "state" in dataset.element_spec:
        sdm = DescribeMetric(dim=dataset.element_spec["state"].shape[1])
    if "action" in dataset.element_spec:
        adm = DescribeMetric(dim=dataset.element_spec["action"].shape[1])
    logger.debug("Computing dataset statistics")

    num_transitions, num_trajectories = 0, 0
    for traj in tqdm.tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
        file=sys.stdout,
        desc="Computing",
    ):
        if sdm:
            sdm.update(torch.tensor(traj["state"]))
        if adm:
            adm.update(torch.tensor(traj["action"]))

        length = torch.tensor([[traj["action"].shape[0]]]).float()
        ldm.update(length)

        num_transitions += traj["action"].shape[0]
        num_trajectories += 1
        if n_samples and num_trajectories >= n_samples:
            break

    def finalize(metric: DescribeMetric):
        if metric is None:
            return None
        if metric.dim == 1:
            return metric.finalize()
        return {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in metric.finalize().items()
        }

    _config = copy.deepcopy(config)
    _config.standardize_fn = None
    _config.chunk_filter_fn = None
    metadata = OXEDatasetStatistics(
        num_transitions=num_transitions,
        num_trajectories=num_trajectories,
        length=finalize(ldm),
        action=finalize(adm),
        state=finalize(sdm),
    )

    write_dataset_statistics(meta_dir, metadata, uid)
    return metadata
