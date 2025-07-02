from typing import TYPE_CHECKING, List, Tuple

from loguru import logger

from ..mixture import OXE_NAMED_MIXES
from ..schema import OpenXConfig
from .dataset_configs import OXE_DATASET_CONFIGS

if TYPE_CHECKING:
    from ..schema import OpenXLoadConfig


def prepare_configs_and_weights(
    config: "OpenXLoadConfig",
) -> Tuple[List[OpenXConfig], List[float]]:
    if isinstance(config.name, str):
        data_mix = OXE_NAMED_MIXES[config.name]
    else:
        data_mix = config.name

    filtered_datasets, included_dataset_names = [], []
    for name, weight in data_mix:
        if name not in included_dataset_names:
            filtered_datasets.append((name, weight))
            included_dataset_names.append(name)
        else:
            logger.warning(f"Skipping duplicate: {(name, weight)}.")
    data_mix = filtered_datasets

    ds_configs, weights = [], []
    for name, weight in data_mix:
        try:
            ds_configs.append(
                OpenXConfig(
                    name=name,
                    **config.model_dump(exclude=("name",)),
                    **OXE_DATASET_CONFIGS[name],
                )
            )
            weights.append(weight)
        except ValueError as e:
            logger.warning(f"Skipping {name} due to error: {e}")

    return ds_configs, weights
