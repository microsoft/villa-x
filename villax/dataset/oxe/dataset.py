import numpy as np
import torch

from .schema import OpenXLoadConfig


def to_tensor(x):
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return x.tolist()
        return torch.tensor(x)
    elif isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    elif isinstance(x, bytes):
        return x.decode()
    else:
        return x


class OpenXDataset(torch.utils.data.IterableDataset):
    def __init__(self, config: OpenXLoadConfig, per_dataset_configs: dict = {}):
        super().__init__()
        from .make_dataset.make_dataset import make_interleaved_dataset
        from .make_dataset.prepare_kwargs import prepare_configs_and_weights

        self.config = config
        ds_configs, weights = prepare_configs_and_weights(config)
        for ds_config in ds_configs:
            update_config = per_dataset_configs.get(ds_config.name, {})
            for k, v in update_config.items():
                if hasattr(ds_config, k):
                    setattr(ds_config, k, v)
                else:
                    raise ValueError(
                        f"Unknown config key {k} for dataset {ds_config.name}"
                    )

        if config.threads == -1:
            config.threads = min(config.max_threads, len(ds_configs))
        dataset, weights, stats = make_interleaved_dataset(
            config=config, dataset_configs=ds_configs, weights=weights
        )
        self._dataset = dataset
        self._stats = stats
        self._weights = weights

        lengths = np.array([stat.num_transitions for stat in stats.values()])
        ratio = np.array(
            [
                (
                    ds_config.train_ratio
                    if ds_config.is_train
                    else 1 - ds_config.train_ratio
                )
                * ds_config.load_ratio
                for ds_config in ds_configs
            ]
        )
        lengths = lengths * ratio
        self._lengths = lengths

    @property
    def stats(self):
        return self._stats

    @property
    def id2stat(self):
        return {i: s for i, s in enumerate(self._stats.values())}

    def __iter__(self):
        for s in self._dataset.as_numpy_iterator():
            yield to_tensor(s)

    @property
    def length(self):
        return int(self._lengths.sum())

    def __len__(self):
        if not self.config.effective_size:
            return int(self._lengths.sum())
        effective_size = 1 / np.sum(np.square(self._weights) / self._lengths)

        return int(effective_size)

    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        include_dataset_name: bool = False,
        drop_last: bool = True,
        prefetch_factor: int | None = None,
        n_max_state_action: int | None = None,
        pin_memory: bool = True,
    ):
        import torch

        from ..collator import CustomCollator

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=CustomCollator(
                load_camera_views=self.config.load_camera_views,
                include_dataset_name=include_dataset_name,
                dataset_mapping={
                    s.config.name: i for i, s in enumerate(self._stats.values())
                },
                n_max_state_action=n_max_state_action,
            ),
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
