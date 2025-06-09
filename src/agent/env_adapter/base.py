import numpy as np


class BaseEnvAdapter:
    def __init__(self):
        pass

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min + eps) + data_min
        return rdata

    def normalize_gaussian(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return (data - mean) / (std + eps)

    def denormalize_gaussian(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return data * (std + eps) + mean
