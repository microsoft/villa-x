from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence

import torch
from torch.nn.functional import pad


def pad_or_truncate(
    x: torch.Tensor,
    max_length: int,
    pad_value: float | int | bool = 0.0,
    pad_side: str = "right",
    pad_dim: int = 1,
) -> torch.Tensor:
    current_length = x.shape[pad_dim]

    if current_length > max_length:
        return x.narrow(pad_dim, 0, max_length)

    elif current_length < max_length:
        pad_len = max_length - current_length
        pad_dims = [0] * (2 * x.dim())
        if pad_side == "right":
            pad_dims[-(2 * pad_dim + 1)] = pad_len  # right
        elif pad_side == "left":
            pad_dims[-(2 * pad_dim + 2)] = pad_len  # left
        else:
            raise ValueError(f"Invalid pad_side: {pad_side}. Use 'left' or 'right'.")

        return pad(x, pad_dims, value=pad_value)

    else:
        return x


def collate_sa(items: list, n_max_state_action: int | None = None):
    def _contain_action(item):
        has_state = item.get("has_state", False)
        return has_state

    has_action = torch.tensor(
        [_contain_action(item) for item in items], dtype=torch.bool
    )
    items = [item for item in items if _contain_action(item)]
    if not items:
        raise ValueError("No state action found. Please filter at an early stage.")

    if n_max_state_action is None:
        n_max_state_action = max(item["state"].shape[1] for item in items)  # [T N D_s]
    action = torch.stack(
        [
            pad_or_truncate(item["action"], max_length=n_max_state_action)
            for item in items
        ]
    )  # [B T N D_a]
    state = torch.stack(
        [
            pad_or_truncate(item["state"], max_length=n_max_state_action)
            for item in items
        ]
    )  # [B T N D_s]
    state_mask = torch.stack(
        [
            pad_or_truncate(
                item["state_mask"], max_length=n_max_state_action, pad_value=False
            )
            for item in items
        ]
    )  # [B T N]
    return {
        "action": action,
        "state": state,
        "state_mask": state_mask,
        "valid_state_mask": torch.ones(
            [state.shape[0], state.shape[-1]], dtype=torch.bool
        ),
        "valid_action_mask": torch.ones(
            [action.shape[0], action.shape[-1]], dtype=torch.bool
        ),
        "has_action": has_action,
        "arm_type": torch.zeros(len(items), dtype=torch.long),
    }


@dataclass
class CustomCollator:
    def __init__(
        self,
        *,
        load_camera_views: list[str] = ["primary"],
        include_dataset_name: bool = False,
        dataset_mapping: dict[str, int] | None = None,
        n_max_state_action: int | None = None,
    ):
        self._load_camera_views = load_camera_views
        self._include_dataset_name = include_dataset_name
        self._dataset_mapping = dataset_mapping
        self._sa_collator = partial(collate_sa, n_max_state_action=n_max_state_action)

    def __call__(
        self, items: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        clips = {
            f"image_{k}": torch.stack(
                [item["observation"]["image_" + k] for item in items]
            )
            for k in self._load_camera_views
        }
        timestep = torch.stack([item["observation"]["timestep"] for item in items])
        camera_mask = {
            f"image_{k}": torch.stack(
                [item["observation"]["camera_mask"]["image_" + k] for item in items]
            )
            for k in self._load_camera_views
        }
        pad_masks = torch.stack([item["observation"]["pad_mask"] for item in items])

        output = {
            "observation": {
                **clips,
                "pad_mask": pad_masks,
                "camera_mask": camera_mask,
                "timestep": timestep,
            }
        }

        if self._include_dataset_name:
            output["dataset_name"] = [str(item["dataset_name"]) for item in items]

        if self._dataset_mapping:
            output["dataset_id"] = torch.tensor(
                [self._dataset_mapping[str(item["dataset_name"])] for item in items]
            )

        output["instruction"] = [str(item["instruction"]) for item in items]
        output["control_frequency"] = torch.tensor(
            [item.get("control_frequency", 0.0) for item in items]
        )
        output["state_per_obs"] = torch.tensor(
            [item.get("state_per_obs", 0.0) for item in items]
        )

        sa_output = self._sa_collator(items)
        output.update(sa_output)
        return output
