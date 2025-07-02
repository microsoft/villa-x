from pathlib import Path

import numpy as np
import torch
from loguru import logger as log
from simpler_env import make_wrapped
from tqdm import tqdm
from transformers import AutoTokenizer

from villax.utils.images import save_images
from villax.utils.monitor import log_allocated_gpu_memory, log_execution_time
from villax.utils.normalize import denormalize_bound, normalize_bound

from ..model.vla.processing import VLAProcessor

ROBOT_SPEC = {
    "google": {
        "la_to_ra_ratio": 2.0,
        "control_freq": 3,
        "dataset_id": 1,
    },
    "widowx": {
        "la_to_ra_ratio": 3.0,
        "control_freq": 5,
        "dataset_id": 0,
    },
}


def prepapre_model_inputs(
    model,
    vlm_processor,
    obs: dict,
    normalization_stat: dict,
    la_to_ra_ratio: float,
    dataset_id: int,
    control_freq: float,
    image_resolution: tuple[int, int] = (224, 224),
    dtype: torch.dtype = torch.float32,
):
    import cv2

    vlm_images = (
        torch.tensor(
            cv2.resize(
                obs["image"]["primary"],
                image_resolution,
                interpolation=cv2.INTER_LANCZOS4,
            )
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    # vlm_images = (
    #     resize(torch.tensor(obs["image"]["primary"]), image_resolution)
    #     .permute(2, 0, 1)
    #     .unsqueeze(0)
    # )
    vlm_inputs = vlm_processor(text=[obs["instruction"]], images=vlm_images)
    proprio = obs["proprio"]
    if normalization_stat:
        proprio = normalize_bound(
            proprio,
            normalization_stat["proprio"]["q01"],
            normalization_stat["proprio"]["q99"],
        )

    wrist_camera_mask = torch.zeros((1,), dtype=torch.bool)

    # build causal mask and position ids for action
    (
        causal_mask,
        vlm_position_ids,
        latent_action_position_ids,
        proprio_position_ids,
        real_action_position_ids,
    ) = model.build_causal_mask_and_position_ids(
        vlm_inputs["attention_mask"],
        wrist_camera_mask,
        dtype=dtype,
        is_train=False,
        mask_ratio=None,
    )
    image_text_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
    B = vlm_inputs["input_ids"].shape[0]
    D = proprio.shape[-1]

    inputs = {
        "input_ids": vlm_inputs["input_ids"],
        "pixel_values": vlm_inputs["pixel_values"].to(dtype),
        "la_to_ra_ratio": la_to_ra_ratio * torch.ones((B,)).to(dtype),
        "wrist_pixel_values": torch.zeros((1, 224, 224, 3), dtype=dtype),
        "image_text_mask": image_text_mask,
        "action_seq_mask": action_mask,
        "vlm_position_ids": vlm_position_ids,
        "latent_action_position_ids": latent_action_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "real_action_position_ids": real_action_position_ids,  # [B, chunk] [[2,3,4,5], * B]
        "proprios": torch.tensor(proprio, dtype=dtype)[None, None, :],  # [B T D]
        "dataset_id": dataset_id * torch.ones((B,), dtype=torch.long),
        "dataset_id_list": [dataset_id] * B,
        "control_frequency": control_freq * torch.ones((B,)).to(dtype),
        "valid_state_mask": torch.ones((1, D), dtype=torch.bool),
        "valid_action_mask": torch.ones((1, 4, D), dtype=torch.bool),
        "wrist_camera_mask": wrist_camera_mask.bool(),
    }
    return inputs


@torch.no_grad()
def infer_action(
    model,
    inputs: dict,
    normalization_stat: dict,
    dtype: torch.dtype = torch.float32,
):
    with torch.autocast(device_type="cuda", dtype=dtype):
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        _, actions = model(**inputs)  # [1, H, 10]

    actions = actions.cpu().numpy()
    if normalization_stat:
        actions = np.concatenate(
            [
                denormalize_bound(
                    actions[..., :-1],
                    normalization_stat["action"]["q01"][:-1],
                    normalization_stat["action"]["q99"][:-1],
                ),
                actions[..., -1:],
            ],
            axis=-1,
        )
    return actions


@torch.no_grad()
def simulate(
    model,
    vlm_processor,
    task: str,
    seed: int,
    reset_options: dict,
    normalization_stat: dict,
    action_steps: int,
    state_encoding: str = "euler",
    action_encoding: str = "euler",
    dtype: torch.dtype = torch.float32,
):
    env = make_wrapped(
        task,
        obs_angle_encoding=state_encoding,
        act_angle_encoding=action_encoding,
        google_robot_sticky_gripper_num_repeat=15,
    )
    obs, _ = env.reset(seed=seed, options=reset_options)

    robot_type = obs["robot_type"]

    save_videos, save_proprios, save_actions = [], [], []
    save_videos.append(obs["image"]["primary"])
    save_proprios.append(obs["proprio"])

    success = False
    for _ in tqdm(range(env.spec.max_episode_steps), desc="Simulating"):
        model_inputs = prepapre_model_inputs(
            model,
            vlm_processor,
            obs,
            normalization_stat,
            **ROBOT_SPEC[robot_type],
            dtype=dtype,
        )
        actions = infer_action(model, model_inputs, normalization_stat, dtype=dtype)
        actions = [env.prepare_env_action(action) for action in actions[0]]
        for action in actions[:action_steps]:
            save_actions.append(action)
            obs, _, terminated, truncated, _ = env.step_env_action(action)
            if terminated or truncated:
                break
        save_videos.append(obs["image"]["primary"])
        save_proprios.append(obs["proprio"])
        if terminated or truncated:
            if terminated:
                success = True
            break

    env.close()
    save_videos = np.stack(save_videos, axis=0)
    save_proprios = np.stack(save_proprios, axis=0)
    save_actions = np.stack(save_actions, axis=0)
    return save_videos, save_proprios, save_actions, success


@log_execution_time(log)
def load_checkpoint(model, path: str):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    data["model"] = {
        k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
    }  # remove "_orig_mod." prefix if saved model was compiled
    model.load_state_dict(data["model"], strict=True)
    log.info(f"Loaded model from {path}")


def evaluate(config):
    import pickle

    import hydra

    n_episodes = config.n_eval_episode
    save_dir = Path(config.log_dir) / "videos"
    model_cls = hydra.utils.get_class(config.policy._target_)
    dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    log.info(f"Evaluting {config.env.task} Using {dtype}")

    id2stat = pickle.load(open(config.id2stat_path, "rb"))
    config.n_datasets = len(id2stat)

    model = model_cls(config, id2stat, use_ddp=False)
    load_checkpoint(model, config.checkpoint_path)
    model.eval().cuda().to(dtype)

    log_allocated_gpu_memory(log, "loading model")
    robot_type = "google" if "google" in config.env.task else "widowx"
    dataset_id = ROBOT_SPEC[robot_type]["dataset_id"]
    stat = id2stat[dataset_id]

    if robot_type == "google":
        # In SIMPLER, gripper min val is different from real dataset
        stat.state.q01[-1] = 0.17449164
    normalize_stat = {
        "proprio": {"q01": np.array(stat.state.q01), "q99": np.array(stat.state.q99)},
        "action": {"q01": np.array(stat.action.q01), "q99": np.array(stat.action.q99)},
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    vlm_processor = VLAProcessor(
        tokenizer,
        num_image_tokens=config.vision.config.num_image_tokens,
        max_seq_len=config.max_seq_len,
        tokenizer_padding=config.tokenizer_padding,
    )
    for i in range(n_episodes):
        video, proprio, action, success = simulate(
            model,
            vlm_processor,
            task=config.env.task,
            seed=i,
            reset_options={"episode_id": i},
            action_steps=config.act_steps,
            normalization_stat=normalize_stat,
            state_encoding=config.state_type,
            action_encoding=config.action_type,
            dtype=dtype,
        )
        postfix = "_success" if success else "_failed"
        log.info(f"Episode {i:03d} finished with success={success}")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_images(video, fp=save_dir / f"{i:03d}{postfix}.mp4", fps=10)
        np.savez_compressed(
            save_dir / f"{i:03d}_proprio_action.npz", proprio=proprio, action=action
        )


class EvalAgent:
    def __init__(self, config):
        self.config = config

    def run(self):
        evaluate(self.config)
