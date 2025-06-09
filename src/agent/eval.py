"""
Main eval agent. Only for Simpler for now.

"""

import logging
import os

import hydra
import imageio
import numpy as np
import simpler_env
import torch

from src.agent.env_adapter.ensembler import AdaptiveEnsembler

from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)


class EvalAgent:
    def __init__(self, cfg):
        self.n_eval_episode = cfg.n_eval_episode
        self.n_video = cfg.n_video
        self.log_dir = cfg.log_dir
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        import pickle

        id2stat = pickle.load(open("./assets/brss_our_id2stat.pkl", "rb"))
        cfg.n_datasets = len(list(id2stat.keys()))
        print("cfg.n_datasets", cfg.n_datasets)

        for k, v in id2stat.items():
            if v.config.name == "agibotworld_beta":  # TODO: fix
                v.config.action_dim = 7
                v.config.state_dim = 7

        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        model_cls = hydra.utils.get_class(cfg.policy._target_)
        self.model = model_cls(cfg, id2stat, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        #
        # self.model.freq_embedder.dtype = self.dtype
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            raise NotImplementedError
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        self.act_steps = (
            cfg.act_steps
        )  # e.g., run first two steps of predicted four steps

        # env --- no parallelized
        self.env = simpler_env.make(cfg.env.task)

        # env specifics
        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)

        self.cfg = cfg
        if self.cfg.normalization_type != "None":
            from udl import load_dataset
            ds_eval = load_dataset()

            assert self.cfg.normalization_type == "quantile"
            self.ds_stats = ds_eval._stats
        else:
            self.ds_stats = None

        if self.cfg.action_ensemble == "adaptive":
            horizon = (
                4 if "widowx" in self.cfg.env.task else 2
            ) 
            alpha = 0.1
            self.action_ensembler = AdaptiveEnsembler(horizon, alpha)
        else:
            self.action_ensembler = None

    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env = self.env
        env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        stats_name = (
            "bridge_dataset"
            if "widowx" in self.cfg.env.task
            else "fractal20220817_data"
        )
        dataset_id = 0 if "widowx" in self.cfg.env.task else 1 
        control_freq = 5.0 if "widowx" in self.cfg.env.task else 3.0
        la_to_ra_ratio = 3.0 if "widowx" in self.cfg.env.task else 2.0

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        obs, reset_info = env.reset(options=env_reset_options)
        step_cnt = 0
        env_adapter.reset()
        if self.action_ensembler:
            self.action_ensembler.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        recording = self.n_video > 0
        if recording:
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"  # avoid tokenizer forking warning about deadlock
            )

            def video_parent_path(x):
                return os.path.join(self.video_dir, f"video_{x}")

            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        log.info(
            f"Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )
        while 1:
            inputs = self.env_adapter.preprocess(
                env,
                obs,
                instruction,
                normalization_info={
                    "norm_type": self.cfg.normalization_type,
                    "state_type": self.cfg.state_type,
                    "stats": self.ds_stats[stats_name],
                },
            )
            raw_pixels = inputs.pop("raw_pixels")
            wrist_camera_mask = torch.zeros((1,), dtype=torch.bool)
            (
                causal_mask,
                vlm_position_ids,
                latent_action_position_ids,
                proprio_position_ids,
                real_action_position_ids,
            ) = self.model.build_causal_mask_and_position_ids(
                inputs["attention_mask"],
                wrist_camera_mask,
                dtype=self.dtype,
                is_train=False,
                mask_ratio=None,
            )
            image_text_mask, action_mask = self.model.split_full_mask_into_submasks(
                causal_mask
            )

            inputs = {
                "input_ids": inputs["input_ids"],
                "pixel_values": inputs["pixel_values"].to(self.dtype),
                "la_to_ra_ratio": la_to_ra_ratio
                * torch.ones(
                    (inputs["input_ids"].shape[0],),
                )
                .to(inputs["pixel_values"].device)
                .to(self.dtype),
                "wrist_pixel_values": torch.zeros((1, 224, 224, 3), dtype=self.dtype),
                "image_text_mask": image_text_mask,
                "action_seq_mask": action_mask,
                "vlm_position_ids": vlm_position_ids,
                "latent_action_position_ids": latent_action_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "real_action_position_ids": real_action_position_ids,  # [B, chunk] [[2,3,4,5], * B]
                "proprios": inputs["proprios"].to(self.dtype),
                "dataset_id": dataset_id
                * torch.ones((inputs["input_ids"].shape[0],), dtype=torch.long).to(
                    inputs["pixel_values"].device
                ),
                "dataset_id_list": (
                    dataset_id
                    * torch.ones((inputs["input_ids"].shape[0],), dtype=torch.long)
                ).tolist(),
                "control_frequency": control_freq
                * torch.ones(
                    (inputs["input_ids"].shape[0],),
                )
                .to(inputs["pixel_values"].device)
                .to(self.dtype),
                "valid_state_mask": torch.ones(
                    (1, inputs["proprios"].shape[-1]), dtype=torch.bool
                ),
                "valid_action_mask": torch.ones(
                    (1, 4, inputs["proprios"].shape[-1]), dtype=torch.bool
                ),
                "wrist_camera_mask": wrist_camera_mask.bool(),
            }

            inputs_ = {}
            for k, v in inputs.items():
                if k == "dataset_id_list":
                    inputs_[k] = v
                else:
                    inputs_[k] = v.to(self.device)
            inputs = inputs_
            with torch.autocast(
                device_type="cuda",
                dtype=self.dtype,
            ):
                with torch.inference_mode():
                    la, actions = self.model(**inputs)  # [1, H, 10]

            step_cnt += 1
            env_actions = self.env_adapter.postprocess(
                actions[0].float().cpu().numpy(),
                normalization_info={
                    "norm_type": self.cfg.normalization_type,
                    "action_type": self.cfg.action_type,
                    "stats": self.ds_stats[stats_name],
                },
                action_ensembler=self.action_ensembler,
            )

            # environment step
            for env_action in env_actions[: self.act_steps]:  # [H, 7]
                obs, reward, success, truncated, info = env.step(env_action)
                if truncated:
                    break

            # video
            if recording:
                video_writer.append_data(self.env_adapter.get_video_frame(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated:
                successes.append(success)
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        log.info(
                            f"Episode {cnt_episode} success! Video saved to {video_parent_path(cnt_episode)}.mp4"
                        )
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                    else:
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_failed.mp4",
                        )
                        log.info(
                            f"Episode {cnt_episode} failed! Video saved to {video_parent_path(cnt_episode)}_failed.mp4"
                        )
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                env_adapter.reset()
                step_cnt = 0
                if self.action_ensembler:
                    self.action_ensembler.reset()
                instruction = env.get_language_instruction()
                log.info(
                    f"Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")

    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")
