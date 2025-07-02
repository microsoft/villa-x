import os
from collections import deque

import bitsandbytes as bnb
import einops
import hydra
import numpy as np
import torch
from loguru import logger as log
from omegaconf import OmegaConf
from scipy.stats import beta as scipy_beta
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb

from ..model.vla.policy_model import VillaXPolicy
from ..model.vla.processing import VLAProcessor
from ..utils.decorator import main_rank_only
from ..utils.metric import get_action_accuracy
from ..utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time
from ..utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from .model_averaging import ModelAveraging


class Trainer:
    def __init__(self, cfg):
        # device setup
        self.cfg = cfg
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = cfg.multi_gpu
        world_size = 1  # single gpu
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            log.info(
                f"GPU local ID: {self.gpu_id}. Global rank: {global_rank}. Local rank: {local_rank}. Local world size: {local_world_size}. World size: {world_size}. Group rank: {group_rank}"
            )
        self.main_rank = not self.multi_gpu or global_rank == 0
        if not self.main_rank:
            log.remove()

        # logging
        self.use_wandb = cfg.get("wandb", False) and self.main_rank
        if self.use_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=self.wandb_id if hasattr(self, "wandb_id") else None,
                resume="allow",  # not using resume_from
            )
        self.debug = cfg.get("debug", False)
        self.save_model_freq = int(cfg.save_model_freq)
        self.save_model_start = int(cfg.get("save_model_start", 0))
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")

        # check if resume:
        if os.path.isdir(self.checkpoint_dir):
            import re

            # resume
            pattern = re.compile(r"step(\d+)\.pt$")
            max_num = -1  # To keep track of the largest step number found
            max_file = None  # To keep the name of the file with the largest step number
            # Loop over the files in the 'checkpoint' subfolder
            for filename in os.listdir(self.checkpoint_dir):
                match = pattern.match(filename)
                if match:
                    # Extract the number from the file name and convert it into an integer
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_file = filename
            if max_file:
                # resume
                assert cfg.resume_checkpoint_path is None
                cfg.resume_checkpoint_path = os.path.join(self.checkpoint_dir, max_file)
        else:
            # start from scratch
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("start from", cfg.resume_checkpoint_path)

        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm
        self.use_amp = cfg.get("use_amp", True)
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        self.use_torch_compile = cfg.get("use_torch_compile", True)

        # model
        assert not ((cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights), (
            "Please load pretrained weights if quantizing VLM or using Lora."
        )
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )  # since the weights have requires_grad=False. However, we are not excluding the weights from the optimizer yet!

        from ..dataset.oxe import OpenXDataset, OpenXLoadConfig

        dataset_config = OmegaConf.to_container(cfg.data, resolve=True)
        config = OpenXLoadConfig(**dataset_config["config"])
        train_config = config.model_copy()
        train_config.is_train = True
        val_config = config.model_copy()
        val_config.is_train = False
        ds_train = OpenXDataset(train_config)
        ds_val = OpenXDataset(val_config)

        self.train_dataloader = ds_train.get_dataloader(
            batch_size=cfg.per_device_batch_size
        )

        self.val_dataiterator = iter(
            ds_val.get_dataloader(batch_size=cfg.per_device_batch_size)
        )

        def dump_stat(stat):
            stat.config.standardize_fn = None
            stat.config.chunk_filter_fn = None
            return stat.model_dump()

        cfg.id2stat = {k: dump_stat(v) for k, v in ds_train.id2stat.items()}
        self.model = VillaXPolicy(cfg, use_ddp=self.multi_gpu)

        if cfg.resume_checkpoint_path:
            self.load_checkpoint(cfg.resume_checkpoint_path)
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if self.use_torch_compile:
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead" speeds up a lot and reduces VRAM usage a lot more, but causes nan loss on L40, maybe issue with cudagraphs or 8-bit optimizer; max-autotune works on H100s, takes a while to compile
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        log.info(f"Using cuda device: {self.device}, dtype: {self.dtype}")
        if self.multi_gpu:
            log.info(
                f"Using {local_world_size} GPUs in each of the {cfg.n_nodes} nodes"
            )
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                gradient_as_bucket_view=True,
                static_graph=False,  # didn't see improvement with True
            )
            model = self.model.module
            dist.barrier()
        else:
            model = self.model
        log_allocated_gpu_memory(log, "loading model", self.gpu_id)

        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = max(
            cfg.global_batch_size // cfg.per_device_batch_size // world_size, 1
        )
        actual_global_batch_size = (
            cfg.per_device_batch_size * self.grad_accumulation_steps * world_size
        )

        self.eval_thresholds = cfg.eval_thresholds
        self.eval_freq = cfg.eval_freq
        self.per_device_num_eval_batch = (
            cfg.eval_size // cfg.per_device_batch_size // world_size
        )
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {actual_global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        # optimizer - action only: 0.315B (0.333B with adaLN and time_dim=256),
        # rest: 2.291B (0.109B with lora rank 64, 0.055B with rank 32)
        self.train_vlm = cfg.train_vlm
        # robot action ===
        self.trained_parameters = model.robot_action_expert_parameters
        self.robot_action_optimizer = bnb.optim.AdamW8bit(
            model.robot_action_expert_parameters,
            lr=cfg.robot_action_lr,
            weight_decay=cfg.robot_action_weight_decay,
        )
        self.robot_action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.robot_action_optimizer,
            first_cycle_steps=cfg.robot_action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.robot_action_lr,
            min_lr=cfg.robot_action_lr_scheduler.min_lr,
            warmup_steps=cfg.robot_action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        # latent action ===
        self.trained_parameters += model.latent_action_expert_parameters
        self.latent_action_optimizer = bnb.optim.AdamW8bit(
            model.latent_action_expert_parameters,
            lr=cfg.latent_action_lr,
            weight_decay=cfg.latent_action_weight_decay,
        )
        self.latent_action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.latent_action_optimizer,
            first_cycle_steps=cfg.latent_action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.latent_action_lr,
            min_lr=cfg.latent_action_lr_scheduler.min_lr,
            warmup_steps=cfg.latent_action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters (Robot Action): {get_num_params_in_billions(self.robot_action_optimizer):.3f}B"
        )
        log.info(
            f"Number of trained parameters (Latent Action): {get_num_params_in_billions(self.latent_action_optimizer):.3f}B"
        )
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            self.trained_parameters += vlm_trained_parameters
            self.vlm_optimizer = bnb.optim.AdamW8bit(
                vlm_trained_parameters,
                lr=cfg.vlm_lr,
                weight_decay=cfg.vlm_weight_decay,
            )
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            log.info(
                f"Number of trained parameters (VLM): {get_num_params_in_billions(self.vlm_optimizer):.3f}B"
            )
        if cfg.resume_checkpoint_path:
            self.load_optimizer(cfg.resume_checkpoint_path)

        ########### Input processing ###########

        # flow matching timestep sampling
        self.flow_sampling = cfg.get("flow_sampling", "beta")
        assert self.flow_sampling in ["uniform", "beta", "sepbeta"], (
            f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        )
        if self.flow_sampling == "beta":
            flow_alpha = cfg.get("flow_alpha", 1.5)
            flow_beta = cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        elif self.flow_sampling == "sepbeta":
            self.flow_alpha1 = cfg.get(
                "flow_alpha1",
            )
            self.flow_beta1 = cfg.get(
                "flow_beta1",
            )
            self.flow_alpha2 = cfg.get(
                "flow_alpha2",
            )
            self.flow_beta2 = cfg.get(
                "flow_beta2",
            )
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)

        # processor --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )

        # TODO: check the lam initialization
        seqtok_cls = hydra.utils.get_class(cfg.seq_tokenizer._target_)
        self.seq_tokenizer = seqtok_cls(config=cfg.seq_tokenizer.config).to(self.device)

        self.la_attn_mask_ratio = cfg.la_attn_mask_ratio
        self.latent_action_use_vq = cfg.get("latent_action_use_vq", True)
        self.seq_tokenizer_use_2_frames = cfg.seq_tokenizer_use_2_frames
        self.latent_seq_n_history = cfg.dataset_cond_hist_steps

        # dump the yaml for reproduction
        OmegaConf.save(cfg, os.path.join(self.log_dir, "config.yaml"))

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        elif self.flow_sampling == "sepbeta":
            x = scipy_beta.rvs(
                self.flow_alpha1, self.flow_beta1, size=bsz
            )  # for robot action
            u = scipy_beta.cdf(x, self.flow_alpha1, self.flow_beta1)
            y = scipy_beta.ppf(
                u, self.flow_alpha2, self.flow_beta2
            )  # for latent action

            t_ra = self.flow_t_max * (
                1 - x
            )  # flip and shift, for robot action, more on less noisy side
            t_la = self.flow_t_max * (1 - y)  # for latent action, more on noisy side

            t_ra = torch.tensor(t_ra)
            t_la = torch.tensor(t_la)

            return t_ra, t_la
        return t

    def run(self):
        timer = Timer()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = (
            0 if not hasattr(self, "cnt_update") else self.cnt_update
        )  # resume training if loaded checkpoint
        loss_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False

        # deal with the various model.module
        model_meta = self.model
        if self.multi_gpu:
            import torch.distributed as dist

            model = self.model.module
        else:
            model = self.model
        model_meta.train()

        # Set up model averaging
        self.model_averaging = ModelAveraging(model, self.cfg, self.device)

        def preprocess_batch(
            batch,
            split_mask: bool,
            sample_fm_time: bool,
            is_train: bool,
        ):
            images = batch["observation"]["image_primary"]
            proprios = batch["state"]
            proprios = proprios[:, self.latent_seq_n_history]
            proprios = proprios[:, :1]
            actions = batch["action"]

            actions = actions[:, self.latent_seq_n_history]

            texts = [text for text in batch["instruction"]]

            # == calculate loss mask ==
            action_pad_mask = batch["state_mask"][:, self.latent_seq_n_history]

            # [B, T]
            latent_action_pad_mask = batch["observation"]["pad_mask"][..., None].repeat(
                1, 1, self.seq_tokenizer.n_latents
            )
            latent_action_pad_mask = torch.bitwise_or(
                latent_action_pad_mask[:, :-1], latent_action_pad_mask[:, 1:]
            )
            latent_action_pad_mask = latent_action_pad_mask[
                :, self.latent_seq_n_history :
            ]
            latent_action_pad_mask = einops.rearrange(
                latent_action_pad_mask, "B T k -> B (T k)"
            )[
                :,
                : self.cfg.latent_action_n_tokens
                - self.latent_seq_n_history * self.seq_tokenizer.n_latents,
            ]

            # remove pad mask loss for actions
            action_pad_mask = torch.bitwise_or(
                action_pad_mask, batch["has_action"][:, None]
            )
            latent_action_pad_mask = torch.bitwise_or(
                latent_action_pad_mask, batch["has_action"][:, None]
            )
            # ==================================

            images_vlm = einops.rearrange(
                images[:, self.latent_seq_n_history], "B H W C -> B C H W"
            )
            model_inputs = self.processor(text=texts, images=images_vlm)

            # == wrist image ==
            wrist1_images = batch["observation"]["image_wrist"]
            wrist_camera_mask = batch["observation"]["camera_mask"]["image_wrist"]

            keep_mask = (
                torch.rand(wrist_camera_mask.shape, device=wrist_camera_mask.device)
                < 0.25
            )  # 25% ratio keep wrist
            wrist_camera_mask = wrist_camera_mask & keep_mask

            # we only need the first frame's image
            wrist1_images = wrist1_images[:, self.latent_seq_n_history]
            wrist_camera_mask = wrist_camera_mask[:, self.latent_seq_n_history]

            # build causal mask and position ids for action
            (
                causal_mask,
                vlm_position_ids,
                latent_action_position_ids,
                proprio_position_ids,
                real_action_position_ids,
            ) = model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"],
                wrist_camera_mask,
                self.dtype,
                is_train,
                mask_ratio=self.la_attn_mask_ratio,
            )

            valid_state_mask = batch["valid_state_mask"]
            valid_action_mask = (
                batch["valid_action_mask"].unsqueeze(1).expand(-1, actions.shape[1], -1)
            )

            inputs = {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": model_inputs["pixel_values"].to(self.dtype),
                "obs_pad_mask": latent_action_pad_mask,
                "action_pad_mask": action_pad_mask,
                "la_to_ra_ratio": batch["state_per_obs"],
                "wrist_pixel_values": wrist1_images,
                "to_tok_pixel_values": images,
                "vlm_position_ids": vlm_position_ids,
                "latent_action_position_ids": latent_action_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "real_action_position_ids": real_action_position_ids,
                "proprios": proprios.to(self.dtype),
                "actions": actions.to(self.dtype),
                "dataset_id": batch["dataset_id"].to(
                    model_inputs["pixel_values"].device
                ),
                "dataset_id_list": batch["dataset_id"].tolist(),
                "control_frequency": batch["control_frequency"]
                .to(model_inputs["pixel_values"].device)
                .to(self.dtype),
                "valid_state_mask": valid_state_mask.bool(),
                "valid_action_mask": valid_action_mask.bool(),
                "wrist_camera_mask": wrist_camera_mask.bool(),
            }

            if split_mask:
                image_text_mask, action_seq_mask = model.split_full_mask_into_submasks(
                    causal_mask
                )
                inputs["image_text_mask"] = image_text_mask
                inputs["action_seq_mask"] = action_seq_mask
            else:
                inputs["causal_mask"] = causal_mask

            # sample flow matching timesteps
            if sample_fm_time:
                t_ra, t_la = self.sample_fm_time(len(texts))  # [B,]
                inputs["t_ra"] = t_ra.to(self.dtype)
                inputs["t_la"] = t_la.to(self.dtype)

            inputs_ = {}
            for k, v in inputs.items():
                if k == "dataset_id_list":
                    inputs_[k] = v
                else:
                    inputs_[k] = v.to(self.device)
            inputs = inputs_

            # tokenizer the images using seq_tokenizer
            to_token_pixels_values = inputs.pop("to_tok_pixel_values")

            if self.seq_tokenizer_use_2_frames:
                to_token_pixels_values_unfold = torch.stack(
                    [to_token_pixels_values[:, :-1], to_token_pixels_values[:, 1:]],
                    dim=1,
                )
                to_token_pixels_values_unfold = einops.rearrange(
                    to_token_pixels_values_unfold, "B q t h w c -> (B t) q h w c"
                )

                with torch.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                ):
                    with torch.inference_mode():
                        tok_seq = self.seq_tokenizer.encode(
                            to_token_pixels_values_unfold.to(self.device),
                            use_vq=self.latent_action_use_vq,
                        )

                tok_seq = einops.rearrange(
                    tok_seq, "(B T) K D -> B (T K) D", B=to_token_pixels_values.shape[0]
                )

            else:
                with torch.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                ):
                    with torch.inference_mode():
                        tok_seq = self.seq_tokenizer.encode(
                            to_token_pixels_values.to(self.device),
                            use_vq=self.latent_action_use_vq,
                        )
            inputs["latent_actions_history"] = tok_seq[
                :, : self.latent_seq_n_history * self.seq_tokenizer.n_latents
            ]
            inputs["latent_actions"] = tok_seq[
                :,
                self.latent_seq_n_history
                * self.seq_tokenizer.n_latents : self.cfg.latent_action_n_tokens,
            ]

            return inputs

        while 1:
            for batch in tqdm(self.train_dataloader, desc="Training"):
                """
                batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
                observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
                task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
                action (torch.Size([bsz, window, horizon, action_dim], float32)
                action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
                """
                inputs = preprocess_batch(
                    batch,
                    split_mask=False,
                    sample_fm_time=True,
                    is_train=True,
                )

                # make sure only syncing when taking gradient steps
                if (cnt_batch + 1) % self.grad_accumulation_steps != 0:
                    with model_meta.no_sync():
                        with torch.autocast(
                            device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                        ):
                            loss = model_meta(**inputs)
                        if self.debug:
                            log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                        normalized_loss = loss / self.grad_accumulation_steps
                        normalized_loss.backward()
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                    ):
                        loss = model_meta(**inputs)
                    if self.debug:
                        log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()  # gradients synced

                    # step
                    torch.nn.utils.clip_grad_norm_(
                        self.trained_parameters,
                        max_norm=self.max_grad_norm,
                    )
                    self.robot_action_optimizer.step()
                    self.latent_action_optimizer.step()
                    self.robot_action_lr_scheduler.step()
                    self.latent_action_lr_scheduler.step()
                    if self.train_vlm:
                        self.vlm_optimizer.step()
                        self.vlm_lr_scheduler.step()
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer step batch {cnt_batch}"
                        )
                    self.robot_action_optimizer.zero_grad(set_to_none=True)
                    self.latent_action_optimizer.zero_grad(set_to_none=True)
                    if self.train_vlm:
                        self.vlm_optimizer.zero_grad(set_to_none=True)
                    cnt_update += 1

                    # initialize ema/swa
                    self.model_averaging.maybe_initialize(cnt_update)

                    # update ema/swa
                    self.model_averaging.maybe_update(cnt_update)

                    # save model at the end of update, models just synced
                    if (
                        cnt_update % self.save_model_freq == 0
                        and cnt_update > self.save_model_start
                    ) or cnt_update == self.n_updates:
                        self.save_training(
                            cnt_update, cnt_batch, main_rank=self.main_rank
                        )
                        dist.barrier()

                # aggregate loss
                if self.multi_gpu:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss_deque.append(loss.item() / dist.get_world_size())
                else:
                    loss_deque.append(loss.item())

                # validation with action accuracy
                if (cnt_batch + 1) % self.eval_freq == 0:
                    log.info(
                        f"Running evaluation for {self.per_device_num_eval_batch} batches..."
                    )
                    new_eval_from_last_log = True
                    model_meta.eval()
                    model_eval = self.model_averaging.get_model_module()
                    eval_accuracy = torch.zeros(
                        len(self.eval_thresholds), device=self.device
                    )
                    eval_l1_loss = torch.tensor(0.0, device=self.device)
                    with torch.inference_mode():
                        for _ in range(self.per_device_num_eval_batch):
                            batch_eval = next(self.val_dataiterator)
                            inputs = preprocess_batch(
                                batch_eval,
                                split_mask=True,
                                sample_fm_time=False,
                                is_train=False,
                            )
                            # TODO: maybe remove this pad mask part?
                            gt_actions = inputs.pop("actions")
                            _ = inputs.pop("latent_actions")
                            gt_obs_pad_mask = inputs.pop("obs_pad_mask").to(torch.bool)
                            gt_action_pad_mask = inputs.pop("action_pad_mask").to(
                                torch.bool
                            )

                            if not gt_obs_pad_mask.any():
                                gt_obs_pad_mask[:, 0] = True  # incase of nan
                            if not gt_action_pad_mask.any():
                                gt_action_pad_mask[:, 0] = True  # incase of nan

                            _, preds_action = model_eval.infer_action(**inputs)

                            eval_accuracy += get_action_accuracy(
                                gt_actions[gt_action_pad_mask],
                                preds_action[gt_action_pad_mask],
                                self.eval_thresholds,
                            )
                            eval_l1_loss += torch.nn.functional.l1_loss(
                                preds_action[gt_action_pad_mask],
                                gt_actions[gt_action_pad_mask],
                            )
                    model_meta.train()

                    # get stats
                    eval_accuracy = eval_accuracy / self.per_device_num_eval_batch
                    eval_l1_loss = eval_l1_loss / self.per_device_num_eval_batch
                    if self.multi_gpu:
                        dist.all_reduce(eval_accuracy, op=dist.ReduceOp.SUM)
                        dist.all_reduce(eval_l1_loss, op=dist.ReduceOp.SUM)
                        eval_accuracy /= dist.get_world_size()
                        eval_l1_loss /= dist.get_world_size()
                    log_msg = f"Eval | l1 Loss: {eval_l1_loss.item():.3f} | "
                    log_msg += " | ".join(
                        [
                            f"acc thres {threshold}: {accuracy.item():.3f}"
                            for threshold, accuracy in zip(
                                self.eval_thresholds, eval_accuracy
                            )
                        ]
                    )
                    log.info(log_msg)

                # log loss
                if cnt_batch % self.log_freq == 0:
                    loss_metric = np.mean(loss_deque)
                    peak_vram = torch.cuda.max_memory_reserved(self.gpu_id) / (1024**3)
                    log_msg = f"Batch {cnt_batch} Update {cnt_update}: t {timer():8.4f} | vram {peak_vram:6.3f} | loss {loss_metric:6.4f} | action lr {self.robot_action_optimizer.param_groups[0]['lr']:10.8f}"
                    if self.train_vlm:
                        log_msg += f" | vlm lr {self.vlm_optimizer.param_groups[0]['lr']:10.8f}"
                    log.info(log_msg)
                    if self.use_wandb:
                        wandb_metrics = {
                            "loss - train": loss_metric,
                            "gradient steps": cnt_update,
                            "robot action lr": self.robot_action_optimizer.param_groups[
                                0
                            ]["lr"],
                            "latent action lr": self.latent_action_optimizer.param_groups[
                                0
                            ]["lr"],
                        }
                        if self.train_vlm:
                            wandb_metrics["vlm lr"] = self.vlm_optimizer.param_groups[
                                0
                            ]["lr"]
                        if new_eval_from_last_log:
                            wandb_metrics.update(
                                {
                                    f"eval acc - thres {threshold}": accuracy.item()
                                    for threshold, accuracy in zip(
                                        self.eval_thresholds, eval_accuracy
                                    )
                                }
                            )
                            wandb_metrics["eval l1 loss"] = eval_l1_loss.item()
                            new_eval_from_last_log = False
                        wandb.log(wandb_metrics, step=cnt_batch, commit=True)

                # count
                cnt_batch += 1
                if cnt_update >= self.n_updates:
                    return

    @main_rank_only
    @log_execution_time(log)
    def save_training(self, cnt_update: int, cnt_batch: int, main_rank: bool):
        avg_state = self.model_averaging.state_dict()
        model_type = avg_state.get("model_type", "normal")
        n_averaged = avg_state.get("n_averaged", 1)
        if avg_state:
            weights = avg_state["state_dict"]
        elif self.multi_gpu:
            weights = self.model.module.state_dict()
        else:
            weights = self.model.state_dict()
        data = {
            "cnt_update": cnt_update,
            "cnt_batch": cnt_batch,
            "model": weights,
            "robot_action_optimizer": self.robot_action_optimizer.state_dict(),
            "latent_action_optimizer": self.latent_action_optimizer.state_dict(),
            "vlm_optimizer": self.vlm_optimizer.state_dict()
            if self.train_vlm
            else None,
            "robot_action_lr_scheduler": self.robot_action_lr_scheduler.state_dict(),
            "latent_action_lr_scheduler": self.latent_action_lr_scheduler.state_dict(),
            "vlm_lr_scheduler": self.vlm_lr_scheduler.state_dict()
            if self.train_vlm
            else None,
            "wandb_id": wandb.run.id if self.use_wandb else None,
            "n_averaged": n_averaged,
        }
        savepath = os.path.join(self.checkpoint_dir, f"step{cnt_update}.pt")
        torch.save(data, savepath)
        checkpoint_size_in_gb = os.path.getsize(savepath) / (1024**3)
        log.info(
            f"Saved model to {savepath}, size: {checkpoint_size_in_gb:.2f} GB, type: {model_type}, averaged: {n_averaged}"
        )

    @log_execution_time(log)
    def load_checkpoint(self, path: str):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.cnt_update = data["cnt_update"]
        self.cnt_batch = data["cnt_batch"]
        self.wandb_id = data["wandb_id"]
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(
            f"Loaded model from {path} at update {self.cnt_update} batch {self.cnt_batch}"
        )

    @log_execution_time(log)
    def load_optimizer(self, path: str):
        """load to cpu first, then move to gpu"""
        from ..utils.optim import optimizer_to

        data = torch.load(path, weights_only=True, map_location="cpu")
        self.robot_action_optimizer.load_state_dict(data["robot_action_optimizer"])
        self.latent_action_optimizer.load_state_dict(data["latent_action_optimizer"])
        optimizer_to(self.robot_action_optimizer, self.device)
        optimizer_to(self.latent_action_optimizer, self.device)
        self.robot_action_lr_scheduler.load_state_dict(
            data["robot_action_lr_scheduler"]
        )
        self.latent_action_lr_scheduler.load_state_dict(
            data["latent_action_lr_scheduler"]
        )

        if self.train_vlm:
            self.vlm_optimizer.load_state_dict(data["vlm_optimizer"])
            optimizer_to(self.vlm_optimizer, self.device)
            self.vlm_lr_scheduler.load_state_dict(data["vlm_lr_scheduler"])
        log.info(f"Loaded optimizer and scheduler states from {path}")
