"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
import math
from typing import Tuple

import hydra
import torch
from einops import rearrange
from scipy.stats import beta as scipy_beta
from torch import nn

from villax.utils.decorator import NoSyncBase
from villax.utils.monitor import log_execution_time

from ..kv_cache import KVCache
from .modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)

log = logging.getLogger(__name__)


class FreqEmbedder(nn.Module):
    """

    Embeds scalar timesteps into vector representations.
    https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/models/rdt/blocks.py#L28
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.to(self.dtype)

    def forward(self, freq):
        # freq: [B,]
        t_freq = self.timestep_embedding(
            freq, self.frequency_embedding_size
        )  # [B, D_mid]
        t_emb = self.mlp(t_freq)  # [B, D]
        return t_emb


class VillaXPolicy(nn.Module, NoSyncBase):
    @log_execution_time(log)
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        self.max_image_text_tokens = cfg.max_image_text_tokens

        self.num_proprio_tokens = cfg.proprio_cond_steps
        self.num_latent_action_tokens = cfg.latent_action_n_tokens
        self.num_real_action_tokens = cfg.real_action_n_tokens

        self.emb_context_tokens = 4  # 1 for dataset ID, 1 for freq embedding, 1 for LA -> RA, 1 for have wrist camera
        self.wrist_image_encoder_tokens = cfg.robot_image_encoder_tokens

        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_latent_action_tokens
            + self.wrist_image_encoder_tokens
            + self.emb_context_tokens
            + self.num_proprio_tokens
            + self.num_real_action_tokens
        )

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.latent_action_hidden_size = cfg.mixture.latent_action.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size

        id2stat = cfg.id2stat
        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.robot_action_dim = max(
            [stat["config"]["action_dim"] for stat in id2stat.values()]
        )
        self.proprio_dim = max(
            [stat["config"]["state_dim"] for stat in id2stat.values()]
        )

        self.latent_action_dim = cfg.latent_action_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        self.latent_action_loss_coef = cfg.latent_action_loss_coef

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )

        # Vision
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)

        # Mixtures
        self.joint_model = hydra.utils.instantiate(cfg.joint)

        # ==
        # add resnet encoder
        self.wrist_image_encoder = hydra.utils.instantiate(
            cfg.robot_image_encoder.pretrained
        )
        self.wrist_image_encoder.to(torch.bfloat16)
        self.wrist_image_encoder.init_cross_attn(
            tokens=self.wrist_image_encoder_tokens,
            query_dim=cfg.robot_image_encoder.query_dim,
            out_dim=self.proprio_hidden_size,
            heads=cfg.robot_image_encoder.heads,
            dim_head=cfg.robot_image_encoder.dim_head,
            dropout=0.0,
        )

        # Action, proprio, time encoders
        self.action_encoder = nn.ModuleDict(
            {
                "dataset_" + str(i): ActionEncoder(
                    stat["config"]["action_dim"],
                    self.action_hidden_size,
                    time_cond=True,
                )
                for i, stat in id2stat.items()
            }
        )
        self.latent_action_encoder = ActionEncoder(
            self.latent_action_dim,
            self.latent_action_hidden_size,
            time_cond=True,
        )
        self.time_embedding = SinusoidalPosEmb(
            self.action_hidden_size, cfg.time_max_period
        )

        # Action decoder
        self.latent_action_decoder = nn.Linear(
            self.latent_action_hidden_size,
            self.latent_action_dim,
        )
        self.proprio_encoder = nn.ModuleDict(
            {
                "dataset_" + str(i): nn.Linear(
                    stat["config"]["state_dim"], self.proprio_hidden_size
                )
                for i, stat in id2stat.items()
            }
        )
        self.action_decoder = nn.ModuleDict(
            {
                "dataset_" + str(i): nn.Linear(
                    self.action_hidden_size, stat["config"]["action_dim"]
                )
                for i, stat in id2stat.items()
            }
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

        # === add embodiment contexts & freq contexts === (also modify: action_expert_parameters)
        self.freq_embedder = FreqEmbedder(
            self.action_hidden_size,
        )
        self.la2ra_embedder = FreqEmbedder(
            self.action_hidden_size,
        )
        self.dataset_embedder = nn.Embedding(len(id2stat) + 1, self.action_hidden_size)
        self.wrist_camera_embedder = nn.Embedding(
            2, self.action_hidden_size
        )  # [0 for wo wrist camera / 1 for w wrist camera]

        # ===
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

    @property
    def robot_action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
            + list(self.freq_embedder.parameters())
            + list(self.la2ra_embedder.parameters())
            + list(self.dataset_embedder.parameters())
            + list(self.wrist_camera_embedder.parameters())
            + list(self.wrist_image_encoder.parameters())
        )  # note: action and proprio share weights

    @property
    def latent_action_expert_parameters(self):
        return (
            list(self.latent_action_encoder.parameters())
            + list(self.latent_action_decoder.parameters())
            + list(self.joint_model.mixtures["latent_action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self,
        attention_mask: torch.Tensor,
        wrist_camera_mask,
        dtype: torch.dtype,
        is_train,
        mask_ratio,
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) nxttok nxttok wrist wrist  freq data_emb proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        nxttok      x        x        x                 x      x
        nxttok      x        x        x                 x      x
        wrist       x        x        x                 x      x      x      x      x      x       x
        wrist       x        x        x                 x      x      x      x      x      x       x
        freq        x        x        x                 x      x      x      x      x      x       x
        data_emb    x        x        x                 x      x      x      x      x      x       x
        proprio     x        x        x                 x      x      x      x      x      x       x
        action      x        x        x                 x      x      x      x      x      x       x       x      x
        action      x        x        x                 x      x      x      x      x      x       x       x      x
        """

        bsz = attention_mask.size(0)
        latent_action_start = self.max_image_text_tokens
        latent_action_end = self.max_image_text_tokens + self.num_latent_action_tokens

        proprio_start = self.max_image_text_tokens + self.num_latent_action_tokens
        proprio_end = (
            self.max_image_text_tokens
            + self.num_latent_action_tokens
            + self.num_proprio_tokens
            + self.emb_context_tokens
            + self.wrist_image_encoder_tokens
        )
        action_start = (
            self.max_image_text_tokens
            + self.num_latent_action_tokens
            + self.num_proprio_tokens
            + self.emb_context_tokens
            + self.wrist_image_encoder_tokens
        )

        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            if wrist_camera_mask[idx]:
                # we have wrist camera
                causal_mask[idx, latent_action_start:, :cnt] = (
                    0  # latent action + proprio/action attend to image/text
                )
            else:
                # we dont have wrist camera
                causal_mask[idx, latent_action_start:latent_action_end, :cnt] = (
                    0  # latent action + proprio/action attend to image/text
                )
                causal_mask[
                    idx, proprio_start + self.wrist_image_encoder_tokens :, :cnt
                ] = 0  # latent action + proprio/action attend to image/text

        causal_mask[
            :,
            latent_action_start:latent_action_end,
            latent_action_start:latent_action_end,
        ] = 0  # latent action attend to itself

        if is_train:
            rand_ratio = mask_ratio

            latent_action_idx = (
                torch.arange(latent_action_start, latent_action_end)
                .unsqueeze(0)
                .repeat(bsz, 1)
            )
            idx = torch.rand(bsz, latent_action_end - latent_action_start).argsort(
                dim=1
            )
            shuffled_indep = latent_action_idx.gather(1, idx)

            selected = shuffled_indep[
                ..., : int(rand_ratio * (latent_action_end - latent_action_start))
            ]
            sorted_idx, _ = selected.sort(dim=-1)

            for bsz_idx in range(bsz):
                wrist_offset = (
                    0 if wrist_camera_mask[bsz_idx] else self.wrist_image_encoder_tokens
                )

                if torch.rand((1,))[0] < 0.5:
                    # 50% ratio, mask out all latent action tokens;
                    pass
                else:
                    # 50% ratio, mask out 50% latent action tokens;
                    # proprio & action: attend to latent action, but 50% random mask
                    causal_mask[
                        bsz_idx, proprio_start + wrist_offset :, sorted_idx[bsz_idx]
                    ] = 0

                causal_mask[
                    bsz_idx,
                    proprio_start + wrist_offset : proprio_end,
                    proprio_start + wrist_offset : proprio_end,
                ] = 0  # proprio attend to itself
                causal_mask[bsz_idx, action_start:, proprio_start + wrist_offset :] = (
                    0  # action attend to proprio & itself
                )
        else:
            for bsz_idx in range(bsz):
                wrist_offset = (
                    0 if wrist_camera_mask[bsz_idx] else self.wrist_image_encoder_tokens
                )

                causal_mask[
                    :,
                    proprio_start + wrist_offset : proprio_end,
                    latent_action_start:latent_action_end,
                ] = 0  # proprio attend to latent action & itself
                causal_mask[
                    :,
                    proprio_start + wrist_offset : proprio_end,
                    proprio_start + wrist_offset : proprio_end,
                ] = 0  # proprio attend to latent action & itself

                causal_mask[:, action_start:, latent_action_start:latent_action_end] = (
                    0  # action attend to latent action & proprio & itself
                )
                causal_mask[:, action_start:, proprio_start + wrist_offset :] = (
                    0  # action attend to latent action & proprio & itself
                )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        latent_action_position_ids = torch.arange(
            1, self.num_latent_action_tokens + 1
        ).repeat(bsz, 1)
        proprio_position_ids = torch.arange(
            1,
            self.wrist_image_encoder_tokens
            + self.num_proprio_tokens
            + self.emb_context_tokens
            + 1,
        ).repeat(bsz, 1)
        real_action_position_ids = torch.arange(
            self.wrist_image_encoder_tokens
            + self.num_proprio_tokens
            + self.emb_context_tokens
            + 1,
            self.wrist_image_encoder_tokens
            + self.num_proprio_tokens
            + self.emb_context_tokens
            + self.num_real_action_tokens
            + 1,
        ).repeat(bsz, 1)
        return (
            causal_mask,
            vlm_position_ids,
            latent_action_position_ids,
            proprio_position_ids,
            real_action_position_ids,
        )

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_mask = causal_mask[
            ...,
            : self.max_image_text_tokens,
            : self.max_image_text_tokens,
        ]
        action_seq_mask = causal_mask[..., self.max_image_text_tokens :, :]
        return image_text_mask, action_seq_mask

    # ---------- Inference ----------#

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        latent_actions_history: torch.FloatTensor,
        image_text_mask: torch.FloatTensor,
        action_seq_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        latent_action_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        real_action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        dataset_id: torch.LongTensor,
        dataset_id_list: list[int],
        control_frequency: torch.FloatTensor,
        la_to_ra_ratio: torch.FloatTensor,
        wrist_pixel_values,
        valid_state_mask,
        valid_action_mask,
        wrist_camera_mask,
        debug_action=None,
        debug_mode=None,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        resnet_image = self.resnet_normalize_image(wrist_pixel_values).to(
            inputs_embeds.dtype
        )
        resnet_image = rearrange(resnet_image, "B H W C -> B C H W")
        resnet_features = self.wrist_image_encoder.compute_latent(
            resnet_image,
        )

        # proprio
        proprio_embeds = torch.stack(
            [
                self.proprio_encoder["dataset_" + str(_id)](
                    proprios[i, :][:, valid_state_mask[i]]
                )
                for i, _id in enumerate(dataset_id_list)
            ]
        )

        dataset_embeds = self.dataset_embedder(dataset_id.to(torch.long))[
            :, None
        ]  # [B, 1, D]
        has_wrist_embeds = self.wrist_camera_embedder(wrist_camera_mask.to(torch.long))[
            :, None
        ]  # [B, 1, D]
        freq_embeds = self.freq_embedder(control_frequency)[:, None]  # [B, 1, D]
        la2ra_embeds = self.la2ra_embedder(la_to_ra_ratio)[:, None]  # [B, 1, D]

        proprio_embeds_new = torch.cat(
            [
                resnet_features,
                has_wrist_embeds,
                dataset_embeds,
                la2ra_embeds,
                freq_embeds,
                proprio_embeds,
            ],
            dim=1,
        )  # [B, 4, D]

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=image_text_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.num_real_action_tokens, self.robot_action_dim),
            device=device,
            dtype=dtype,
        )
        if debug_action is not None:
            latent_action = debug_action.clone()
        else:
            latent_action = torch.randn(
                (
                    bsz,
                    self.num_latent_action_tokens - latent_actions_history.shape[1],
                    self.latent_action_dim,
                ),
                device=device,
                dtype=dtype,
            )

        latent_actions_history = latent_actions_history.to(dtype)

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        # prev_t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # same schedule as training

            raw_x = (1.0 - t / self.flow_t_max).cpu().to(torch.float32).numpy()
            u = scipy_beta.cdf(raw_x, self.flow_alpha1, self.flow_beta1)
            y = scipy_beta.ppf(
                u, self.flow_alpha2, self.flow_beta2
            )  # for latent action

            t_la = self.flow_t_max * (1 - y)  # for latent action, more on noisy side
            t_la = torch.tensor(t_la).cuda().to(t.dtype)
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            time_cond_la = self.time_embedding(t_la)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = torch.cat(
                [
                    self.action_encoder["dataset_" + str(_id)](
                        action[i, :][:, valid_action_mask[i][0]].unsqueeze(
                            0
                        ),  # [1 4 D]
                        time_cond[i, :].unsqueeze(0),
                    )
                    for i, _id in enumerate(dataset_id_list)
                ]
            )  # [B 4 D]
            latent_action_embeds = self.latent_action_encoder(
                latent_action, time_cond_la
            )
            latent_action_embeds_history = self.latent_action_encoder(
                latent_actions_history, time_cond_la
            )

            latent_action_embeds_all = torch.cat(
                [latent_action_embeds_history, latent_action_embeds], dim=1
            )
            latent_action_embeds_all = latent_action_embeds_all[
                :, : self.num_latent_action_tokens
            ]

            # [Batch_Size, Horizon_Steps, Embed_Dim]
            joint_model_output = self.joint_model(
                attention_mask=action_seq_mask,  # [B, 1, L+1+K, H+L+1+K]
                position_ids_all={
                    "latent_action": latent_action_position_ids,  # [B, L]
                    "proprio": proprio_position_ids,  # [B, 1]
                    "action": real_action_position_ids,  # [B, K]
                },
                embeds_all={
                    "latent_action": latent_action_embeds_all,  # [B, L, D]
                    "proprio": proprio_embeds_new,  # [B, 1, D]
                    "action": action_embeds,  # [B, K, D]
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )
            action_embeds = joint_model_output["action"]
            latent_action_embeds = joint_model_output["latent_action"]

            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            def pad_lastdim(x: torch.Tensor, length: int, pad_value: float = 0.0):
                pad_size = length - x.size(-1)
                if pad_size <= 0:
                    return x
                pad_shape = [*list(x.shape[:-1]), pad_size]
                pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
                return torch.cat([x, pad], dim=-1)

            action_vel = torch.stack(
                [
                    pad_lastdim(
                        self.action_decoder["dataset_" + str(_id)](action_embeds[i, :]),
                        length=valid_action_mask[i][0].shape[-1],
                    )  # [1, 4, D]
                    for i, _id in enumerate(dataset_id_list)
                ],
                dim=0,
            )  # [B, 4, D]
            latent_action_vel = self.latent_action_decoder(
                latent_action_embeds[:, latent_action_embeds_history.shape[1] :]
            )
            action += delta_t * action_vel
            t += delta_t

            latent_action += delta_t * latent_action_vel

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return latent_action, action

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def resnet_normalize_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
        image = image / 255.0
        image = (image - mean) / std
        return image

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        latent_actions: torch.FloatTensor,
        latent_actions_history: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        latent_action_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        real_action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t_ra: torch.FloatTensor,
        t_la: torch.FloatTensor,
        dataset_id: torch.LongTensor,
        dataset_id_list,
        control_frequency: torch.FloatTensor,
        la_to_ra_ratio: torch.FloatTensor,
        wrist_pixel_values,
        valid_state_mask,
        valid_action_mask,
        wrist_camera_mask,
        obs_pad_mask=None,  # pad loss  [B, T]
        action_pad_mask=None,  # pad loss  [B, K]
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""

        # noisy latent action
        # [Batch_Size, N latent action tokens, latent action dim]
        latent_action_x0 = torch.randn_like(
            latent_actions, device=t_la.device, dtype=t_la.dtype
        )
        latent_action_x1 = latent_actions
        latent_action_psi_t = self.psi_t(
            latent_action_x0, latent_action_x1, t_la
        )  # [B, N_l, D_l]

        # noisy real action
        real_action_x0 = torch.randn_like(actions, device=t_ra.device, dtype=t_ra.dtype)
        real_action_x1 = actions
        real_action_psi_t = self.psi_t(
            real_action_x0, real_action_x1, t_ra
        )  # [B, N_r, D_r]

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        resnet_image = self.resnet_normalize_image(wrist_pixel_values).to(
            inputs_embeds.dtype
        )
        resnet_image = rearrange(resnet_image, "B H W C -> B C H W")
        resnet_features = self.wrist_image_encoder.compute_latent(
            resnet_image,
        )

        # proprio
        proprio_embeds = torch.stack(
            [
                self.proprio_encoder["dataset_" + str(_id)](
                    proprios[i, :][:, valid_state_mask[i]]
                )
                for i, _id in enumerate(dataset_id_list)
            ]
        )  # [B, 1, D_p]

        # dataset embeddings:
        # dataset_id: [B,]
        # control_frequency: [B,]
        dataset_embeds = self.dataset_embedder(dataset_id.to(torch.long))[
            :, None
        ]  # [B, 1, D]
        has_wrist_embeds = self.wrist_camera_embedder(wrist_camera_mask.to(torch.long))[
            :, None
        ]  # [B, 1, D]
        freq_embeds = self.freq_embedder(control_frequency)[:, None]  # [B, 1, D]
        la2ra_embeds = self.la2ra_embedder(la_to_ra_ratio)[:, None]  # [B, 1, D]

        proprio_embeds_new = torch.cat(
            [
                resnet_features,
                has_wrist_embeds,
                dataset_embeds,
                la2ra_embeds,
                freq_embeds,
                proprio_embeds,
            ],
            dim=1,
        )  # [B, 16x2+4, D]

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond_ra = self.time_embedding(t_ra)
        time_cond_la = self.time_embedding(t_la)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        latent_action_embeds = self.latent_action_encoder(
            latent_action_psi_t, time_cond_la
        )
        latent_action_embeds_history = self.latent_action_encoder(
            latent_actions_history, time_cond_la
        )

        action_embeds = torch.cat(
            [
                self.action_encoder["dataset_" + str(_id)](
                    real_action_psi_t[i, :][:, valid_action_mask[i][0]].unsqueeze(0),
                    time_cond_ra[i, :].unsqueeze(0),
                )
                for i, _id in enumerate(dataset_id_list)
            ]
        )

        # ==
        # first pass: latent action
        pass_1_causal_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_latent_action_tokens,
            : self.max_image_text_tokens + self.num_latent_action_tokens,
        ]
        pass_2_causal_mask = causal_mask[
            ..., self.max_image_text_tokens + self.num_latent_action_tokens :, :
        ]

        pass1_kv_caches = {
            "vlm": KVCache(),
            "latent_action": KVCache(),
        }
        # return {name: KVCache() for name in self.cache_names}
        latent_action_embeds_all = torch.cat(
            [latent_action_embeds_history, latent_action_embeds], dim=1
        )
        joint_model_outputs_1, pass1_kv_caches = self.joint_model(
            attention_mask=pass_1_causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "latent_action": latent_action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "latent_action": latent_action_embeds_all,
            },
            time_cond=None,
            kv_caches=pass1_kv_caches,
            return_caches=True,
        )

        # == detach the kv cache for latent action ==
        for i in range(len(pass1_kv_caches["latent_action"].key_cache)):
            pass1_kv_caches["latent_action"].key_cache[i] = (
                pass1_kv_caches["latent_action"].key_cache[i].detach()
            )
        for i in range(len(pass1_kv_caches["latent_action"].value_cache)):
            pass1_kv_caches["latent_action"].value_cache[i] = (
                pass1_kv_caches["latent_action"].value_cache[i].detach()
            )
        # ==

        joint_model_outputs_2 = self.joint_model(
            attention_mask=pass_2_causal_mask,
            position_ids_all={
                "proprio": proprio_position_ids,
                "action": real_action_position_ids,
            },
            embeds_all={
                "proprio": proprio_embeds_new,
                "action": action_embeds,
            },
            time_cond=None,
            kv_caches=pass1_kv_caches,  # no caching during training
        )

        action_embeds = joint_model_outputs_2["action"]
        latent_action_embeds = joint_model_outputs_1["latent_action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        latent_action_embeds = latent_action_embeds[:, -latent_actions.shape[1] :]
        latent_action_v_psi = self.latent_action_decoder(latent_action_embeds)
        action_v_psi = torch.cat(
            [
                self.action_decoder["dataset_" + str(_id)](action_embeds[i, :])[
                    action_pad_mask[i]
                ].flatten()  # [4D]
                for i, _id in enumerate(dataset_id_list)
            ],
            dim=-1,
        )  # [B * 4 * D]

        # compare to true velocity
        latent_action_d_psi = (
            latent_action_x1 - (1 - self.flow_sig_min) * latent_action_x0
        )
        action_d_psi = real_action_x1 - (1 - self.flow_sig_min) * real_action_x0

        action_pad_mask_expand = action_pad_mask.unsqueeze(-1).expand(
            -1, -1, action_d_psi.shape[-1]
        )
        action_pad_mask_expand = action_pad_mask_expand & valid_action_mask
        action_d_psi = action_d_psi[action_pad_mask_expand]  # [B * 4 * D]

        return torch.sum(
            ((latent_action_v_psi - latent_action_d_psi) ** 2) * obs_pad_mask[..., None]
        ) / (
            1e-6
            + torch.sum(
                obs_pad_mask[..., None].repeat(1, 1, latent_action_d_psi.shape[-1])
            )
        ) * self.latent_action_loss_coef + torch.mean(
            (action_v_psi - action_d_psi) ** 2
        )


class VillaXPolicyInference(VillaXPolicy):
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        latent_actions_history: torch.FloatTensor,
        image_text_mask: torch.FloatTensor,
        action_seq_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        latent_action_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        real_action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        dataset_id: torch.LongTensor,
        dataset_id_list,
        control_frequency: torch.FloatTensor,
        la_to_ra_ratio,
        wrist_pixel_values,
        valid_state_mask,
        valid_action_mask,
        wrist_camera_mask,
    ) -> torch.FloatTensor:
        return super().infer_action(
            input_ids,
            pixel_values,
            latent_actions_history,
            image_text_mask,
            action_seq_mask,
            vlm_position_ids,
            latent_action_position_ids,
            proprio_position_ids,
            real_action_position_ids,
            proprios,
            dataset_id,
            dataset_id_list,
            control_frequency,
            la_to_ra_ratio,
            wrist_pixel_values,
            valid_state_mask,
            valid_action_mask,
            wrist_camera_mask,
        )
