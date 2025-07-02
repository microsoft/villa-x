from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from igor import IgorModel


class IgorLAM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.use_continuous = config["use_continuous"]
        ckpt_dir = config["ckpt_dir"]
        self.model = IgorModel.from_pretrained(ckpt_dir)
        self.model = self.model.eval()

        self.n_latents = self.model.config.num_learned_tokens
        self.n_dim = self.model.config.action_latent_dim

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor, use_vq: bool = True):
        out = self.model.idm(imgs, return_dict=True)
        B, T, *_ = imgs.shape
        if not use_vq:
            return torch.stack(out["tokens"]).reshape(
                B, (T - 1) * self.n_latents, self.n_dim
            )
        if self.use_continuous:
            return out["vq_tokens"].reshape(B, (T - 1) * self.n_latents, self.n_dim)
        return out["indices"].reshape(B, (T - 1) * self.n_latents)

    def decode(
        self,
        imgs: torch.Tensor,
        action_emb: torch.Tensor,
        use_vq: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_vq:
            vq_action_emb = self.model.vq(action_emb)
            action_emb_in = vq_action_emb[0]

        action_emb_in = rearrange(
            action_emb, "B (T L) D -> B T L D", L=self.model.encoder.map_block.n_latents
        )  # [B T 4 128]
        action_emb_in = rearrange(action_emb_in, "B T L D -> (B T) (L D)")[
            :, None
        ]  # [B * T, 1, 512]
        dec_clips = self.model.preprocess(imgs.to("cuda"), augment_type="resize")
        # pad for one step for the decoder function
        dec_clips = torch.cat([dec_clips, dec_clips[:, -1:]], dim=1)

        reconstructions, _ = self.model.decoder(
            dec_clips, action_emb_in, [dec_clips.shape[1]] * dec_clips.shape[0]
        )
        reconstructions = rearrange(
            reconstructions,
            "bsz (height width) (patch_h patch_w c) -> bsz (height patch_h) (width patch_w) c",
            patch_h=14,
            patch_w=14,
            height=16,
        )  # (B*T_a, 224, 224, 3)

        mean = (
            torch.tensor([0.485, 0.456, 0.406])
            .reshape((1, 1, 1, 3))
            .to(reconstructions.device)
        )
        std = (
            torch.tensor([0.229, 0.224, 0.225])
            .reshape((1, 1, 1, 3))
            .to(reconstructions.device)
        )

        reconstructions = torch.clip(
            (reconstructions * std + mean) * 255.0, min=0.0, max=255.0
        )
        reconstructions = reconstructions.to(torch.uint8)  # [B*T, 224, 224, 3]
        reconstructions = rearrange(
            reconstructions, "(B T) H W C -> B T H W C", B=imgs.shape[0]
        )

        return reconstructions

    def iter_decode(
        self,
        first_img,
        action_emb,
        use_vq=False,
    ):
        # first_img : [B, H, W, C]
        # action_emb : [B, T*L, D]
        action_emb = rearrange(
            action_emb, "B (T L) D -> B T L D", L=self.model.encoder.map_block.n_latents
        )
        T = action_emb.shape[1]
        per_img = first_img[:, None]
        img_results = [
            per_img,
        ]
        for t in range(T):
            per_img = self.decode(per_img, action_emb[:, t], use_vq=use_vq)
            img_results.append(per_img)

        img_results = torch.cat(img_results, dim=1)
        return img_results
