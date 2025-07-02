from dataclasses import dataclass

import torch


@dataclass
class LatentActionOutput:
    latent_action: torch.Tensor
    encoder_output: torch.Tensor | None = None


class LatentActionMixin:
    @torch.inference_mode()
    def idm(
        self,
        clips: torch.Tensor,
        *,
        return_dict: bool = False,
        return_vq_tokens: bool = False,
    ):
        tokens = self.encoder.idm(clips)  # [B T-1 D']

        vq_tokens, _, (_, _, indices) = self.vq(torch.stack(tokens).contiguous())

        if not return_dict:
            return vq_tokens if return_vq_tokens else tokens
        return {"tokens": tokens, "vq_tokens": vq_tokens, "indices": indices}
