"""
Actor Model for villa-X with Hugging Face Hub integration.

This module provides the Actor Model implementation with PyTorchModelHubMixin
for easy integration with Hugging Face Hub.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .base import PretrainedConfig
from .model import IgorModel


@dataclass
class ActorModelOutput:
    """Output class for ActorModel."""
    
    actions: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class ActorModelConfig(PretrainedConfig):
    """Configuration class for ActorModel."""
    
    # Vision-Language Model config
    vlm_model_name: str = "microsoft/molmo-7B-D-0924"  # Default to Molmo
    vlm_hidden_size: int = 4096
    
    # Action prediction config
    action_dim: int = 7  # 6DoF + gripper
    action_horizon: int = 10
    action_chunk_size: int = 100
    
    # Architecture config
    hidden_size: int = 512
    num_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout_prob: float = 0.1
    
    # Latent action integration
    use_latent_actions: bool = True
    latent_action_dim: int = 512  # From LAM: num_learned_tokens * action_latent_dim
    
    # Training config
    loss_type: str = "mse"  # "mse", "l1", "smooth_l1"
    action_loss_weight: float = 1.0
    
    def model_post_init(self, __context):
        """Post-initialization checks and derived configurations."""
        if self.vlm_model_name and "molmo" in self.vlm_model_name.lower():
            self.vlm_hidden_size = 4096
        elif self.vlm_model_name and "internvl" in self.vlm_model_name.lower():
            self.vlm_hidden_size = 4096  # Adjust based on specific InternVL variant


class ActorModel(nn.Module, PyTorchModelHubMixin):
    """
    Actor Model for villa-X that predicts robot actions from visual observations
    and optionally latent actions from the Latent Action Model (LAM).
    
    This model integrates with Hugging Face Hub via PyTorchModelHubMixin for easy
    sharing and distribution.
    """
    
    config_class = ActorModelConfig
    
    def __init__(self, config: ActorModelConfig):
        super().__init__()
        self.config = config
        
        # Vision-Language Model (will be loaded separately)
        self.vlm_projection = nn.Linear(config.vlm_hidden_size, config.hidden_size)
        
        # Latent Action projection (if using LAM)
        if config.use_latent_actions:
            self.latent_action_projection = nn.Linear(
                config.latent_action_dim, config.hidden_size
            )
        
        # Multi-modal fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_prob,
                batch_first=True,
            )
            for _ in range(config.num_layers)
        ])
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.intermediate_size, config.action_horizon * config.action_dim),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        latent_actions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ActorModelOutput, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the Actor Model.
        
        Args:
            visual_features (torch.Tensor): Visual features from VLM [batch_size, seq_len, vlm_hidden_size]
            latent_actions (torch.Tensor, optional): Latent actions from LAM [batch_size, seq_len, latent_action_dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            return_dict (bool): Whether to return ActorModelOutput or tuple
            
        Returns:
            ActorModelOutput or tuple containing:
                - actions: Predicted actions [batch_size, action_horizon, action_dim]
                - hidden_states: Final hidden states
                - attention_weights: Attention weights from last layer
        """
        batch_size, seq_len = visual_features.shape[:2]
        
        # Project visual features
        visual_hidden = self.vlm_projection(visual_features)  # [B, L, H]
        
        # Combine with latent actions if available
        if self.config.use_latent_actions and latent_actions is not None:
            latent_hidden = self.latent_action_projection(latent_actions)  # [B, L-1, H]
            
            # Handle sequence length mismatch (latent actions are typically shorter)
            if latent_hidden.shape[1] != visual_hidden.shape[1]:
                # Pad latent actions to match visual sequence length
                if latent_hidden.shape[1] < visual_hidden.shape[1]:
                    # Pad with last latent action
                    last_latent = latent_hidden[:, -1:, :]  # [B, 1, H]
                    pad_length = visual_hidden.shape[1] - latent_hidden.shape[1]
                    latent_hidden = torch.cat([latent_hidden] + [last_latent] * pad_length, dim=1)
                else:
                    # Truncate if latent actions are longer
                    latent_hidden = latent_hidden[:, :visual_hidden.shape[1], :]
            
            # Fusion: simple addition (could be more sophisticated)
            hidden_states = visual_hidden + latent_hidden
        else:
            hidden_states = visual_hidden
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Pass through transformer layers
        for layer in self.fusion_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Global average pooling over sequence dimension
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states * mask_expanded
            pooled_hidden = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_hidden = hidden_states.mean(dim=1)  # [B, H]
        
        # Predict actions
        action_logits = self.action_head(pooled_hidden)  # [B, action_horizon * action_dim]
        actions = action_logits.view(batch_size, self.config.action_horizon, self.config.action_dim)
        
        if not return_dict:
            return (actions, hidden_states, None)
        
        return ActorModelOutput(
            actions=actions,
            hidden_states=hidden_states,
            attention_weights=None,  # Could implement attention visualization
        )
    
    def compute_loss(
        self,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute action prediction loss.
        
        Args:
            predicted_actions: Predicted actions [batch_size, action_horizon, action_dim]
            target_actions: Target actions [batch_size, action_horizon, action_dim]
            mask: Optional mask for valid actions [batch_size, action_horizon]
            
        Returns:
            loss: Scalar loss tensor
        """
        if self.config.loss_type == "mse":
            loss_fn = nn.MSELoss(reduction='none')
        elif self.config.loss_type == "l1":
            loss_fn = nn.L1Loss(reduction='none')
        elif self.config.loss_type == "smooth_l1":
            loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Compute loss
        loss = loss_fn(predicted_actions, target_actions)  # [B, H, D]
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded
            loss = loss.sum() / mask_expanded.sum()
        else:
            loss = loss.mean()
        
        return loss * self.config.action_loss_weight
    
    def predict_actions(
        self,
        visual_features: torch.Tensor,
        latent_actions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict actions for inference.
        
        Args:
            visual_features: Visual features from VLM
            latent_actions: Optional latent actions from LAM
            temperature: Temperature for action sampling (currently unused)
            
        Returns:
            actions: Predicted actions [batch_size, action_horizon, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(visual_features, latent_actions)
            return outputs.actions
    
    @classmethod
    def from_pretrained_lam(
        cls,
        lam_model_path: str,
        actor_config: Optional[ActorModelConfig] = None,
        **kwargs
    ) -> 'ActorModel':
        """
        Create ActorModel with a pre-trained LAM model.
        
        Args:
            lam_model_path: Path to pre-trained LAM model
            actor_config: Actor model configuration
            **kwargs: Additional arguments
            
        Returns:
            ActorModel instance with LAM integration
        """
        if actor_config is None:
            actor_config = ActorModelConfig()
        
        # Load LAM model
        lam_model = IgorModel.from_pretrained(lam_model_path)
        
        # Create actor model
        actor_model = cls(actor_config)
        
        # Store reference to LAM model for latent action extraction
        actor_model.lam_model = lam_model
        
        return actor_model
    
    def extract_latent_actions(self, video_clips: torch.Tensor) -> torch.Tensor:
        """
        Extract latent actions from video clips using LAM.
        
        Args:
            video_clips: Video clips [batch_size, frames, channels, height, width]
            
        Returns:
            latent_actions: Extracted latent actions [batch_size, frames-1, latent_action_dim]
        """
        if not hasattr(self, 'lam_model'):
            raise ValueError("LAM model not loaded. Use from_pretrained_lam() to load with LAM.")
        
        self.lam_model.eval()
        with torch.no_grad():
            latent_actions = self.lam_model.idm(video_clips)
            # Convert list of tensors to single tensor and reshape
            latent_actions = torch.cat([la.squeeze(1) for la in latent_actions], dim=0)
            latent_actions = latent_actions.view(video_clips.shape[0], -1, self.config.latent_action_dim)
        
        return latent_actions


# Utility functions for model setup and usage

def setup_actor_model_for_training(
    vlm_model_name: str = "microsoft/molmo-7B-D-0924",
    use_latent_actions: bool = True,
    action_dim: int = 7,
    action_horizon: int = 10,
    **config_kwargs
) -> ActorModel:
    """
    Setup ActorModel for training with specified configuration.
    
    Args:
        vlm_model_name: Name of the vision-language model
        use_latent_actions: Whether to use latent actions from LAM
        action_dim: Dimension of robot actions
        action_horizon: Number of future actions to predict
        **config_kwargs: Additional configuration arguments
        
    Returns:
        Configured ActorModel instance
    """
    config = ActorModelConfig(
        vlm_model_name=vlm_model_name,
        use_latent_actions=use_latent_actions,
        action_dim=action_dim,
        action_horizon=action_horizon,
        **config_kwargs
    )
    
    return ActorModel(config)


def prepare_for_hub_upload(
    model: ActorModel,
    repo_id: str,
    commit_message: str = "Upload Actor Model for villa-X",
    **push_kwargs
):
    """
    Prepare and upload ActorModel to Hugging Face Hub.
    
    Args:
        model: Trained ActorModel instance
        repo_id: Repository ID on Hugging Face Hub (e.g., "microsoft/villa-x-actor")
        commit_message: Commit message for the upload
        **push_kwargs: Additional arguments for push_to_hub
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Push to hub using PyTorchModelHubMixin
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        **push_kwargs
    )
    
    print(f"✅ Actor Model successfully uploaded to: https://huggingface.co/{repo_id}")


# Example usage and integration guide
def example_usage():
    """Example of how to use the ActorModel with Hugging Face Hub."""
    
    # 1. Setup model for training
    actor_model = setup_actor_model_for_training(
        vlm_model_name="microsoft/molmo-7B-D-0924",
        use_latent_actions=True,
        action_dim=7,
        action_horizon=10,
    )
    
    # 2. Load with pre-trained LAM (when available)
    # actor_model = ActorModel.from_pretrained_lam(
    #     lam_model_path="path/to/lam/model",
    #     actor_config=ActorModelConfig(...)
    # )
    
    # 3. Training loop (simplified)
    # for batch in dataloader:
    #     visual_features = extract_vlm_features(batch['images'])
    #     latent_actions = actor_model.extract_latent_actions(batch['videos'])
    #     
    #     outputs = actor_model(visual_features, latent_actions)
    #     loss = actor_model.compute_loss(outputs.actions, batch['target_actions'])
    #     
    #     # Backpropagation...
    
    # 4. Upload to Hub after training
    # prepare_for_hub_upload(
    #     model=actor_model,
    #     repo_id="microsoft/villa-x-actor-molmo",
    #     commit_message="Upload Actor Model trained on Molmo VLM"
    # )
    
    # 5. Load from Hub for inference
    # actor_model = ActorModel.from_pretrained("microsoft/villa-x-actor-molmo")
    
    print("Actor Model setup complete! Ready for training and Hub integration.")


if __name__ == "__main__":
    example_usage()
