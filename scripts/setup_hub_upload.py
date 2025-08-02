#!/usr/bin/env python3
"""
Automated setup script for villa-X Actor Model Hugging Face Hub integration.

This script helps automate the process of creating and uploading Actor Models
to Hugging Face Hub with proper documentation and model cards.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
import torch

from lam.actor_model import ActorModel, ActorModelConfig


class HubUploadManager:
    """Manages the upload process to Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.token = token
    
    def create_model_card(
        self,
        vlm_name: str,
        config: ActorModelConfig,
        training_details: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
    ) -> str:
        """Create a comprehensive model card for the Actor Model."""
        
        vlm_display_name = vlm_name.split('/')[-1]
        
        # Load template and customize
        template_path = Path(__file__).parent.parent / "docs" / "ACTOR_MODEL_README_TEMPLATE.md"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                model_card = f.read()
        else:
            # Fallback minimal model card
            model_card = f"""---
license: mit
library_name: pytorch
tags:
- robotics
- vision-language-action
- villa-x
- actor-model
- {vlm_display_name.lower()}
pipeline_tag: robotics
---

# villa-X Actor Model ({vlm_display_name})

This Actor Model predicts robot actions from visual observations using {vlm_display_name} as the vision-language backbone.

## Usage

```python
from lam.actor_model import ActorModel

model = ActorModel.from_pretrained("{{repo_id}}")
actions = model.predict_actions(visual_features)
```
"""
        
        # Add specific details
        model_card += f"\n\n## Model Configuration\n\n"
        model_card += f"- **VLM Backbone**: {vlm_name}\n"
        model_card += f"- **Action Dimension**: {config.action_dim}\n"
        model_card += f"- **Action Horizon**: {config.action_horizon}\n"
        model_card += f"- **Hidden Size**: {config.hidden_size}\n"
        model_card += f"- **Latent Actions**: {'Enabled' if config.use_latent_actions else 'Disabled'}\n"
        
        if training_details:
            model_card += f"\n\n## Training Details\n\n"
            for key, value in training_details.items():
                model_card += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        if performance_metrics:
            model_card += f"\n\n## Performance\n\n"
            for metric, value in performance_metrics.items():
                model_card += f"- **{metric}**: {value}\n"
        
        return model_card
    
    def create_config_files(
        self,
        config: ActorModelConfig,
        temp_dir: Path
    ) -> None:
        """Create configuration files for the model."""
        
        # Save model config as JSON
        config_dict = {
            "model_type": "villa-x-actor",
            "architectures": ["ActorModel"],
            **config.__dict__
        }
        
        with open(temp_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Create a simple generation config (if needed)
        generation_config = {
            "do_sample": False,
            "max_length": config.action_horizon,
        }
        
        with open(temp_dir / "generation_config.json", 'w') as f:
            json.dump(generation_config, f, indent=2)
    
    def upload_model(
        self,
        model: ActorModel,
        repo_id: str,
        commit_message: str = "Upload villa-X Actor Model",
        private: bool = False,
        training_details: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
    ) -> str:
        """Upload the complete model to Hugging Face Hub."""
        
        print(f"🚀 Starting upload process for {repo_id}")
        
        # Create repository
        print("📝 Creating repository...")
        repo_url = create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            token=self.token,
        )
        
        # Create temporary directory for upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print("💾 Preparing model files...")
            
            # Save model using PyTorch Hub format
            model.eval()
            model.save_pretrained(temp_path, safe_serialization=True)
            
            # Create configuration files
            self.create_config_files(model.config, temp_path)
            
            # Create model card
            print("📄 Creating model card...")
            model_card = self.create_model_card(
                vlm_name=model.config.vlm_model_name,
                config=model.config,
                training_details=training_details,
                performance_metrics=performance_metrics,
            )
            
            with open(temp_path / "README.md", 'w') as f:
                f.write(model_card)
            
            # Upload to Hub
            print("⬆️ Uploading to Hub...")
            upload_folder(
                folder_path=temp_path,
                repo_id=repo_id,
                commit_message=commit_message,
                token=self.token,
            )
        
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")
        return repo_url


def setup_actor_model_for_vlm(
    vlm_name: str,
    action_dim: int = 7,
    action_horizon: int = 10,
    use_latent_actions: bool = True,
    **config_kwargs
) -> ActorModel:
    """Setup Actor Model for a specific VLM."""
    
    print(f"🔧 Setting up Actor Model for {vlm_name}")
    
    # Determine VLM-specific configuration
    if "molmo" in vlm_name.lower():
        vlm_hidden_size = 4096
    elif "internvl" in vlm_name.lower():
        vlm_hidden_size = 4096
    else:
        vlm_hidden_size = 4096  # Default
        print(f"⚠️ Unknown VLM {vlm_name}, using default hidden size")
    
    config = ActorModelConfig(
        vlm_model_name=vlm_name,
        vlm_hidden_size=vlm_hidden_size,
        action_dim=action_dim,
        action_horizon=action_horizon,
        use_latent_actions=use_latent_actions,
        **config_kwargs
    )
    
    model = ActorModel(config)
    print(f"✅ Actor Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Setup villa-X Actor Model for Hugging Face Hub")
    
    # Model configuration
    parser.add_argument("--vlm_model", type=str, required=True,
                       help="Vision-Language Model name (e.g., microsoft/molmo-7B-D-0924)")
    parser.add_argument("--action_dim", type=int, default=7,
                       help="Robot action dimension")
    parser.add_argument("--action_horizon", type=int, default=10,
                       help="Number of future actions to predict")
    parser.add_argument("--no_latent_actions", action="store_true",
                       help="Disable latent action integration")
    
    # Upload configuration
    parser.add_argument("--repo_id", type=str,
                       help="Hugging Face repository ID (e.g., microsoft/villa-x-actor-molmo)")
    parser.add_argument("--upload", action="store_true",
                       help="Upload model to Hub")
    parser.add_argument("--private", action="store_true",
                       help="Create private repository")
    parser.add_argument("--token", type=str,
                       help="Hugging Face token (or set HF_TOKEN env var)")
    
    # Model details
    parser.add_argument("--checkpoint_path", type=str,
                       help="Path to trained model checkpoint")
    parser.add_argument("--training_details", type=str,
                       help="JSON file with training details")
    parser.add_argument("--performance_metrics", type=str,
                       help="JSON file with performance metrics")
    
    args = parser.parse_args()
    
    # Setup model
    model = setup_actor_model_for_vlm(
        vlm_name=args.vlm_model,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        use_latent_actions=not args.no_latent_actions,
    )
    
    # Load trained weights if provided
    if args.checkpoint_path:
        print(f"📥 Loading weights from {args.checkpoint_path}")
        if args.checkpoint_path.endswith('.pt') or args.checkpoint_path.endswith('.pth'):
            # PyTorch checkpoint
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Assume it's a directory with saved model
            model = ActorModel.from_pretrained(args.checkpoint_path)
        print("✅ Weights loaded successfully")
    
    # Load additional details
    training_details = None
    if args.training_details and Path(args.training_details).exists():
        with open(args.training_details) as f:
            training_details = json.load(f)
    
    performance_metrics = None
    if args.performance_metrics and Path(args.performance_metrics).exists():
        with open(args.performance_metrics) as f:
            performance_metrics = json.load(f)
    
    # Upload to Hub if requested
    if args.upload:
        if not args.repo_id:
            vlm_short = args.vlm_model.split('/')[-1].lower().replace('-', '').replace('_', '')
            args.repo_id = f"microsoft/villa-x-actor-{vlm_short}"
            print(f"🏷️ Auto-generated repo ID: {args.repo_id}")
        
        # Get token
        token = args.token or os.getenv('HF_TOKEN')
        if not token:
            print("❌ Hugging Face token required. Set --token or HF_TOKEN environment variable")
            return
        
        # Upload
        upload_manager = HubUploadManager(token=token)
        upload_manager.upload_model(
            model=model,
            repo_id=args.repo_id,
            commit_message=f"Upload villa-X Actor Model for {args.vlm_model}",
            private=args.private,
            training_details=training_details,
            performance_metrics=performance_metrics,
        )
    else:
        print("✅ Model setup complete!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"VLM: {args.vlm_model}")
        print("Add --upload flag to upload to Hugging Face Hub")


if __name__ == "__main__":
    main()
