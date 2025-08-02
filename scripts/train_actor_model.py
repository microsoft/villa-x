#!/usr/bin/env python3
"""
Training and upload script for villa-X Actor Model.

This script provides utilities for training the Actor Model with different VLMs
and uploading to Hugging Face Hub.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

from lam.actor_model import ActorModel, ActorModelConfig, prepare_for_hub_upload


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActorModelTrainer:
    """Trainer class for the Actor Model."""
    
    def __init__(
        self,
        model: ActorModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_dataloader) * 100,  # Assume 100 epochs max
            eta_min=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            visual_features = batch['visual_features'].to(self.device)
            target_actions = batch['target_actions'].to(self.device)
            latent_actions = batch.get('latent_actions')
            if latent_actions is not None:
                latent_actions = latent_actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(visual_features, latent_actions)
            loss = self.model.compute_loss(outputs.actions, target_actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_dataloader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                visual_features = batch['visual_features'].to(self.device)
                target_actions = batch['target_actions'].to(self.device)
                latent_actions = batch.get('latent_actions')
                if latent_actions is not None:
                    latent_actions = latent_actions.to(self.device)
                
                outputs = self.model(visual_features, latent_actions)
                loss = self.model.compute_loss(outputs.actions, target_actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        save_every: int = 10,
        validate_every: int = 5,
    ) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if epoch % validate_every == 0:
                val_loss = self.validate()
                logger.info(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_path / "best_model.pt", epoch, val_loss)
                    logger.info(f"New best model saved with val loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pt", epoch)
                logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # Save final model
        self.save_checkpoint(save_path / "final_model.pt", num_epochs)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed!")
        return history
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.model.config.__dict__,
        }
        torch.save(checkpoint, path)


def create_hub_repository(
    repo_id: str,
    model_description: str,
    vlm_name: str,
    private: bool = False
) -> str:
    """Create repository on Hugging Face Hub."""
    api = HfApi()
    
    # Create repository
    repo_url = create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
    )
    
    # Create model card content
    model_card_content = f"""---
license: mit
library_name: pytorch
tags:
- robotics
- vision-language-action
- villa-x
- actor-model
- {vlm_name.lower()}
pipeline_tag: robotics
---

# villa-X Actor Model ({vlm_name})

{model_description}

## Model Description

This Actor Model is part of the villa-X project and predicts robot actions from visual observations. 
It's trained using the {vlm_name} vision-language model as the visual encoder.

## Key Features

- **Vision-Language Integration**: Uses {vlm_name} for rich visual understanding
- **Latent Action Support**: Can leverage latent actions from the LAM (Latent Action Model)
- **Action Horizon**: Predicts multiple future actions for smooth robot control
- **Hugging Face Integration**: Easy to load and use with `from_pretrained()`

## Usage

```python
from lam.actor_model import ActorModel

# Load the model
model = ActorModel.from_pretrained("{repo_id}")

# Use for inference
visual_features = extract_vlm_features(images)  # Extract features using {vlm_name}
actions = model.predict_actions(visual_features)
```

## Model Architecture

- **VLM Backbone**: {vlm_name}
- **Fusion Layers**: Multi-layer Transformer encoder
- **Action Head**: MLP for action prediction
- **Action Dimension**: 7 (6DoF + gripper)
- **Action Horizon**: 10 future steps

## Training

This model was trained on robotic demonstration data with the following setup:

- **Vision-Language Model**: {vlm_name}
- **Loss Function**: MSE loss on action predictions
- **Optimizer**: AdamW with cosine annealing
- **Data Augmentation**: Standard vision augmentations

## Citation

If you use this model, please cite:

```bibtex
@article{{chen2025villa0x0,
  title   = {{villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models}},
  author  = {{Xiaoyu Chen and Hangxing Wei and Pushi Zhang and Chuheng Zhang and Kaixin Wang and Yanjiang Guo and Rushuai Yang and Yucen Wang and Xinquan Xiao and Li Zhao and Jianyu Chen and Jiang Bian}},
  year    = {{2025}},
  journal = {{arXiv preprint arXiv: 2507.23682}}
}}
```

## License

This model is released under the MIT License. See LICENSE for details.
"""
    
    # Upload model card
    api.upload_file(
        path_or_fileobj=model_card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    
    return repo_url


def main():
    parser = argparse.ArgumentParser(description="Train and upload villa-X Actor Model")
    parser.add_argument("--vlm_model", type=str, default="microsoft/molmo-7B-D-0924",
                      help="Vision-Language Model to use")
    parser.add_argument("--action_dim", type=int, default=7,
                      help="Dimension of robot actions")
    parser.add_argument("--action_horizon", type=int, default=10,
                      help="Number of future actions to predict")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                      help="Directory to save model checkpoints")
    parser.add_argument("--upload_to_hub", action="store_true",
                      help="Upload model to Hugging Face Hub after training")
    parser.add_argument("--repo_id", type=str,
                      help="Repository ID for Hugging Face Hub upload")
    parser.add_argument("--private_repo", action="store_true",
                      help="Create private repository on Hub")
    
    args = parser.parse_args()
    
    # Setup model configuration
    config = ActorModelConfig(
        vlm_model_name=args.vlm_model,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        use_latent_actions=True,
    )
    
    # Create model
    model = ActorModel(config)
    
    logger.info(f"Created Actor Model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Using VLM: {args.vlm_model}")
    
    # Note: In a real implementation, you would load your actual dataset here
    logger.warning("This is a template script. You need to implement the dataloader for your specific dataset.")
    
    # Placeholder for training (uncomment and implement with real data)
    """
    # Load datasets
    train_dataloader = create_train_dataloader(args.batch_size)
    val_dataloader = create_val_dataloader(args.batch_size)
    
    # Setup trainer
    trainer = ActorModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
    )
    
    # Load best model for upload
    best_model_path = Path(args.save_dir) / "best_model.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    """
    
    # Upload to Hub if requested
    if args.upload_to_hub:
        if not args.repo_id:
            vlm_short_name = args.vlm_model.split('/')[-1].lower()
            args.repo_id = f"microsoft/villa-x-actor-{vlm_short_name}"
        
        logger.info(f"Uploading model to Hub: {args.repo_id}")
        
        # Create repository and model card
        vlm_name = args.vlm_model.split('/')[-1]
        model_description = f"Actor Model for villa-X trained with {vlm_name} vision-language model."
        
        repo_url = create_hub_repository(
            repo_id=args.repo_id,
            model_description=model_description,
            vlm_name=vlm_name,
            private=args.private_repo,
        )
        
        # Upload model
        prepare_for_hub_upload(
            model=model,
            repo_id=args.repo_id,
            commit_message=f"Upload Actor Model trained with {vlm_name}",
        )
        
        logger.info(f"✅ Model successfully uploaded to: {repo_url}")
    
    logger.info("Script completed!")


if __name__ == "__main__":
    main()
