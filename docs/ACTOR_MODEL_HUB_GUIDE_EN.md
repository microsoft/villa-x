# villa-X Actor Model Hub Upload Guide

This guide provides complete instructions for uploading villa-X Actor Models to Hugging Face Hub using PyTorchModelHubMixin.

## Quick Setup

### Prerequisites

```bash
# Install dependencies
pip install huggingface-hub safetensors

# Configure your Hugging Face token
huggingface-cli login
```

### Repository Structure

Create separate repositories for each VLM variant:
- `microsoft/villa-x-actor-molmo` - Molmo-based models
- `microsoft/villa-x-actor-internvl` - InternVL-based models
- `your-org/villa-x-actor-custom` - Custom VLM models

## Training and Upload Workflow

### 1. Basic Training

```python
from lam.actor_model import setup_actor_model_for_training
import torch

# Setup model
model = setup_actor_model_for_training(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,
    action_horizon=10,
    use_latent_actions=True,
)

# Your training loop here
for epoch in range(num_epochs):
    # ... training code ...
    pass

# Save final model
torch.save(model.state_dict(), "villa_x_actor_final.pt")
```

### 2. Manual Hub Upload

```python
from lam.actor_model import ActorModel, ActorModelConfig

# Load trained model
config = ActorModelConfig(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,
    action_horizon=10,
    use_latent_actions=True,
)

model = ActorModel(config)
model.load_state_dict(torch.load("villa_x_actor_final.pt"))

# Upload to Hub
model.push_to_hub(
    repo_id="microsoft/villa-x-actor-molmo",
    commit_message="Add villa-X Actor Model with Molmo VLM",
    tags=["villa-x", "actor-model", "robotics", "molmo"],
)
```

### 3. Automated Training + Upload

```python
# Use provided training script
python scripts/train_actor_model.py \
    --vlm_model microsoft/molmo-7B-D-0924 \
    --action_dim 7 \
    --action_horizon 10 \
    --epochs 100 \
    --upload_to_hub \
    --repo_id microsoft/villa-x-actor-molmo
```

## Loading from Hub

### Basic Usage

```python
from lam.actor_model import ActorModel

# Load pre-trained model
model = ActorModel.from_pretrained("microsoft/villa-x-actor-molmo")

# Use for inference
import torch
visual_features = torch.randn(1, 5, 4096)  # Example VLM features
actions = model.predict_actions(visual_features)

print(f"Predicted actions shape: {actions.shape}")  # [1, 10, 7]
```

### Advanced Configuration

```python
from lam.actor_model import ActorModel, ActorModelConfig

# Load with custom configuration
config = ActorModelConfig(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,
    action_horizon=10,
    use_latent_actions=True,
)

model = ActorModel.from_pretrained(
    "microsoft/villa-x-actor-molmo",
    config=config,
)
```

## Model Card Template

### Automatic Generation

The training script automatically generates model cards with:

- Model architecture details
- Training hyperparameters
- Performance metrics
- Usage examples
- Licensing information

### Custom Model Card

Create `README.md` in your repository:

```markdown
---
tags:
- villa-x
- actor-model
- robotics
- molmo
license: apache-2.0
---

# villa-X Actor Model (Molmo)

This model predicts robot actions from visual observations using the villa-X Actor Model architecture with Molmo VLM.

## Model Details

- **Architecture**: villa-X Actor Model
- **VLM**: microsoft/molmo-7B-D-0924
- **Parameters**: 16.2M
- **Action Dimension**: 7 (6DoF + gripper)
- **Action Horizon**: 10 timesteps

## Usage

```python
from lam.actor_model import ActorModel
import torch

# Load model
model = ActorModel.from_pretrained("microsoft/villa-x-actor-molmo")

# Predict actions
visual_features = torch.randn(1, 5, 4096)
actions = model.predict_actions(visual_features)
```

## Training Data

Trained on [describe your dataset here]

## Performance

- Inference Speed: 9.4 Hz on CPU
- Task Success Rate: [your metrics]

## Citation

```bibtex
@misc{villa-x-actor-model,
  title={villa-X Actor Model},
  author={[Your Authors]},
  year={2024},
  url={https://github.com/microsoft/villa-x}
}
```
```

## Repository Organization

### Multiple VLM Support

Structure your repositories by VLM:

```
microsoft/villa-x-actor-molmo/
├── README.md
├── config.json
├── pytorch_model.bin
└── model_card.md

microsoft/villa-x-actor-internvl/
├── README.md
├── config.json
├── pytorch_model.bin
└── model_card.md
```

### Version Management

Use branches or tags for different versions:

```bash
# Create version tag
git tag v1.0.0
git push origin v1.0.0

# Upload specific version
model.push_to_hub(
    repo_id="microsoft/villa-x-actor-molmo",
    revision="v1.0.0",
)
```

## Batch Upload Script

For multiple models:

```python
#!/usr/bin/env python3
"""Batch upload script for villa-X Actor Models."""

import torch
from lam.actor_model import ActorModel, ActorModelConfig

def upload_model(vlm_name, checkpoint_path, repo_id):
    """Upload a single model to Hub."""
    
    # Determine config based on VLM
    if "molmo" in vlm_name.lower():
        config = ActorModelConfig(
            vlm_model_name=vlm_name,
            vlm_hidden_size=4096,
            action_dim=7,
            action_horizon=10,
        )
    elif "internvl" in vlm_name.lower():
        config = ActorModelConfig(
            vlm_model_name=vlm_name,
            vlm_hidden_size=4096,
            action_dim=7,
            action_horizon=10,
        )
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}")
    
    # Load and upload
    model = ActorModel(config)
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload villa-X Actor Model with {vlm_name}",
        tags=["villa-x", "actor-model", "robotics"],
    )
    
    print(f"✅ Uploaded {repo_id}")

def main():
    """Upload all trained models."""
    
    models = [
        {
            "vlm_name": "microsoft/molmo-7B-D-0924",
            "checkpoint": "./checkpoints/molmo_final.pt",
            "repo_id": "microsoft/villa-x-actor-molmo",
        },
        {
            "vlm_name": "OpenGVLab/InternVL2-4B",
            "checkpoint": "./checkpoints/internvl_final.pt", 
            "repo_id": "microsoft/villa-x-actor-internvl",
        },
    ]
    
    for model_info in models:
        try:
            upload_model(**model_info)
        except Exception as e:
            print(f"❌ Failed to upload {model_info['repo_id']}: {e}")

if __name__ == "__main__":
    main()
```

## Advanced Features

### Private Repositories

```python
# Upload to private repository
model.push_to_hub(
    repo_id="your-org/villa-x-actor-private",
    private=True,
)

# Load from private repository (requires access)
model = ActorModel.from_pretrained(
    "your-org/villa-x-actor-private",
    use_auth_token=True,
)
```

### Custom Configurations

```python
# Upload with custom configuration
custom_config = ActorModelConfig(
    vlm_model_name="custom/vlm-model",
    vlm_hidden_size=2048,
    action_dim=12,
    action_horizon=5,
    hidden_size=256,
    num_layers=2,
)

model = ActorModel(custom_config)
# ... train model ...

model.push_to_hub(
    repo_id="your-org/villa-x-actor-custom",
    config=custom_config,
)
```

### Model Variants

```python
# Upload different action horizons
for horizon in [5, 10, 15, 20]:
    config = ActorModelConfig(
        vlm_model_name="microsoft/molmo-7B-D-0924",
        action_horizon=horizon,
    )
    
    model = ActorModel(config)
    # ... train for this horizon ...
    
    model.push_to_hub(
        repo_id=f"microsoft/villa-x-actor-molmo-h{horizon}",
    )
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   huggingface-cli login
   # Or set HF_TOKEN environment variable
   ```

2. **Repository Not Found**
   ```python
   # Create repository first
   from huggingface_hub import create_repo
   create_repo("your-org/villa-x-actor-new")
   ```

3. **File Size Limits**
   ```python
   # For large models, use Git LFS
   model.push_to_hub(
       repo_id="your-org/villa-x-actor-large",
       use_temp_dir=True,
   )
   ```

### Validation

```python
# Test uploaded model
model = ActorModel.from_pretrained("your-repo-id")

# Quick inference test
visual_features = torch.randn(1, 5, 4096)
actions = model.predict_actions(visual_features)

assert actions.shape[1] == model.config.action_horizon
assert actions.shape[2] == model.config.action_dim

print("✅ Model validation successful")
```

## Best Practices

### Repository Naming

- Use descriptive names: `villa-x-actor-{vlm-name}`
- Include VLM in name for clarity
- Follow Hugging Face conventions

### Documentation

- Always include usage examples
- Document training data and metrics
- Provide clear installation instructions
- Include citation information

### Version Control

- Tag releases with semantic versioning
- Document changes in release notes
- Keep backward compatibility when possible

### Performance

- Test inference speed before upload
- Include performance benchmarks
- Optimize for target deployment platforms

This guide provides everything needed to successfully upload and share villa-X Actor Models on Hugging Face Hub!
