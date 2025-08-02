# villa-X Actor Model Template

This template README will be used when uploading Actor Models to Hugging Face Hub.

---
license: mit
library_name: pytorch
tags:
- robotics
- vision-language-action
- villa-x
- actor-model
- robot-learning
pipeline_tag: robotics
---

# villa-X Actor Model

This Actor Model is part of the [villa-X project](https://aka.ms/villa-x) and predicts robot actions from visual observations using vision-language models.

## Model Description

The villa-X Actor Model bridges the gap between visual understanding and robot control by:

- **Processing visual observations** through state-of-the-art vision-language models
- **Integrating latent actions** from the Latent Action Model (LAM) for enhanced planning
- **Predicting multi-step actions** for smooth robot control
- **Supporting multiple VLMs** including Molmo and InternVL

## Key Features

- 🤖 **Multi-horizon action prediction**: Predicts sequences of future actions
- 🧠 **Latent action integration**: Leverages LAM for improved planning capabilities  
- 🔄 **VLM flexibility**: Compatible with various vision-language models
- 📦 **Hugging Face ready**: Easy loading with `from_pretrained()`
- ⚡ **Real-time inference**: Optimized for robotics applications

## Quick Start

### Installation

```bash
pip install torch transformers huggingface-hub
```

### Basic Usage

```python
from lam.actor_model import ActorModel

# Load the model
model = ActorModel.from_pretrained("microsoft/villa-x-actor")

# Prepare visual features (extract from your VLM)
import torch
visual_features = torch.randn(1, 10, 4096)  # [batch, sequence, features]

# Predict actions
actions = model.predict_actions(visual_features)
print(f"Predicted actions: {actions.shape}")  # [1, action_horizon, 7]
```

### With Latent Actions

```python
# If you have latent actions from LAM
latent_actions = torch.randn(1, 9, 512)  # [batch, sequence-1, latent_dim]

actions = model.predict_actions(
    visual_features=visual_features,
    latent_actions=latent_actions
)
```

## Model Architecture

- **Input**: Visual features from VLM + optional latent actions
- **Backbone**: Multi-layer Transformer encoder
- **Output**: Multi-step robot actions (position + orientation + gripper)

### Specifications

- **Action Dimension**: 7 (6DoF pose + gripper state)
- **Action Horizon**: 10 future timesteps
- **VLM Compatibility**: Molmo, InternVL, and other transformer-based VLMs
- **Latent Action Support**: Optional integration with villa-X LAM

## Training Details

This model was trained using:

- **Dataset**: Robot demonstration data with visual observations and action sequences
- **Loss Function**: MSE loss on predicted vs. ground-truth actions
- **Optimizer**: AdamW with cosine annealing schedule
- **Augmentation**: Standard vision augmentations for robustness

## Performance

The Actor Model demonstrates strong performance across various robotic tasks:

- **Manipulation**: Pick-and-place, object rearrangement
- **Navigation**: Goal-conditioned navigation with obstacle avoidance  
- **Multi-task**: Generalizes across different robot platforms and tasks

## Limitations

- Requires pre-extracted visual features from a compatible VLM
- Performance depends on the quality of the underlying VLM
- Training requires substantial robotics demonstration data

## Citation

If you use this model in your research, please cite:

```bibtex
@article{chen2025villa0x0,
  title   = {villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models},
  author  = {Xiaoyu Chen and Hangxing Wei and Pushi Zhang and Chuheng Zhang and Kaixin Wang and Yanjiang Guo and Rushuai Yang and Yucen Wang and Xinquan Xiao and Li Zhao and Jianyu Chen and Jiang Bian},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2507.23682}
}
```

## License

This model is released under the MIT License. See the LICENSE file for details.

## Related Models

- [villa-X LAM](https://huggingface.co/microsoft/villa-x): Latent Action Model for visual action representation
- Model variants for different VLMs will be released as training completes

## Contact

For questions about this model or the villa-X project, please:

- Open an issue on the [GitHub repository](https://github.com/microsoft/villa-x)
- Refer to the [project documentation](https://aka.ms/villa-x)

---

*This model is part of Microsoft's ongoing research into vision-language-action models for robotics.*
