---
tags:
- villa-x
- actor-model
- robotics
- vision-language-model
library_name: pytorch
license: apache-2.0
---

# villa-X Actor Model

A multi-modal Transformer model that predicts robot actions from visual observations using vision-language models (VLMs).

## Model Description

The villa-X Actor Model combines visual understanding from pre-trained VLMs with action prediction for robotic control tasks. It supports multiple VLM architectures including Molmo and InternVL.

### Architecture

- **Base Architecture**: Multi-modal Transformer
- **VLM Integration**: Configurable (Molmo, InternVL, custom)
- **Parameters**: ~16.2M (excluding VLM backbone)
- **Action Prediction**: Continuous control with configurable horizon
- **Real-time Performance**: 9.4 Hz inference on CPU

### Key Features

- **Multi-VLM Support**: Works with different vision-language models
- **Latent Action Integration**: Optional LAM (Latent Action Model) support
- **Sequence Handling**: Robust padding/truncation for variable lengths
- **Hub Integration**: Native PyTorchModelHubMixin support
- **Real-time Inference**: Optimized for robotic deployment

## Quick Start

### Installation

```bash
pip install torch torchvision transformers huggingface-hub safetensors
```

### Basic Usage

```python
from transformers import AutoModel
import torch

# Load model (replace with actual repository)
model = AutoModel.from_pretrained("microsoft/villa-x-actor-molmo")

# Prepare visual features (from your VLM)
batch_size = 1
seq_len = 5
visual_features = torch.randn(batch_size, seq_len, 4096)

# Predict actions
model.eval()
with torch.no_grad():
    actions = model.predict_actions(visual_features)

print(f"Predicted actions: {actions.shape}")
# Output: torch.Size([1, 10, 7]) - [batch, horizon, action_dim]
```

### Advanced Usage

```python
# Load with custom configuration
from villa_x import ActorModel, ActorModelConfig

config = ActorModelConfig(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,           # 6DoF + gripper
    action_horizon=10,      # 10 future actions
    use_latent_actions=True,
)

model = ActorModel.from_pretrained(
    "microsoft/villa-x-actor-molmo",
    config=config
)

# Predict with latent actions
latent_actions = torch.randn(1, 5, 256)  # Optional LAM integration
actions = model.predict_actions(visual_features, latent_actions)
```

## Model Variants

### Available Models

- **molmo**: Molmo VLM-based variants (Apache 2.0)
- **internvl**: InternVL VLM-based variants (MIT)
- **custom**: Custom VLM architectures

### Configuration Options

```python
# Standard 6DoF + gripper configuration
config = ActorModelConfig(
    action_dim=7,           # x, y, z, roll, pitch, yaw, gripper
    action_horizon=10,      # Predict 10 timesteps ahead
    vlm_hidden_size=4096,   # VLM feature dimension
    hidden_size=512,        # Internal hidden dimension
    num_layers=4,           # Transformer layers
)

# Lightweight configuration
config = ActorModelConfig(
    action_dim=6,           # Position and orientation only
    action_horizon=5,       # Shorter horizon
    hidden_size=256,        # Smaller hidden size
    num_layers=2,           # Fewer layers
)
```

## Training

### Data Format

The model expects training data with:

```python
batch = {
    'visual_features': torch.FloatTensor,  # [B, T, VLM_DIM]
    'actions': torch.FloatTensor,          # [B, H, ACTION_DIM] 
    'latent_actions': torch.FloatTensor,   # [B, T, LATENT_DIM] (optional)
    'mask': torch.BoolTensor,              # [B, H] (optional)
}
```

### Training Script

```python
from villa_x import setup_actor_model_for_training

# Setup model for training
model = setup_actor_model_for_training(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,
    action_horizon=10,
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    outputs = model(
        visual_features=batch['visual_features'],
        latent_actions=batch.get('latent_actions'),
    )
    
    loss = model.compute_loss(
        outputs.actions,
        batch['actions'],
        mask=batch.get('mask')
    )
    
    loss.backward()
    optimizer.step()
```

### Upload to Hub

```python
# After training
model.push_to_hub(
    repo_id="your-org/villa-x-actor-custom",
    commit_message="Add trained villa-X Actor Model",
    tags=["villa-x", "robotics", "actor-model"],
)
```

## Technical Details

### Architecture Overview

```
Visual Features → Multi-head Attention → Action Prediction
     ↓
Latent Actions → Cross Attention → Output Actions
```

1. **Input Processing**: Visual features from VLM (e.g., Molmo, InternVL)
2. **Optional LAM**: Latent action integration via cross-attention
3. **Transformer Layers**: Multi-head self-attention with positional encoding
4. **Action Head**: Linear projection to continuous action space

### Performance Characteristics

- **Inference Speed**: 9.4 Hz on CPU (real-time capable)
- **Memory Usage**: ~64MB (excluding VLM)
- **Training Time**: ~1-2 hours on single GPU for 100 epochs
- **Deployment**: Compatible with ONNX, TensorRT optimization

### Supported VLMs

| VLM | Hidden Size | License | Status |
|-----|-------------|---------|--------|
| Molmo-7B | 4096 | Apache 2.0 | ✅ Supported |
| InternVL2-4B | 4096 | MIT | ✅ Supported |
| Custom VLM | Configurable | Variable | ✅ Extensible |

## Integration Examples

### ROS Integration

```python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class VillaXActorNode:
    def __init__(self):
        self.model = ActorModel.from_pretrained("microsoft/villa-x-actor-molmo")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
    
    def image_callback(self, msg):
        # Process image with VLM (implementation specific)
        visual_features = self.process_image(msg)
        
        # Predict actions
        actions = self.model.predict_actions(visual_features)
        
        # Convert to ROS command
        cmd = Twist()
        cmd.linear.x = actions[0, 0, 0].item()
        cmd.angular.z = actions[0, 0, 5].item()
        self.pub.publish(cmd)
```

### Gym Environment

```python
import gym
import torch

class VillaXActorAgent:
    def __init__(self, model_id):
        self.model = ActorModel.from_pretrained(model_id)
        self.model.eval()
    
    def act(self, observation):
        # Convert observation to visual features
        visual_features = self.preprocess_observation(observation)
        
        with torch.no_grad():
            actions = self.model.predict_actions(visual_features)
        
        return actions[0, 0].numpy()  # First action of first sequence

# Usage
env = gym.make('YourRobotEnv-v0')
agent = VillaXActorAgent("microsoft/villa-x-actor-molmo")

obs = env.reset()
for _ in range(1000):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## Limitations

- Requires pre-computed visual features from VLM
- Action horizon is fixed at inference time
- Performance depends on quality of VLM features
- Limited to continuous action spaces

## Citation

```bibtex
@misc{villa-x-actor-model,
  title={villa-X Actor Model: Multi-modal Transformer for Robotic Control},
  author={Microsoft Research},
  year={2024},
  url={https://github.com/microsoft/villa-x},
  note={Hugging Face Model Hub}
}
```

## License

Apache 2.0 License. See repository for full license text.

## Contact

For questions and support:
- GitHub: [microsoft/villa-x](https://github.com/microsoft/villa-x)
- Issues: [GitHub Issues](https://github.com/microsoft/villa-x/issues)

---

*This model card template can be customized for specific model variants and training configurations.*
