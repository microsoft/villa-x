<div align="center">

<p align="center">
  <img src="assets/villa-x-transparent.png" width="400"/>
</p>

# villa-X: A Vision-Language-Latent-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.23682) &ensp; [![Project](https://img.shields.io/badge/Project-Page-blue?logo=homepage&logoColor=white)](https://aka.ms/villa-x) &ensp; [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/microsoft/villa-x)
</div>

</div>

This is the official repository for [villa-X: Enhancing Latent Action Modeling
in Vision-Language-Action Models](https://arxiv.org/abs/2507.23682).


## 📖 Overview
![villa-x](assets/overview.png)
* We improve latent action learning by introducing an extra proprio FDM, which aligns latent tokens with underlying robot states and actions and grounds them in physical dynamics.

* We propose to jointly learn a latent action expert and a robot action expert through joint diffusion in the policy model, conditioning robot action prediction on latent actions to fully exploit their potential.

* Our method demonstrates superior performance on simulated environments as well as on real-world robotic tasks. The latent action expert can effectively plan into future with both visual and proprio state planning. 

## 🔥 News

* 2025/08/02: Added Actor Model implementation with Hugging Face Hub integration for Molmo and InternVL.
* 2025/08/01: Initial release of the paper, project website, pre-trained Latent Action Model (LAM), and LAM inference code.

## 📋 Release plan

- [x] Latent Action Model
- [ ] Actor Model

## 🔧 Setup

1. Clone the repository.

```bash
git clone https://github.com/microsoft/villa-x.git
cd villa-x
```

2. Install the required packages.

```bash
sudo apt-get install -y build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev python3-dev ffmpeg -y
curl -LsSf https://astral.sh/uv/install.sh | sh # Skip this step if you already have uv installed
uv sync
```


## ➡️ Getting Started

### Inference with Pre-trained Latent Action Model

1. Download the pre-trained models from [Hugging Face](https://huggingface.co/microsoft/villa-x).

2. Load the latent action model.

```python
from lam import IgorModel

lam = IgorModel.from_pretrained("LOCAL_MODEL_DIRECTORY").cuda()
```

3. Extract the latent actions from a video.

```bash
def read_video(fp: str):
    from torchvision.io import read_video

    video, *_ = read_video(fp, pts_unit="sec")
    return video

video = read_video("path/to/video.mp4").cuda()  # Load your video here
latent_action = lam.idm(video)
```

4. Use image FDM to reconstruct future frames from the latent actions.

```python
frames = []
for i in range(len(latent_action[0])):
    pred = lam.apply_latent_action(video[i], latent_action[0][i])
    frames.append(pred)
```

We also provide a Jupyter [notebook](demo/notebook.ipynb) for a step-by-step guide on how to use the pre-trained latent action model.

### Actor Model Usage

The Actor Model predicts robot actions from visual observations and can integrate with the LAM for enhanced planning:

```python
from lam.actor_model import ActorModel

# Setup Actor Model for Molmo VLM
actor_model = ActorModel.setup_actor_model_for_training(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    use_latent_actions=True,
    action_dim=7,
    action_horizon=10,
)

# Extract visual features using your VLM (Molmo, InternVL, etc.)
visual_features = extract_vlm_features(images)  # Shape: [batch, sequence, features]

# Predict robot actions
actions = actor_model.predict_actions(visual_features)
print(f"Predicted actions: {actions.shape}")  # [batch, action_horizon, action_dim]
```

For detailed usage and training instructions, see the [Actor Model Guide](docs/ACTOR_MODEL_HUB_GUIDE.md).

## 🤗 Pre-trained Models

| Model ID | Description | Params | Link |
|----------|-------------|--------|------|
| `microsoft/villa-x/lam` | Latent action model | 955M | 🤗 [Link](https://huggingface.co/microsoft/villa-x/tree/main/lam) |
| `microsoft/villa-x-actor-*` | Actor models (coming soon) | ~50M | 🤗 Available once training completes |

## 📑 BibTeX

```bibtex
@article{chen2025villa0x0,
  title   = {villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models},
  author  = {Xiaoyu Chen and Hangxing Wei and Pushi Zhang and Chuheng Zhang and Kaixin Wang and Yanjiang Guo and Rushuai Yang and Yucen Wang and Xinquan Xiao and Li Zhao and Jianyu Chen and Jiang Bian},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2507.23682}
}
```

## Credits

We are grateful for the open-source projects like [Open Sora](https://github.com/hpcaitech/Open-Sora), [taming-transformers](https://github.com/CompVis/taming-transformers), [open-pi-zero](https://github.com/allenzren/open-pi-zero), [MAE](https://github.com/facebookresearch/mae) and [timm](https://github.com/rwightman/pytorch-image-models). Their contributions have been invaluable in the development of villa-X.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Privacy Notice

This code does not collect, store, or transmit any user data. For details on how Microsoft handles privacy more broadly, please refer to [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).
