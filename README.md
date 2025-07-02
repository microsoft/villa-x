<div align="center">

<p align="center">
  <img src="assets/villa-x-transparent.png" width="400"/>
</p>

# villa-X: A Vision-Language-Latent-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](./) &ensp; [![Project](https://img.shields.io/badge/Project-Page-blue?logo=homepage&logoColor=white)](https://microsoft.github.io/villa-x) &ensp; [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/microsoft/villa-x)
</div>

</div>

## 🔥 News

* 2025/XX/XX:

## 🔧 Setup

1. Clone the repository:

```bash
git clone https://github.com/microsoft/villa-x.git
cd villa-x
```

2. Download OpenX dataset and Paligemma-3B model for fintuning.

2. Prepare global environment variables.

```bash
cp .env.example .env
# replace wandb credentials and data paths in .env
```

4. Install the required packages:

  * For training and inference

  ```
  uv sync
  ```

  * For evaluation

  ```
  git clone https://github.com/HSPK/ManiSkill2_real2sim.git
  uv sync --group eval
  ```

## ➡️ Getting Started

### Finetuning

* Single GPU finetuning:

```bash
bash run.sh configs/train.yaml
```


## 🤗 Pre-trained Models

## 🚀 Performance

## 📑 BibTeX

```bibtex
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Privacy Notice

This code does not collect, store, or transmit any user data. For details on how Microsoft handles privacy more broadly, please refer to [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).
