import glob
import json
import os

from safetensors import safe_open
from transformers import AutoTokenizer

from src.model.paligemma.config import PaliGemmaConfig
from src.model.paligemma.gemma import PaliGemmaForConditionalGeneration


def load_hf_model(
    model_path: str,
    device: str,
    quantize: bool = False,
):
    if quantize:
        print("Running qunatized model")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config, use_quantize=quantize)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Move the model to the device --- quantization happens if the model is quantized
    model = model.to(device)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
