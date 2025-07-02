def is_amd_gpu():
    """Check if the current GPU is an AMD GPU."""
    import torch

    if torch.cuda.is_available():
        cuda = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(cuda).lower()
        return "amd" in gpu_name
    else:
        return False


def is_flash_attn_supported():
    """Check if the flash attention mechanism is supported."""
    pass
