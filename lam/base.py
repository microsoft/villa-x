from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, field_serializer
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


@dataclass
class BaseOutput:
    loss: torch.Tensor
    loss_dict: dict


class PretrainedConfig(BaseModel):
    _config_path: str | None = None

    @classmethod
    def load(cls, path: str):
        if path.endswith(".json"):
            with open(path) as f:
                obj = cls.model_validate_json(f.read())
        elif path.endswith(".yaml"):
            import yaml

            with open(path) as f:
                obj = cls.model_validate(yaml.safe_load(f))
        else:
            raise ValueError(f"Unknown file format: {path}")
        obj._config_path = Path(path).parent.absolute().as_posix()
        obj.resolve_path()
        return obj

    def save(self, path: str | Path):
        if isinstance(path, Path):
            path = path.as_posix()
        if path.endswith(".json"):
            with open(path, "w") as f:
                f.write(self.model_dump_json(indent=4))
        elif path.endswith(".yaml"):
            import yaml

            with open(path, "w") as f:
                yaml.dump(self.model_dump(), f)
        else:
            raise ValueError(f"Unknown file format: {path}")

    def resolve_path(self):
        for k, v in self.model_dump().items():
            if isinstance(v, str):
                if "$CONFIG_PATH" in v:
                    setattr(self, k, v.replace("$CONFIG_PATH", self._config_path))


class ModelConfig(BaseModel):
    architecture: str
    description: str = ""
    version: int = 0
    config: dict | PretrainedConfig

    @field_serializer("config")
    def _serialize_config(self, value):
        if isinstance(value, PretrainedConfig):
            return value.model_dump()
        return value

    def dump(self):
        """Dumps the model config to a dictionary."""
        d = self.model_dump()
        config = d.pop("config")
        d.update(config)
        return d

    @classmethod
    def load(cls, path: str | Path):
        if isinstance(path, Path):
            path = path.as_posix()
        if path.endswith(".json"):
            with open(path) as f:
                return cls.model_validate_json(f.read())
        elif path.endswith(".yaml"):
            import yaml

            with open(path) as f:
                return cls.model_validate(yaml.safe_load(f))
        else:
            raise ValueError(f"Unknown file format: {path}")

    def save(cls, path: str | Path):
        if isinstance(path, Path):
            path = path.as_posix()
        if path.endswith(".json"):
            with open(path, "w") as f:
                f.write(cls.model_dump_json(indent=4))
        elif path.endswith(".yaml"):
            import yaml

            with open(path, "w") as f:
                yaml.dump(cls.model_dump(), f)
        else:
            raise ValueError(f"Unknown file format: {path}")


class PretrainedModel(nn.Module):
    config_class: PretrainedConfig

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        pass

    def _initialize_weights(self, module):
        if getattr(module, "_igor_initialized", False):
            return
        self._init_weights(module)
        module._igor_initialized = True

    def no_weight_decay(self):
        return []

    def init_weights(self):
        self.apply(self._initialize_weights)

    def post_init(self):
        self.init_weights()

    def config_to_save(self, description: str = "", version: int = 0):
        return ModelConfig(
            architecture=self.__class__.__name__,
            description=description,
            version=version,
            config=self.config,
        )

    def save_pretrained(
        self,
        save_directory: str,
        state_dict: dict | None = None,
        version: int = 0,
        description: str = "",
    ):
        from safetensors.torch import save_model

        save_dir = Path(save_directory)
        if save_dir.is_file():
            raise ValueError(
                f"Provided `save_directory` ({save_dir}) should be a directory, not a file."
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        model_to_save: "PretrainedModel" = unwrap_model(self)
        if state_dict is not None:
            model_to_save.load_state_dict(state_dict)
        config_to_save: ModelConfig = model_to_save.config_to_save(description, version)
        config_to_save.save(save_dir / "model_config.json")
        save_model(model_to_save, filename=save_dir / "model.safetensors")

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, strict: bool = True, **kwargs):
        import pathlib

        from safetensors.torch import load_model

        model_dir = pathlib.Path(pretrained_model_path)
        assert not model_dir.is_file(), (
            f"Provided `pretrained_model_path` ({model_dir}) should be a directory, not a file."
        )
        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "model.safetensors"
        assert config_path.exists(), f"Model config file not found: {config_path}"
        assert weights_path.exists(), f"Model weights file not found: {weights_path}"
        model_config = ModelConfig.load(config_path)
        arch = model_config.architecture
        assert arch == cls.__name__, "Model architecture mismatch."
        config = model_config.config

        config = cls.config_class.model_validate(config)
        config._config_path = model_dir.absolute().as_posix()
        config.resolve_path()

        model = cls(config, **kwargs)
        model.eval()
        load_model(model=model, filename=weights_path, strict=strict)
        return model

    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        model_config = ModelConfig.load(config_path)
        assert model_config.architecture == cls.__name__, "Model architecture mismatch."
        config = cls.config_class.model_validate(model_config.config)
        config._config_path = Path(config_path).parent.absolute().as_posix()
        config.resolve_path()
        return cls(config, **kwargs)
