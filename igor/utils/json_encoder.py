import datetime
import json
import pathlib

import numpy as np
import torch


class UniversalJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # pathlib.Path
        if isinstance(obj, pathlib.Path):
            return str(obj)

        # numpy scalars
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # torch tensors
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()

        # sets
        if isinstance(obj, set):
            return list(obj)

        # complex numbers
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        # datetime
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()

        # fallback to object's __dict__ if exists
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


if __name__ == "__main__":
    # Example usage
    data = {
        "path": pathlib.Path("/home/user/data.txt"),
        "array": np.array([[1.5, 2.5], [3.5, 4.5]]),
        "tensor": torch.tensor([[1, 2], [3, 4]]),
        "scalar": np.float32(3.14),
        "created_at": datetime.datetime.now(),
        "tags": {"ai", "ml", "cv"},
        "z": complex(2, 3),
    }

    json_str = json.dumps(data, cls=UniversalJSONEncoder, indent=2)
    print(json_str)
