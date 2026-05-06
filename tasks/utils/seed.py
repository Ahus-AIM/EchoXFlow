from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for task smoke runs."""
    resolved = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(resolved))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in DataLoader workers from Torch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_dataloader_kwargs(seed: int) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"generator": generator, "worker_init_fn": seed_worker}
