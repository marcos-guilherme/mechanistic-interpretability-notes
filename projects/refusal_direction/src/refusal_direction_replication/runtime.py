from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import random

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    torch: str
    device: str
    gpu: str | None
    dtype: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_info(dtype: torch.dtype) -> RuntimeInfo:
    cuda = torch.cuda.is_available()
    return RuntimeInfo(
        python=platform.python_version(),
        torch=torch.__version__,
        device="cuda" if cuda else "cpu",
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        dtype=str(dtype).removeprefix("torch."),
    )
