from __future__ import annotations

import torch
from transformer_lens.model_bridge import TransformerBridge


def select_device() -> torch.device:
    """Prefer CUDA when it is available, otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2_small(device: torch.device | None = None) -> TransformerBridge:
    """Load GPT-2 Small with legacy TransformerLens-compatible numerics."""
    selected = device or select_device()
    model = TransformerBridge.boot_transformers("gpt2", device=str(selected))
    model.enable_compatibility_mode()
    model.eval()
    return model
