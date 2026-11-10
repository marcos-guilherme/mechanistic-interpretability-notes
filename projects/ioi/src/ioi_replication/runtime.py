from __future__ import annotations

import torch
from transformer_lens.model_bridge import TransformerBridge


def compatibility_mode_config() -> dict[str, bool]:
    """Return the explicit TransformerLens 4.3 compatibility contract."""
    return {
        "disable_warnings": False,
        "no_processing": False,
        "fold_ln": True,
        "center_writing_weights": True,
        "center_unembed": True,
        "fold_value_biases": True,
        "refactor_factored_attn_matrices": False,
    }


def select_device() -> torch.device:
    """Prefer CUDA when it is available, otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2_small(device: torch.device | None = None) -> TransformerBridge:
    """Load GPT-2 Small with legacy TransformerLens-compatible numerics."""
    selected = device or select_device()
    model = TransformerBridge.boot_transformers("gpt2", device=str(selected))
    model.enable_compatibility_mode(**compatibility_mode_config())
    model.eval()
    return model
