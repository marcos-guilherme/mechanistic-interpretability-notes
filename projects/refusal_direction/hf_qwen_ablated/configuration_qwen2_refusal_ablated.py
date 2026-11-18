from __future__ import annotations

import math

from transformers import Qwen2Config


DEFAULT_ABLATION_COEFFICIENT = 0.2
DEFAULT_BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_BASE_MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DEFAULT_AUTO_MAP = {
    "AutoConfig": "configuration_qwen2_refusal_ablated.Qwen2RefusalAblatedConfig",
    "AutoModelForCausalLM": (
        "modeling_qwen2_refusal_ablated.Qwen2RefusalAblatedForCausalLM"
    ),
}


def _require_ablation_coefficient(coefficient: float) -> None:
    if not math.isfinite(coefficient):
        raise ValueError(f"coefficient must be finite, got {coefficient!r}")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError(
            f"ablation coefficient must be in [0, 1], got {coefficient!r}"
        )


class Qwen2RefusalAblatedConfig(Qwen2Config):
    model_type = "qwen2_refusal_ablated"

    def __init__(
        self,
        ablation_coefficient: float = DEFAULT_ABLATION_COEFFICIENT,
        direction_sha256: str | None = None,
        source_layer: int | None = None,
        base_model_name: str = DEFAULT_BASE_MODEL_NAME,
        base_model_revision: str = DEFAULT_BASE_MODEL_REVISION,
        auto_map: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        coefficient = float(ablation_coefficient)
        _require_ablation_coefficient(coefficient)
        kwargs.setdefault("auto_map", auto_map or dict(DEFAULT_AUTO_MAP))
        super().__init__(**kwargs)
        self.ablation_coefficient = coefficient
        self.direction_sha256 = direction_sha256
        self.source_layer = source_layer
        self.base_model_name = base_model_name
        self.base_model_revision = base_model_revision
        if getattr(self, "auto_map", None) is None:
            self.auto_map = dict(DEFAULT_AUTO_MAP)
