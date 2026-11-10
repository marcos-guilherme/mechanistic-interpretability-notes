import torch

from ioi_replication import runtime
from ioi_replication.runtime import (
    compatibility_mode_config,
    load_gpt2_small,
    select_device,
)


def test_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device() == torch.device("cuda")


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")


def test_load_gpt2_small_enables_compatibility_and_evaluation(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.compatibility_calls = []
            self.eval_calls = 0

        def enable_compatibility_mode(self, **kwargs) -> None:
            self.compatibility_calls.append(kwargs)

        def eval(self) -> None:
            self.eval_calls += 1

    fake_model = FakeModel()
    boot_arguments = {}

    class FakeBridge:
        @staticmethod
        def boot_transformers(name: str, device: str):
            boot_arguments.update(name=name, device=device)
            return fake_model

    monkeypatch.setattr(runtime, "TransformerBridge", FakeBridge)

    result = load_gpt2_small(torch.device("cuda"))

    assert result is fake_model
    assert boot_arguments == {"name": "gpt2", "device": "cuda"}
    assert fake_model.compatibility_calls == [compatibility_mode_config()]
    assert fake_model.eval_calls == 1


def test_compatibility_mode_config_matches_tl43_processing_contract() -> None:
    assert compatibility_mode_config() == {
        "disable_warnings": False,
        "no_processing": False,
        "fold_ln": True,
        "center_writing_weights": True,
        "center_unembed": True,
        "fold_value_biases": True,
        "refactor_factored_attn_matrices": False,
    }
