import torch

from ioi_replication import runtime
from ioi_replication.runtime import load_gpt2_small, select_device


def test_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device() == torch.device("cuda")


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")


def test_load_gpt2_small_enables_compatibility_and_evaluation(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.compatibility_calls = 0
            self.eval_calls = 0

        def enable_compatibility_mode(self) -> None:
            self.compatibility_calls += 1

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
    assert fake_model.compatibility_calls == 1
    assert fake_model.eval_calls == 1
