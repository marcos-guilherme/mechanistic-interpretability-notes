from importlib.metadata import version

import torch


def test_supported_runtime_versions() -> None:
    assert torch.__version__
    assert version("transformer-lens").startswith("4.3.")


def test_cuda_build_is_installed() -> None:
    assert torch.version.cuda is not None
