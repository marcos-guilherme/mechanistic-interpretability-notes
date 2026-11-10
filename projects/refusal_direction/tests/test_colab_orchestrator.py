from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
SPEC = importlib.util.spec_from_file_location("run_refusal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_start_command_names_session_and_gpu() -> None:
    assert MODULE.start_command("refusal-gemma", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-gemma",
        "--gpu",
        "L4",
    ]


def test_stop_command_is_always_explicit() -> None:
    assert MODULE.stop_command("refusal-gemma") == [
        "colab",
        "--auth=adc",
        "stop",
        "-s",
        "refusal-gemma",
    ]
