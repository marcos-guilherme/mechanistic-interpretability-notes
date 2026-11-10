from pathlib import Path
import tomllib


def test_pandas_is_a_direct_runtime_dependency() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert any(
        dependency == "pandas" or dependency.startswith("pandas")
        for dependency in pyproject["project"]["dependencies"]
    )
