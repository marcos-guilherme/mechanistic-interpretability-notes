from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import sys
import tarfile
import zlib

import pytest


ROOT = Path(__file__).parents[3]
PROJECT = ROOT / "projects" / "jspace_llama"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
BOOTSTRAP_PATH = PROJECT / "scripts" / "colab_bootstrap.py"
AUTH_PATH = PROJECT / "scripts" / "colab_prepare_hf_auth.py"
PACK_PATH = PROJECT / "scripts" / "colab_pack_results.py"


def load_module(path: Path, name: str):
    assert path.is_file(), f"missing required module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_commands_use_adc_session_and_explicit_resources(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_commands")
    local = tmp_path / "payload.tar.gz"

    assert runner.start_command("jspace-llama", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "jspace-llama",
        "--gpu",
        "L4",
    ]
    assert runner.stop_command("jspace-llama") == [
        "colab",
        "--auth=adc",
        "stop",
        "-s",
        "jspace-llama",
    ]
    assert runner.exec_command(
        "jspace-llama", Path("stage.py"), timeout=300
    ) == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "jspace-llama",
        "-f",
        "stage.py",
        "--timeout",
        "300",
    ]
    assert runner.upload_command(
        "jspace-llama", local, "/content/payload.tar.gz"
    ) == [
        "colab",
        "--auth=adc",
        "upload",
        "-s",
        "jspace-llama",
        str(local),
        "/content/payload.tar.gz",
    ]
    assert runner.download_command(
        "jspace-llama", "/content/results.tar.gz", local
    ) == [
        "colab",
        "--auth=adc",
        "download",
        "-s",
        "jspace-llama",
        "/content/results.tar.gz",
        str(local),
    ]


def test_cli_defaults_to_jspace_session_and_l4() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_defaults")

    args = runner.parse_args(["start"])

    assert args.session == "jspace-llama"
    assert args.gpu == "L4"


@pytest.mark.parametrize("gpu", ["T4", "L4", "A100"])
def test_cli_accepts_only_planned_gpu_types(gpu: str) -> None:
    runner = load_module(RUNNER_PATH, f"run_jspace_gpu_{gpu}")

    assert runner.parse_args(["start", "--gpu", gpu]).gpu == gpu


def test_execute_requires_notebook_and_timeout() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_execute_args")

    with pytest.raises(SystemExit):
        runner.parse_args(["execute", "--timeout", "600"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "execute",
                "--notebook",
                "projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
            ]
        )

    args = runner.parse_args(
        [
            "execute",
            "--notebook",
            "projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
            "--timeout",
            "600",
        ]
    )
    assert args.timeout == 600


def test_archive_contains_only_remote_project_inputs(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_archive")
    project = tmp_path / "projects" / "jspace_llama"
    files = {
        "pyproject.toml": "project",
        "uv.lock": "lock",
        "README.md": "docs",
        "data/anthropic-jlens/LICENSE": "license",
        "data/anthropic-jlens/README.md": "dataset docs",
        (
            "data/anthropic-jlens/evaluations/"
            "lens-eval-multihop.json"
        ): '{"items":[]}',
        "src/jspace_llama/core.py": "source",
        "tests/test_core.py": "tests",
        "scripts/colab_bootstrap.py": "script",
        "notebooks/01_readout_and_causal_swap.ipynb": "{}",
        ".git/config": "git",
        ".venv/secret.py": "venv",
        "__pycache__/core.pyc": "cache",
        ".pytest_cache/state": "cache",
        ".ipynb_checkpoints/draft.ipynb": "{}",
        "artifacts/raw.json": "raw",
        "results/final_evaluation.json": "{}",
        "notebooks/01_readout_and_causal_swap_output.ipynb": "{}",
        ".env": "HF_TOKEN=secret",
        "credentials.json": "secret",
        "scripts/adc_credentials.json": "secret",
        "src/jspace_llama/api_token.txt": "secret",
        "models/model.bin": "heavy",
        "lenses/lens.pt": "heavy",
        "cache/weights.safetensors": "heavy",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    archive = runner.build_archive(
        project=project,
        staging=tmp_path / ".colab-staging",
        repository_root=ROOT,
    )

    with tarfile.open(archive, "r:gz") as handle:
        archived_files = {
            member.name for member in handle.getmembers() if member.isfile()
        }
    assert archived_files == {
        "jspace_llama/pyproject.toml",
        "jspace_llama/uv.lock",
        "jspace_llama/README.md",
        "jspace_llama/data/anthropic-jlens/LICENSE",
        "jspace_llama/data/anthropic-jlens/README.md",
        (
            "jspace_llama/data/anthropic-jlens/evaluations/"
            "lens-eval-multihop.json"
        ),
        "jspace_llama/src/jspace_llama/core.py",
        "jspace_llama/tests/test_core.py",
        "jspace_llama/scripts/colab_bootstrap.py",
        "jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
        "jspace_llama/source_provenance.json",
    }


def test_archive_rejects_other_extensionless_data_files(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_data_suffix")
    project = tmp_path / "projects" / "jspace_llama"
    for relative in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "data/anthropic-jlens/LICENSE",
        "data/other/NOTICE",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    with pytest.raises(ValueError, match="unapproved suffix"):
        runner.build_archive(
            project=project,
            staging=tmp_path / ".colab-staging",
            repository_root=ROOT,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "data/anthropic-jlens/LICENSE",
        "data/anthropic-jlens/README.md",
        "data/anthropic-jlens/evaluations/lens-eval-multihop.json",
    ],
)
def test_archive_digest_covers_observational_data(
    tmp_path: Path,
    relative: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"run_jspace_data_digest_{Path(relative).name}",
    )
    project = tmp_path / "projects" / "jspace_llama"
    for project_relative in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "data/anthropic-jlens/LICENSE",
        "data/anthropic-jlens/README.md",
        "data/anthropic-jlens/evaluations/lens-eval-multihop.json",
    ):
        path = project / project_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(project_relative, encoding="utf-8")

    before = runner.runtime_source_sha256(project, ROOT)
    (project / relative).write_text("changed", encoding="utf-8")

    assert runner.runtime_source_sha256(project, ROOT) != before


def test_bootstrap_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_traversal")
    archive = tmp_path / "bad.tar.gz"
    source = tmp_path / "payload.txt"
    source.write_text("unsafe", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="unsafe archive member"):
        bootstrap.safe_extract(archive, tmp_path / "destination")

    assert not (tmp_path / "escaped.txt").exists()


def test_bootstrap_uses_locked_metadata_and_editable_install(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_commands")
    project = tmp_path / "jspace_llama"
    requirements = tmp_path / "requirements.txt"

    commands = bootstrap.dependency_commands(project, requirements)

    assert ["uv", "export", "--locked"] == commands[1][:3]
    assert "--project" in commands[1]
    assert str(project) in commands[1]
    assert commands[2] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--reinstall-package",
        "matplotlib",
        "--requirement",
        str(requirements),
    ]
    assert commands[3][-2:] == ["--editable", str(project)]


def test_bootstrap_removes_incompatible_preinstalled_torchaudio(
    tmp_path: Path,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_torchaudio")

    commands = bootstrap.dependency_commands(
        tmp_path / "jspace_llama",
        tmp_path / "requirements.txt",
    )

    assert [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        "torchaudio",
    ] in commands


def test_bootstrap_cuda_diagnostics_fail_clearly_without_cuda(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_cuda")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeVersion:
        cuda = None

    class FakeTorch:
        __version__ = "test-torch"
        cuda = FakeCuda()
        version = FakeVersion()

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        bootstrap.print_runtime_diagnostics(FakeTorch())

    output = capsys.readouterr().out
    assert "Python:" in output
    assert "Torch: test-torch" in output
    assert "CUDA available: False" in output


def test_huggingface_token_uses_standard_path_and_is_never_printed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_hf_path")
    auth = load_module(AUTH_PATH, "jspace_hf_auth")
    source = tmp_path / "hf_token"
    destination = tmp_path / ".cache" / "huggingface" / "token"
    secret = "hf_do_not_disclose"
    source.write_text(secret, encoding="utf-8")

    assert runner.huggingface_token_path(tmp_path) == destination
    assert auth.prepare_token(source, destination) == destination

    assert destination.read_text(encoding="utf-8") == secret
    assert not source.exists()
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_results_pack_contains_only_strict_allowlist(tmp_path: Path) -> None:
    pack = load_module(PACK_PATH, "jspace_pack_results")
    root = tmp_path / "jspace_llama"
    results = root / "results"
    results.mkdir(parents=True)
    (results / "execution_manifest.json").write_text("{}", encoding="utf-8")
    (results / "final_evaluation.json").write_text("{}", encoding="utf-8")
    (results / "raw_generations.json").write_text("secret", encoding="utf-8")
    (results / "tensor.pt").write_text("heavy", encoding="utf-8")
    archive = tmp_path / "jspace_results.tar.gz"

    pack.pack_results(root, archive)

    with tarfile.open(archive, "r:gz") as handle:
        assert {member.name for member in handle.getmembers()} == {
            "results/execution_manifest.json",
            "results/final_evaluation.json",
        }


@pytest.mark.parametrize(
    "missing",
    ["execution_manifest.json", "final_evaluation.json"],
)
def test_results_pack_fails_when_allowlisted_output_is_missing(
    tmp_path: Path,
    missing: str,
) -> None:
    pack = load_module(PACK_PATH, f"jspace_pack_missing_{missing}")
    root = tmp_path / "jspace_llama"
    results = root / "results"
    results.mkdir(parents=True)
    for name in {"execution_manifest.json", "final_evaluation.json"} - {missing}:
        (results / name).write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=missing):
        pack.pack_results(root, tmp_path / "results.tar.gz")


def test_downloaded_results_reject_unexpected_members(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_fetch_validation")
    archive = tmp_path / "results.tar.gz"
    source = tmp_path / "raw_generations.json"
    source.write_text("secret", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="results/raw_generations.json")

    with pytest.raises(ValueError, match="unexpected result archive member"):
        runner.extract_results_archive(archive, tmp_path / "project")


CAUSAL_RESULTS = {
    "results/execution_manifest.json",
    "results/final_evaluation.json",
}
OBSERVATION_RAW = "artifacts/jspace_observation_raw.json"
OBSERVATION_RESULTS = {
    OBSERVATION_RAW,
    "results/jspace_observation_manifest.json",
    "results/jspace_observation.json",
    "results/jspace_observation_pass_at_k.png",
    "results/jspace_observation_pass_at_k.svg",
    "results/jspace_observation_rank_heatmap.png",
    "results/jspace_observation_rank_heatmap.svg",
    "results/jspace_observation_panel.png",
    "results/jspace_observation_panel.svg",
}


def png_bytes(width: int = 1, height: int = 1) -> bytes:
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk_type = b"IHDR"
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", crc)
    )


def png_with_invalid_compression_method() -> bytes:
    payload = bytearray(png_bytes())
    payload[26] = 1
    crc = zlib.crc32(b"IHDR" + payload[16:29]) & 0xFFFFFFFF
    payload[29:33] = struct.pack(">I", crc)
    return bytes(payload)


def observation_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in OBSERVATION_RESULTS:
        if name == OBSERVATION_RAW:
            payloads[name] = b'{"raw_fixture":"observation"}'
        elif name.endswith(".json"):
            payloads[name] = b"{}"
        elif name.endswith(".png"):
            payloads[name] = png_bytes()
        else:
            payloads[name] = (
                b'<svg xmlns="http://www.w3.org/2000/svg" '
                b'width="1" height="1"><path d="M0 0"/></svg>'
            )
    return payloads


def production_shaped_observation_raw_bytes() -> bytes:
    layers = list(range(32))
    panel_items = []
    for item_index in range(12):
        top_tokens = {
            method: [
                {
                    "layer": layer,
                    "tokens": [
                        {
                            "token_id": token_index,
                            "token": (
                                f"{method}-item-{item_index:02d}-"
                                f"layer-{layer:02d}-token-{token_index:02d}"
                            ),
                            "rank": token_index + 1,
                        }
                        for token_index in range(40)
                    ],
                }
                for layer in layers
            ]
            for method in ("jlens", "logit_lens")
        }
        intermediate = {
            "text": f"intermediate-{item_index}",
            "token_id": item_index,
            "single_token": True,
        }
        for method in ("jlens", "logit_lens"):
            intermediate[method] = {
                "ranks": [
                    {"layer": layer, "rank": item_index + 1}
                    for layer in layers
                ],
                "best_rank": item_index + 1,
                "best_layer": 12,
            }
        panel_items.append(
            {
                "name": f"panel-item-{item_index:02d}",
                "prompt": f"production-shaped prompt {item_index}",
                "target": f"target-{item_index}",
                "intermediates": [intermediate],
                "top_tokens": top_tokens,
            }
        )

    direct_items = list(panel_items)
    for item_index in range(12, 93):
        intermediate = {
            "text": f"intermediate-{item_index}",
            "token_id": item_index,
            "single_token": True,
        }
        for method in ("jlens", "logit_lens"):
            intermediate[method] = {
                "ranks": [
                    {"layer": layer, "rank": item_index + 1}
                    for layer in layers
                ],
                "best_rank": item_index + 1,
                "best_layer": 12,
            }
        direct_items.append(
            {
                "name": f"ordinary-item-{item_index:02d}",
                "prompt": f"production-shaped prompt {item_index}",
                "target": f"target-{item_index}",
                "intermediates": [intermediate],
            }
        )

    document = {
        "provenance": {"source": "production-shaped-transport-fixture"},
        "tokenization_audit": [
            {
                "name": item["name"],
                "intermediates": [
                    {
                        "text": item["intermediates"][0]["text"],
                        "token_id": item["intermediates"][0]["token_id"],
                        "token_ids": [item["intermediates"][0]["token_id"]],
                        "variant": "leading_space",
                        "single_token": True,
                    }
                ],
            }
            for item in direct_items
        ],
        "direct": {"items": direct_items},
        "panel": {"items": panel_items},
        "layers": {"band": list(range(12, 29)), "all_layers": layers},
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    assert 2_400_000 < len(payload) < 5 * 1_048_576
    return payload


def matplotlib_svg_bytes() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buffer = io.StringIO()
    figure, axes = plt.subplots()
    axes.imshow([[0.0, 1.0], [1.0, 0.0]])
    axes.set_title("safe observation")
    try:
        figure.savefig(buffer, format="svg", metadata={"Date": None})
    finally:
        plt.close(figure)
    text = re.sub(
        r"<!DOCTYPE.*?>\s*",
        "",
        buffer.getvalue(),
        count=1,
        flags=re.DOTALL,
    )
    return text.encode("utf-8")


def write_tar_payloads(
    archive: Path,
    payloads: dict[str, bytes] | list[tuple[str, bytes]],
) -> None:
    entries = payloads.items() if isinstance(payloads, dict) else payloads
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def write_extension_metadata_archive(
    archive: Path,
    extension: str,
    metadata_bytes: int,
) -> None:
    if extension == "pax":
        with tarfile.open(
            archive,
            "w:gz",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": "A" * metadata_bytes},
        ) as handle:
            for name in sorted(CAUSAL_RESULTS):
                info = tarfile.TarInfo(name)
                info.size = 2
                handle.addfile(info, io.BytesIO(b"{}"))
        return
    if extension == "gnu-longname":
        with tarfile.open(
            archive,
            "w:gz",
            format=tarfile.GNU_FORMAT,
        ) as handle:
            info = tarfile.TarInfo("A" * metadata_bytes)
            info.size = 0
            handle.addfile(info, io.BytesIO())
        return
    raise AssertionError(f"unknown extension fixture: {extension}")


def test_result_profiles_are_exact_and_independent() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_result_profiles")
    pack = load_module(PACK_PATH, "pack_jspace_result_profiles")

    assert runner.RESULT_SETS == {
        "causal": frozenset(CAUSAL_RESULTS),
        "observation": frozenset(OBSERVATION_RESULTS),
    }
    assert pack.RESULT_SETS == runner.RESULT_SETS
    assert runner.result_allowlist("causal") == CAUSAL_RESULTS
    assert runner.result_allowlist("observation") == OBSERVATION_RESULTS
    assert pack.result_size_limit(OBSERVATION_RAW) == 5 * 1_048_576
    assert runner.result_size_limit(OBSERVATION_RAW) == 5 * 1_048_576
    assert runner.decompressed_tar_limit("observation") == 44_044_800
    with pytest.raises(ValueError, match="unknown result set"):
        runner.result_allowlist("unknown")
    with pytest.raises(ValueError, match="unknown result set"):
        pack.result_allowlist("unknown")


def test_fetch_cli_selects_observation_profile() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_fetch_profile")

    args = runner.parse_args(["fetch", "--result-set", "observation"])

    assert args.result_set == "observation"


def test_pack_observation_maps_pipeline_figures_to_canonical_names(
    tmp_path: Path,
) -> None:
    pack = load_module(PACK_PATH, "pack_jspace_observation")
    root = tmp_path / "jspace_llama"
    results = root / "results"
    artifacts = root / "artifacts"
    figures = results / "figures"
    figures.mkdir(parents=True)
    artifacts.mkdir()
    (artifacts / "jspace_observation_raw.json").write_text(
        '{"raw_fixture":"observation"}',
        encoding="utf-8",
    )
    (results / "jspace_observation_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    (results / "jspace_observation.json").write_text("{}", encoding="utf-8")
    for stem in ("pass_at_k", "rank_heatmap", "panel"):
        (figures / f"observation_{stem}.png").write_bytes(png_bytes())
        (figures / f"observation_{stem}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"/>',
            encoding="utf-8",
        )
    archive = tmp_path / "results.tar.gz"

    pack.pack_results(root, archive, "observation")

    with tarfile.open(archive, "r:gz") as handle:
        assert {member.name for member in handle.getmembers()} == (
            OBSERVATION_RESULTS
        )


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        (
            "results/jspace_observation.json",
            b'{"duplicate":1,"duplicate":2}',
            "valid JSON",
        ),
        (
            "results/jspace_observation.json",
            b'{"value":NaN}',
            "valid JSON",
        ),
        (
            "results/jspace_observation.json",
            b'{"value":1e9999}',
            "valid JSON",
        ),
        (
            "results/jspace_observation_panel.png",
            b"not png",
            "PNG",
        ),
        (
            "results/jspace_observation_panel.png",
            png_bytes(0, 1),
            "dimensions",
        ),
        (
            "results/jspace_observation_panel.png",
            png_bytes(10_001, 1),
            "dimensions",
        ),
        (
            "results/jspace_observation_panel.png",
            png_with_invalid_compression_method(),
            "IHDR",
        ),
        (
            "results/jspace_observation_panel.svg",
            b'<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg"/>',
            "DOCTYPE",
        ),
        (
            "results/jspace_observation_panel.svg",
            b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
            "active",
        ),
        (
            "results/jspace_observation_panel.svg",
            (
                b'<svg xmlns="http://www.w3.org/2000/svg">'
                b'<image href="https://example.test/a.png"/></svg>'
            ),
            "external",
        ),
        (
            "results/jspace_observation_panel.svg",
            (
                b'<svg xmlns="http://www.w3.org/2000/svg">'
                b"<style>.a{fill:url(#local)}"
                b".b{fill:url(https://example.test/a.svg)}</style></svg>"
            ),
            "external",
        ),
    ],
)
def test_observation_extract_rejects_hostile_artifact_without_replacement(
    tmp_path: Path,
    name: str,
    payload: bytes,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"run_jspace_hostile_{Path(name).suffix}_{len(payload)}",
    )
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    sentinels = {}
    for result_name in OBSERVATION_RESULTS:
        destination = project / result_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old:{result_name}".encode())
        sentinels[result_name] = destination.read_bytes()
    payloads = observation_payloads()
    payloads[name] = payload
    archive = tmp_path / "results.tar.gz"
    write_tar_payloads(archive, payloads)

    with pytest.raises(ValueError, match=message):
        runner.extract_results_archive(archive, project, "observation")

    assert {
        result_name: (project / result_name).read_bytes()
        for result_name in OBSERVATION_RESULTS
    } == sentinels


def test_observation_extract_rejects_bad_ihdr_crc(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_bad_png_crc")
    payloads = observation_payloads()
    name = "results/jspace_observation_panel.png"
    payloads[name] = payloads[name][:-1] + bytes([payloads[name][-1] ^ 0xFF])
    archive = tmp_path / "results.tar.gz"
    write_tar_payloads(archive, payloads)

    with pytest.raises(ValueError, match="IHDR"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")


@pytest.mark.parametrize("target", ["pack", "fetch"])
@pytest.mark.parametrize(
    "payload",
    [
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b"<style>.x{fill:u&#114;l(https://example.test/a.svg)}</style>"
            b"</svg>"
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b"<style>.x{fill:u\\72l(https://example.test/a.svg)}</style>"
            b"</svg>"
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b"<style>@\\69mport 'https://example.test/a.css';</style>"
            b"</svg>"
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b'<path style="fill:u&#114;l(https://example.test/a.svg)"/>'
            b"</svg>"
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b'<set attributeName="href" to="https://example.test/a.svg"/>'
            b"</svg>"
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg">'
            b'<animate attributeName="href" values="#safe;https://example.test/a"/>'
            b"</svg>"
        ),
    ],
)
def test_svg_validators_reject_encoded_external_and_smil_content(
    target: str,
    payload: bytes,
) -> None:
    if target == "pack":
        module = load_module(PACK_PATH, f"pack_svg_attack_{len(payload)}")
        validator = module._validate_svg
    else:
        module = load_module(RUNNER_PATH, f"fetch_svg_attack_{len(payload)}")
        validator = module._validate_svg_bytes

    with pytest.raises(ValueError, match="SVG|external|active|element"):
        validator("results/observation.svg", payload)


@pytest.mark.parametrize("target", ["pack", "fetch"])
def test_svg_validators_accept_generated_matplotlib_svg(target: str) -> None:
    payload = matplotlib_svg_bytes()
    if target == "pack":
        module = load_module(PACK_PATH, "pack_svg_matplotlib")
        validator = module._validate_svg
    else:
        module = load_module(RUNNER_PATH, "fetch_svg_matplotlib")
        validator = module._validate_svg_bytes

    validator("results/observation.svg", payload)


@pytest.mark.parametrize(
    "entries",
    [
        [
            *observation_payloads().items(),
            ("../jspace_observation.json", b"{}"),
        ],
        [
            *observation_payloads().items(),
            ("results//jspace_observation.json", b"{}"),
        ],
        [
            *observation_payloads().items(),
            ("results/jspace_observation.json", b"{}"),
        ],
    ],
)
def test_observation_extract_rejects_traversal_and_duplicates(
    tmp_path: Path,
    entries: list[tuple[str, bytes]],
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_archive_names")
    archive = tmp_path / "results.tar.gz"
    write_tar_payloads(archive, entries)

    with pytest.raises(ValueError, match="duplicate|unexpected|unsafe"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")


def test_observation_extract_rejects_noncanonical_member_path(
    tmp_path: Path,
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_noncanonical_name")
    payloads = observation_payloads()
    payload = payloads.pop("results/jspace_observation.json")
    entries = [
        *payloads.items(),
        ("results/./jspace_observation.json", payload),
    ]
    archive = tmp_path / "results.tar.gz"
    write_tar_payloads(archive, entries)

    with pytest.raises(ValueError, match="unsafe"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")


def test_observation_extract_bounds_high_member_count_before_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_bounded_members")
    archive = tmp_path / "many-results.tar.gz"
    entries = [
        *observation_payloads().items(),
        *[
            (f"results/unexpected-{index}.json", b"{}")
            for index in range(2_000)
        ],
    ]
    write_tar_payloads(archive, entries)
    next_calls = 0
    real_next = runner.tarfile.TarFile.next

    def counted_next(handle):
        nonlocal next_calls
        next_calls += 1
        return real_next(handle)

    monkeypatch.setattr(runner.tarfile.TarFile, "next", counted_next)
    monkeypatch.setattr(
        runner.tarfile.TarFile,
        "getmembers",
        lambda handle: pytest.fail("getmembers materialized the full archive"),
    )

    with pytest.raises(
        ValueError,
        match="too many result archive members|unexpected result archive member",
    ):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")

    # TarFile.open() reads the first header, then iteration reads at most one
    # header beyond the selected profile before the validator rejects.
    assert next_calls <= len(OBSERVATION_RESULTS) + 2
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("extension", ["pax", "gnu-longname"])
def test_result_extract_bounds_extension_metadata_before_tarfile_parse(
    tmp_path: Path,
    extension: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"run_jspace_bounded_{extension.replace('-', '_')}",
    )
    archive = tmp_path / f"{extension}.tar.gz"
    write_extension_metadata_archive(
        archive,
        extension,
        metadata_bytes=8 * 1_048_576,
    )

    assert archive.stat().st_size < 100_000
    assert archive.stat().st_size < runner.MAX_ARCHIVE_BYTES
    with pytest.raises(ValueError, match="decompressed TAR exceeds"):
        runner.extract_results_archive(archive, tmp_path / "project", "causal")

    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize(
    ("result_set", "payloads"),
    [
        (
            "causal",
            {
                "results/execution_manifest.json": b"{}",
                "results/final_evaluation.json": b"[]",
            },
        ),
        ("observation", observation_payloads()),
    ],
)
def test_result_extract_accepts_valid_bounded_profile_archive(
    tmp_path: Path,
    result_set: str,
    payloads: dict[str, bytes],
) -> None:
    runner = load_module(RUNNER_PATH, f"run_jspace_valid_{result_set}")
    archive = tmp_path / f"{result_set}.tar.gz"
    project = tmp_path / "project"
    write_tar_payloads(archive, payloads)

    runner.extract_results_archive(archive, project, result_set)

    assert {
        name: (project / name).read_bytes() for name in payloads
    } == payloads


def test_observation_raw_sidecar_round_trips_with_exact_hash(
    tmp_path: Path,
) -> None:
    pack = load_module(PACK_PATH, "pack_jspace_raw_round_trip")
    runner = load_module(RUNNER_PATH, "run_jspace_raw_round_trip")
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    payloads = observation_payloads()
    raw_payload = (
        b'{"provenance":{"source":"fixture"},'
        b'"tokenization_audit":[],"direct":{"items":[]},"panel":[]}'
    )
    payloads[OBSERVATION_RAW] = raw_payload
    for name, payload in payloads.items():
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    archive = tmp_path / "observation-results.tar.gz"

    pack.pack_results(remote, archive, "observation")
    runner.extract_results_archive(archive, local, "observation")

    with tarfile.open(archive, "r:gz") as handle:
        assert {member.name for member in handle} == OBSERVATION_RESULTS
    fetched_raw = (local / OBSERVATION_RAW).read_bytes()
    assert fetched_raw == raw_payload
    assert hashlib.sha256(fetched_raw).digest() == hashlib.sha256(
        raw_payload
    ).digest()


def test_production_shaped_observation_raw_over_2_4_mib_round_trips(
    tmp_path: Path,
) -> None:
    pack = load_module(PACK_PATH, "pack_jspace_production_raw")
    runner = load_module(RUNNER_PATH, "run_jspace_production_raw")
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    payloads = observation_payloads()
    raw_payload = production_shaped_observation_raw_bytes()
    payloads[OBSERVATION_RAW] = raw_payload
    for name, payload in payloads.items():
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    archive = tmp_path / "production-observation-results.tar.gz"

    pack.pack_results(remote, archive, "observation")
    runner.extract_results_archive(archive, local, "observation")

    fetched_raw = (local / OBSERVATION_RAW).read_bytes()
    assert fetched_raw == raw_payload
    pack._validate_artifact(OBSERVATION_RAW, fetched_raw)
    runner._validate_result_bytes(OBSERVATION_RAW, fetched_raw)


@pytest.mark.parametrize("target", ["pack", "fetch"])
def test_observation_raw_over_5_mib_rejects_on_both_ends(
    tmp_path: Path,
    target: str,
) -> None:
    oversized_raw = (
        b'{"provenance":{"source":"oversized-fixture"},"padding":"'
        + b"A" * (5 * 1_048_576)
        + b'"}'
    )
    payloads = observation_payloads()
    payloads[OBSERVATION_RAW] = oversized_raw

    if target == "pack":
        pack = load_module(PACK_PATH, "pack_jspace_oversized_raw")
        remote = tmp_path / "remote"
        for name, payload in payloads.items():
            path = remote / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        with pytest.raises(ValueError, match="too large"):
            pack.pack_results(
                remote,
                tmp_path / "oversized-pack.tar.gz",
                "observation",
            )
        assert not (tmp_path / "oversized-pack.tar.gz").exists()
    else:
        runner = load_module(RUNNER_PATH, "run_jspace_oversized_raw")
        archive = tmp_path / "oversized-fetch.tar.gz"
        project = tmp_path / "project"
        write_tar_payloads(archive, payloads)

        with pytest.raises(ValueError, match="too large"):
            runner.extract_results_archive(archive, project, "observation")
        assert not project.exists()


def test_observation_extract_reads_no_payload_before_complete_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_validate_before_read")
    payloads = observation_payloads()
    payloads.pop("results/jspace_observation_panel.svg")
    archive = tmp_path / "missing-results.tar.gz"
    write_tar_payloads(archive, payloads)
    extracted: list[str] = []
    real_extractfile = runner.tarfile.TarFile.extractfile

    def track_extractfile(handle, member):
        extracted.append(member.name)
        return real_extractfile(handle, member)

    monkeypatch.setattr(
        runner.tarfile.TarFile,
        "extractfile",
        track_extractfile,
    )

    with pytest.raises(FileNotFoundError, match="missing result archive member"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")

    assert extracted == []
    assert not (tmp_path / "project").exists()


def test_observation_extract_rejects_missing_and_oversized_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_limits")
    payloads = observation_payloads()
    missing = payloads.pop("results/jspace_observation_panel.svg")
    archive = tmp_path / "missing.tar.gz"
    write_tar_payloads(archive, payloads)
    with pytest.raises(FileNotFoundError, match="missing result archive member"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")

    payloads["results/jspace_observation_panel.svg"] = missing
    archive = tmp_path / "large.tar.gz"
    write_tar_payloads(archive, payloads)
    monkeypatch.setattr(runner, "MAX_FIGURE_BYTES", 1)
    with pytest.raises(ValueError, match="too large"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")

    monkeypatch.setattr(runner, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="archive is too large"):
        runner.extract_results_archive(archive, tmp_path / "project", "observation")


def test_observation_extract_rolls_back_replacement_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_transaction")
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    sentinels = {}
    for name in OBSERVATION_RESULTS:
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"old:{name}".encode())
        sentinels[name] = destination.read_bytes()
    archive = tmp_path / "results.tar.gz"
    write_tar_payloads(archive, observation_payloads())
    real_replace = runner.os.replace
    replacements = 0

    def fail_second_publication(source: Path, destination: Path) -> None:
        nonlocal replacements
        if Path(destination).parent == results:
            replacements += 1
            if replacements == 2:
                raise OSError("injected replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_second_publication)

    with pytest.raises(OSError, match="injected replacement failure"):
        runner.extract_results_archive(archive, project, "observation")

    assert {
        name: (project / name).read_bytes() for name in OBSERVATION_RESULTS
    } == sentinels


def test_documentation_lists_exact_jspace_workflow_and_gated_auth() -> None:
    colab_docs = (ROOT / "tools" / "colab" / "README.md").read_text(
        encoding="utf-8"
    )
    project_docs = (PROJECT / "README.md").read_text(encoding="utf-8")
    commands = [
        "python tools/colab/run_jspace.py probe",
        "python tools/colab/run_jspace.py start --gpu L4",
        "python tools/colab/run_jspace.py sync",
        "python tools/colab/run_jspace.py bootstrap",
        "python tools/colab/run_jspace.py auth-hf",
        (
            "python tools/colab/run_jspace.py execute "
            "--notebook projects/jspace_llama/notebooks/"
            "01_readout_and_causal_swap.ipynb --timeout 7200"
        ),
        "python tools/colab/run_jspace.py fetch",
        "python tools/colab/run_jspace.py stop",
    ]

    for documentation in (colab_docs, project_docs):
        for command in commands:
            assert command in documentation
        assert "meta-llama/Llama-3.1-8B-Instruct" in documentation
        assert "uv run hf auth login" in documentation
