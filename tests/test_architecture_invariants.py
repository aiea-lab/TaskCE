"""Architecture, boundary, and cryptographic pin guardrails for flattened TaskCE."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from formula_search.verify_2a import load_verified_2a
from taskce.data import DataRoot
from taskce.executor import ANALYSES
from taskce.profiles import load_profile
from taskce.runtime import (
    ALLOWED_PACKAGE_IMPORTS,
    PINNED_2A_HASHES,
    PINNED_2A_MODULE_KEYS,
    PINNED_FORMULA_EVAL_SHA256,
    PINNED_SNLI_JOINT_MANIFEST_SHA256,
    RunWorkspace,
    sha256_file,
)


ANALYSIS_PACKAGES = (
    "capture",
    "activation_analysis",
    "formula_search",
    "formula_comparison",
    "joint_search",
    "cross_layer_recurrence",
    "graph_analysis",
    "ablation",
)
ALL_PACKAGES = ("taskce", *ANALYSIS_PACKAGES)
DELETED_TREES = ("experimental", "theoretical", "experiments", "evaluation")
HEAVY_MODULES = ("torch", "transformers", "peft", "datasets", "huggingface_hub", "z3")

MODEL_ID = "creator/model"
LAYER = "model.layers.8.feed_forward"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def save_placeholder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"activation")
    return path


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _imported_roots(tree: ast.AST, *, current_package: str) -> list[str]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.append(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                roots.append(current_package)
            elif node.module:
                roots.append(node.module.split(".", 1)[0])
    return roots


@pytest.mark.parametrize("directory", DELETED_TREES)
def test_legacy_and_evaluation_directories_do_not_exist(directory: str) -> None:
    assert not (repo_root() / directory).exists()


def test_registry_covers_exact_eight_top_level_analyses() -> None:
    assert tuple(ANALYSES) == ANALYSIS_PACKAGES
    for name in ANALYSIS_PACKAGES:
        pkg = repo_root() / name
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "runner.py").is_file()


@pytest.mark.parametrize("package", ALL_PACKAGES)
def test_package_imports_obey_acyclic_dag(package: str) -> None:
    allowed_internal = {package, *ALLOWED_PACKAGE_IMPORTS[package]}
    internal_packages = set(ALL_PACKAGES) | set(DELETED_TREES)
    violations: list[tuple[str, str]] = []

    for source_path in (repo_root() / package).rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        for root_mod in _imported_roots(tree, current_package=package):
            if root_mod in internal_packages and root_mod not in allowed_internal:
                violations.append((str(source_path.relative_to(repo_root())), root_mod))

    assert violations == []


@pytest.mark.parametrize("package", ALL_PACKAGES)
def test_no_package_mutates_sys_path(package: str) -> None:
    violations: list[str] = []
    for source_path in (repo_root() / package).rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"insert", "append"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
            ):
                violations.append(str(source_path.relative_to(repo_root())))
    assert violations == []


def test_cli_parser_startup_does_not_import_heavy_ml_or_solver_modules() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root())
    code = (
        "import json, sys; "
        "from taskce.cli import build_parser; "
        "build_parser(); "
        f"heavy = [m for m in {list(HEAVY_MODULES)!r} if m in sys.modules]; "
        "print(json.dumps(heavy))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout.strip()) == []


def test_cryptographic_pins_and_zero_syspath_2a_loader() -> None:
    root = repo_root()
    pkg_dir = root / "formula_search" / "vectorized_search"
    for filename, expected_sha in PINNED_2A_HASHES.items():
        assert sha256_file(pkg_dir / filename) == expected_sha

    assert (
        sha256_file(root / "formula_search" / "formula_eval.py")
        == PINNED_FORMULA_EVAL_SHA256
    )
    assert (
        sha256_file(root / "joint_search" / "manifests" / "snli_10k_v1.json")
        == PINNED_SNLI_JOINT_MANIFEST_SHA256
    )

    before_path = list(sys.path)
    verified = load_verified_2a(require_runtime=False)
    assert list(sys.path) == before_path
    assert set(verified.proof.modules) == set(PINNED_2A_MODULE_KEYS)
    assert verified.evaluator_sha256 == PINNED_FORMULA_EVAL_SHA256


def test_data_root_resolves_env_and_pvc_activation_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pvc_data = tmp_path / "TaskCE" / "data"
    pvc_baseline = tmp_path / "TaskCE" / "baselines" / "57a1e74"
    pvc_data.mkdir(parents=True)
    pvc_baseline.mkdir(parents=True)

    monkeypatch.setenv("TASKCE_DATA_DIR", str(pvc_data))
    monkeypatch.delenv("TASKCE_BASELINE_DIR", raising=False)

    root = DataRoot.from_env()
    assert root.data_dir == pvc_data.resolve()
    assert root.baseline_dir == pvc_baseline.resolve()

    model_directory = pvc_data / "creator" / "model"
    base_path = save_placeholder(model_directory / "base" / LAYER / "snli_activation.pt")
    checkpoint_9 = save_placeholder(
        model_directory / "snli" / "checkpoint-9" / LAYER / "snli_activation.pt"
    )
    checkpoint_10 = save_placeholder(
        model_directory
        / "snli"
        / "checkpoint-10"
        / f"base_model.model.{LAYER}"
        / "snli_activation.pt"
    )
    (model_directory / "base" / f"base_model.model.{LAYER}").mkdir()

    assert root.activation_path(MODEL_ID, "snli", LAYER) == base_path
    assert root.activation_path(MODEL_ID, "snli", LAYER, task_name="snli") == checkpoint_10
    assert (
        root.activation_path(
            MODEL_ID, "snli", LAYER, task_name="snli", checkpoint_name="checkpoint-9"
        )
        == checkpoint_9
    )


def test_data_root_reports_missing_and_ambiguous_layers(tmp_path: Path) -> None:
    root = DataRoot(data_dir=tmp_path)
    base_directory = tmp_path / "creator" / "model" / "base"
    base_directory.mkdir(parents=True)

    with pytest.raises(KeyError, match=re.escape(LAYER)):
        root.activation_path(MODEL_ID, "snli", LAYER)

    save_placeholder(base_directory / f"first.{LAYER}" / "snli_activation.pt")
    save_placeholder(base_directory / f"second.{LAYER}" / "snli_activation.pt")

    with pytest.raises(ValueError, match="multiple saved activation"):
        root.activation_path(MODEL_ID, "snli", LAYER)


def test_profile_lookup_supports_pvc_profiles_and_extends(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "bowman.toml").write_text(
        'name = "bowman"\n'
        'description = "Bowman parity profile"\n'
        'extends = "snli"\n\n'
        "[model]\n"
        'lora_task = "bowman"\n\n'
        "[dataset]\n"
        'name = "bowman"\n'
        'tokenizer_name = "spacy-pos-bowman-features"\n',
        encoding="utf-8",
    )
    data_root = DataRoot(data_dir=tmp_path)
    bowman = load_profile("bowman", data_root=data_root)
    assert bowman.name == "bowman"
    assert bowman.model.lora_task == "bowman"
    assert bowman.model.model_id == "LiquidAI/LFM2.5-1.2B-Thinking"
    assert bowman.search.beam == 40
    assert bowman.dataset.tokenizer_name == "spacy-pos-bowman-features"


def test_run_workspace_isolates_runs_and_rejects_resume_profile_drift(
    tmp_path: Path,
) -> None:
    root = DataRoot(data_dir=tmp_path)
    snli = load_profile("snli")
    claim = load_profile("claim")

    ws1 = RunWorkspace.open(
        analysis="joint_search",
        profile=snli,
        data_root=root,
        requested_stages=("validate",),
    )
    ws2 = RunWorkspace.open(
        analysis="joint_search",
        profile=snli,
        data_root=root,
        requested_stages=("validate",),
    )
    assert ws1.output_dir != ws2.output_dir
    ws1.finalize("validated")

    with pytest.raises(ValueError, match="Cannot resume"):
        RunWorkspace.open(
            analysis="joint_search",
            profile=claim,
            data_root=root,
            requested_stages=("validate",),
            run_id=ws1.run_id,
            resume=True,
        )

    with pytest.raises(ValueError, match="profile_sha256"):
        RunWorkspace.open(
            analysis="joint_search",
            profile=claim,
            data_root=root,
            requested_stages=("validate",),
            output_dir=ws1.output_dir,
            resume=True,
        )


def test_notebooks_import_flattened_packages() -> None:
    for purpose in ("threshold_coverage", "neuron_fate", "task_separation"):
        source = notebook_source(
            repo_root() / "activation_analysis" / "notebooks" / f"{purpose}.ipynb"
        )
        assert f"from activation_analysis.{purpose} import" in source
        assert "from taskce.data import DataRoot" in source

    assert "from graph_analysis.analysis import" in notebook_source(
        repo_root() / "graph_analysis" / "notebook.ipynb"
    )
    assert "from ablation.analysis import" in notebook_source(
        repo_root() / "ablation" / "notebook.ipynb"
    )
