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
from taskce.executor import ANALYSES, doctor, execute
from taskce.profiles import load_profile
from taskce.runtime import (
    ALLOWED_PACKAGE_IMPORTS,
    PINNED_2A_HASHES,
    PINNED_2A_MODULE_KEYS,
    PINNED_FORMULA_EVAL_SHA256,
    PINNED_SNLI_JOINT_MANIFEST_SHA256,
    RunId,
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


def test_check_only_rejects_resource_stages_and_empty_default_stages(
    tmp_path: Path,
) -> None:
    root = DataRoot(data_dir=tmp_path)
    with pytest.raises(ValueError, match="cannot run resource-requiring stage"):
        execute(
            "joint_search",
            profile="snli",
            data_root=root,
            stages=("sweep",),
            check_only=True,
        )
    with pytest.raises(ValueError, match="no read-only default stages"):
        execute(
            "capture",
            profile="snli",
            data_root=root,
            check_only=True,
        )
    assert not (tmp_path / "runs").exists()


def test_resume_check_only_preserves_completed_run_and_rejects_bad_run_id(
    tmp_path: Path,
) -> None:
    root = DataRoot(data_dir=tmp_path)
    snli = load_profile("snli")

    ws = RunWorkspace.open(
        analysis="joint_search",
        profile=snli,
        data_root=root,
        requested_stages=("validate", "sweep"),
    )
    ws.finalize("complete")

    with pytest.raises(ValueError, match="completed run"):
        RunWorkspace.open(
            analysis="joint_search",
            profile=snli,
            data_root=root,
            requested_stages=("validate",),
            run_id=ws.run_id,
            resume=True,
            check_only=True,
        )
    meta = json.loads(ws.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "complete"
    assert root.latest_completed_run("joint_search", "snli") == ws.output_dir

    with pytest.raises(ValueError, match="Invalid run_id"):
        RunWorkspace.open(
            analysis="joint_search",
            profile=snli,
            data_root=root,
            requested_stages=("validate",),
            run_id=RunId("../escape"),
        )


def test_data_dir_override_preserves_profile_baseline_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TASKCE_BASELINE_DIR", raising=False)
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    profiles_dir.mkdir(parents=True)
    custom_baseline = tmp_path / "baselines" / "deadbeef"
    custom_baseline.mkdir(parents=True)
    (profiles_dir / "custom.toml").write_text(
        'name = "custom"\n'
        'extends = "snli"\n\n'
        "[artifacts]\n"
        'baseline_commit = "deadbeef"\n',
        encoding="utf-8",
    )

    checks = {item.name: item for item in doctor(profile="custom", data_dir=data_dir)}
    assert checks["baseline_dir"].ok
    assert Path(checks["baseline_dir"].detail) == custom_baseline


def test_runners_isolate_default_outputs_inside_run_workspace(tmp_path: Path) -> None:
    from ablation.runner import AblationRunnerOptions
    from formula_comparison.runner import FormulaComparisonOptions

    data_dir = tmp_path / "data"
    formula_csv = data_dir / "compositional_explanations" / "snli_beam_results.csv"
    formula_csv.parent.mkdir(parents=True)
    formula_csv.write_text(
        "neuron,formula,iou,weight_entailment,weight_neutral,weight_contradiction\n"
        "0,(premise:cat AND hypothesis:dog),0.75,0.5,-0.1,0.2\n"
        "1,(NOT hypothesis:dog),0.20,-0.4,0.3,0.1\n",
        encoding="utf-8",
    )

    ablation_result = execute(
        "ablation",
        profile="snli",
        data_dir=data_dir,
        options=AblationRunnerOptions(),
        stages=("select",),
    )
    assert ablation_result.status == "complete"
    assert (ablation_result.output_dir / "ablation_analysis_report.txt").is_file()
    assert not (data_dir / "ablation_analysis_report.txt").exists()

    cmp_result = execute(
        "formula_comparison",
        profile="snli",
        data_dir=data_dir,
        options=FormulaComparisonOptions(csv_paths=(formula_csv,)),
        stages=("ast_viewer",),
    )
    assert cmp_result.status == "complete"
    assert (cmp_result.output_dir / "formula_ast_viewer.html").is_file()


def test_task_separation_supports_custom_profile_via_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import torch

    from activation_analysis import task_separation

    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    inputs_dir = data_dir / "inputs"
    profiles_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (profiles_dir / "bowman.toml").write_text(
        'name = "bowman"\n'
        'extends = "snli"\n\n'
        "[model]\n"
        'lora_task = "bowman"\n\n'
        "[dataset]\n"
        'name = "bowman"\n',
        encoding="utf-8",
    )
    np.savez(
        inputs_dir / "features.npz",
        vectors=np.zeros((2, 4), dtype=bool),
        names=np.array(["a", "b"]),
        split=np.array([0, 0, 1, 2]),
        labels=np.array([0, 1, 2, 0]),
    )
    act_path = tmp_path / "bowman_activation.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(4, 3), act_path)

    captured: dict[str, object] = {}

    def fake_analyze(states, labels):
        captured["shape"] = tuple(states.shape)
        captured["labels"] = list(labels)
        return {"labels": list(labels), "shape": tuple(states.shape)}

    monkeypatch.setattr(task_separation, "analyze_task_separation", fake_analyze)

    root = DataRoot(data_dir=data_dir)
    profile = load_profile("bowman", data_root=root)
    result = task_separation.run("bowman", act_path, profile=profile, data_root=root)
    assert result["labels"] == ["entailment", "neutral", "contradiction", "entailment"]
    assert result["shape"] == (4, 3)
