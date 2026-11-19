"""Analysis registry, programmatic executor, and read-only preflight doctor."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import shutil
from typing import Any, Protocol

from taskce.data import DataRoot
from taskce.profiles import Profile, load_profile
from taskce.runtime import (
    AnalysisName,
    PINNED_2A_HASHES,
    PINNED_BASELINE_COMMIT,
    PINNED_FORMULA_EVAL_SHA256,
    PINNED_SNLI_JOINT_MANIFEST_SHA256,
    ResourceFlags,
    RunId,
    RunResult,
    RunWorkspace,
    sha256_file,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    resources: ResourceFlags = ResourceFlags.NONE
    default: bool = True


class AnalysisRunner(Protocol):
    """Uniform contract implemented by each `<analysis>.runner` module."""

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None: ...
    def options_from_args(self, args: argparse.Namespace) -> Any: ...
    def run(
        self,
        workspace: RunWorkspace,
        profile: Profile,
        options: Any,
        stages: tuple[str, ...],
    ) -> Mapping[str, Path]: ...


@dataclass(frozen=True)
class AnalysisSpec:
    name: AnalysisName
    summary: str
    runner_module: str
    stages: tuple[StageSpec, ...]
    default_profile: str = "snli"

    @property
    def cli_name(self) -> str:
        return self.name.replace("_", "-")

    @property
    def default_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.default)

    def stage_map(self) -> dict[str, StageSpec]:
        return {stage.name: stage for stage in self.stages}

    def resolve_stages(self, requested: Sequence[str] | None) -> tuple[str, ...]:
        """Validate requested stage names against this spec."""
        if not requested:
            return self.default_stages
        valid = self.stage_map()
        resolved: list[str] = []
        for name in requested:
            if name not in valid:
                choices = ", ".join(valid)
                raise ValueError(
                    f"Unknown stage {name!r} for analysis {self.name!r}; "
                    f"available stages: {choices}"
                )
            if name not in resolved:
                resolved.append(name)
        return tuple(resolved)

    def required_resources(self, stages: Sequence[str]) -> ResourceFlags:
        """Union ResourceFlags across the selected stages only."""
        by_name = self.stage_map()
        flags = ResourceFlags.NONE
        for name in stages:
            flags |= by_name[name].resources
        return flags


ANALYSES: Mapping[AnalysisName, AnalysisSpec] = {
    "capture": AnalysisSpec(
        name="capture",
        summary="Capture final-token model activations via forward hooks",
        runner_module="capture.runner",
        stages=(
            StageSpec(
                "base",
                "Capture base model activations",
                ResourceFlags.HF_MODEL | ResourceFlags.HF_DATASET,
            ),
            StageSpec(
                "finetuned",
                "Capture task LoRA activations",
                ResourceFlags.HF_MODEL | ResourceFlags.HF_DATASET,
            ),
        ),
    ),
    "activation_analysis": AnalysisSpec(
        name="activation_analysis",
        summary="Analyze threshold coverage, neuron fate, and task separation over saved captures",
        runner_module="activation_analysis.runner",
        stages=(
            StageSpec(
                "threshold_coverage",
                "Quantile threshold evidence coverage",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "neuron_fate",
                "Base vs fine-tuned affine/CCC neuron fate",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "task_separation",
                "2D PCA projection colored by task labels",
                ResourceFlags.NONE,
            ),
        ),
    ),
    "formula_search": AnalysisSpec(
        name="formula_search",
        summary="Prepare concept features and search Boolean compositional explanations",
        runner_module="formula_search.runner",
        stages=(
            StageSpec(
                "prepare",
                "Build aligned features.npz and <layer>.npz",
                ResourceFlags.HF_DATASET,
                default=False,
            ),
            StageSpec(
                "search",
                "Chunked resumable formula search (pinned 2.a or new_search)",
                ResourceFlags.CUDA,
                default=True,
            ),
            StageSpec(
                "attach_weights",
                "Attach LoRA lm_head class weights to formula CSV",
                ResourceFlags.HF_MODEL,
                default=False,
            ),
        ),
    ),
    "formula_comparison": AnalysisSpec(
        name="formula_comparison",
        summary="Diff base vs fine-tuned formula CSVs and build offline diff/AST viewers",
        runner_module="formula_comparison.runner",
        stages=(
            StageSpec(
                "diff",
                "Compute structural formula diffs and score heatmap",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "diff_viewer",
                "Render offline HTML formula-diff viewer",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "ast_viewer",
                "Render offline HTML formula-AST viewer",
                ResourceFlags.NONE,
                default=False,
            ),
            StageSpec(
                "verify_ast_viewer",
                "Verify embedded AST viewer matches source CSVs",
                ResourceFlags.NONE,
                default=False,
            ),
        ),
    ),
    "joint_search": AnalysisSpec(
        name="joint_search",
        summary="Run manifest-pinned pairwise joint-firing length sweep and baseline verification",
        runner_module="joint_search.runner",
        stages=(
            StageSpec(
                "validate",
                "Verify manifest, input SHA-256s, 2.a pin, and symbolic baselines",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "sweep",
                "Run CUDA length-budget sweep for all manifest pairs",
                ResourceFlags.CUDA,
            ),
        ),
    ),
    "cross_layer_recurrence": AnalysisSpec(
        name="cross_layer_recurrence",
        summary="Index recurrent compound subformulas, 3-stage chains, and earlier-layer matches",
        runner_module="cross_layer_recurrence.runner",
        stages=(
            StageSpec(
                "full_sweep",
                "Run pinned 2.a search across all configured layers",
                ResourceFlags.CUDA,
                default=False,
            ),
            StageSpec(
                "index",
                "Index exact compound subformulas across full/*.csv",
                ResourceFlags.NONE,
                default=True,
            ),
            StageSpec(
                "chains",
                "Extract strictly increasing 3-stage nested compound chains",
                ResourceFlags.NONE,
                default=True,
            ),
            StageSpec(
                "match",
                "Match top recurrent compounds against earlier single neurons",
                ResourceFlags.NONE,
                default=True,
            ),
        ),
    ),
    "graph_analysis": AnalysisSpec(
        name="graph_analysis",
        summary="Construct neuron correlation/cosine graphs, communities, plots, and reports",
        runner_module="graph_analysis.runner",
        stages=(
            StageSpec(
                "build_and_report",
                "Build Pearson/cosine graphs, communities, plots, reports",
                ResourceFlags.NONE,
            ),
        ),
    ),
    "ablation": AnalysisSpec(
        name="ablation",
        summary="Rank explained neurons and run cumulative logit-hook ablation inference",
        runner_module="ablation.runner",
        stages=(
            StageSpec(
                "select",
                "Partition neurons into good/bad/IoU groups and write pre-run report",
                ResourceFlags.NONE,
            ),
            StageSpec(
                "infer",
                "Run baseline and cumulative logit-hook ablation inference + plots",
                ResourceFlags.HF_MODEL | ResourceFlags.HF_DATASET,
            ),
        ),
    ),
}


def normalize_analysis_name(name: str) -> AnalysisName:
    """Normalize hyphenated or underscored analysis names to canonical AnalysisName."""
    normalized = name.strip().replace("-", "_")
    if normalized not in ANALYSES:
        available = ", ".join(spec.cli_name for spec in ANALYSES.values())
        raise KeyError(f"Unknown analysis {name!r}; available analyses: {available}")
    return normalized  # type: ignore[return-value]


def execute(
    analysis: AnalysisName | str,
    *,
    profile: Profile | str | Path = "snli",
    data_root: DataRoot | None = None,
    options: Any = None,
    stages: Sequence[str] | None = None,
    run_id: RunId | None = None,
    output_dir: Path | None = None,
    resume: bool = False,
    check_only: bool = False,
) -> RunResult:
    """Single programmatic execution boundary for all eight analyses."""
    canonical_name = normalize_analysis_name(str(analysis))
    spec = ANALYSES[canonical_name]

    provisional_root = data_root or DataRoot.from_env()
    loaded_profile = (
        profile
        if isinstance(profile, Profile)
        else load_profile(profile, data_root=provisional_root)
    )

    if data_root is not None:
        root = data_root
    else:
        baseline_commit = (
            loaded_profile.artifacts.baseline_commit or PINNED_BASELINE_COMMIT
        )
        root = DataRoot.from_env(baseline_commit=baseline_commit)

    selected_stages = spec.resolve_stages(stages)
    if check_only and not stages:
        by_name = spec.stage_map()
        selected_stages = tuple(
            stage_name
            for stage_name in selected_stages
            if by_name[stage_name].resources == ResourceFlags.NONE
        )

    # Only import taskce.hf and validate HF_HOME when selected stages actually need HF.
    required = spec.required_resources(selected_stages)
    if required & (ResourceFlags.HF_MODEL | ResourceFlags.HF_DATASET):
        from taskce.hf import configure_hf_cache

        configure_hf_cache(root)

    workspace = RunWorkspace.open(
        analysis=canonical_name,
        profile=loaded_profile,
        data_root=root,
        requested_stages=selected_stages,
        run_id=run_id,
        output_dir=output_dir,
        resume=resume,
        check_only=check_only,
    )

    try:
        runner = importlib.import_module(spec.runner_module)
        produced = runner.run(workspace, loaded_profile, options, selected_stages)
        if produced:
            for key, path in produced.items():
                workspace.record_artifact(key, path)
    except Exception as exc:
        workspace.finalize("failed", error=f"{type(exc).__name__}: {exc}")
        raise

    return workspace.finalize("validated" if check_only else "complete")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def doctor(
    *,
    profile: Profile | str | Path = "snli",
    data_root: DataRoot | None = None,
) -> tuple[DoctorCheck, ...]:
    """Read-only environment and cryptographic preflight with zero model downloads."""
    checks: list[DoctorCheck] = []
    provisional_root = data_root or DataRoot.from_env()
    repo_root = provisional_root.repo_root or Path(__file__).resolve().parents[1]

    loaded_profile: Profile | None = None
    try:
        loaded_profile = (
            profile
            if isinstance(profile, Profile)
            else load_profile(profile, data_root=provisional_root)
        )
        checks.append(
            DoctorCheck(
                "profile",
                True,
                f"{loaded_profile.name} ({loaded_profile.source_path})",
            )
        )
    except Exception as exc:
        checks.append(DoctorCheck("profile", False, f"{type(exc).__name__}: {exc}"))

    if data_root is not None:
        root = data_root
    elif loaded_profile is not None:
        root = DataRoot.from_env(
            baseline_commit=loaded_profile.artifacts.baseline_commit
            or PINNED_BASELINE_COMMIT
        )
    else:
        root = provisional_root

    checks.append(
        DoctorCheck(
            "data_dir",
            root.data_dir.is_dir(),
            str(root.data_dir),
        )
    )
    checks.append(
        DoctorCheck(
            "baseline_dir",
            root.baseline_dir is not None and root.baseline_dir.is_dir(),
            str(root.baseline_dir) if root.baseline_dir is not None else "not configured",
        )
    )

    if loaded_profile is not None:
        try:
            bundle = root.resolve_prepared_inputs(loaded_profile)
            checks.append(
                DoctorCheck(
                    "prepared_inputs",
                    True,
                    f"{bundle.directory} ({len(bundle.layer_paths)} layer files)",
                )
            )
        except FileNotFoundError as exc:
            checks.append(DoctorCheck("prepared_inputs", False, str(exc)))

    pkg_dir = repo_root / "formula_search" / "vectorized_search"
    mismatches: list[str] = []
    for filename, expected_sha in PINNED_2A_HASHES.items():
        target = pkg_dir / filename
        if not target.is_file():
            mismatches.append(f"missing {target}")
            continue
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            mismatches.append(f"{filename}: {actual_sha} != {expected_sha}")
    checks.append(
        DoctorCheck(
            "pinned_vectorized_search",
            not mismatches,
            "all 3 pinned files match SHA-256" if not mismatches else "; ".join(mismatches),
        )
    )

    eval_path = repo_root / "formula_search" / "formula_eval.py"
    if eval_path.is_file():
        actual_eval = sha256_file(eval_path)
        eval_ok = actual_eval == PINNED_FORMULA_EVAL_SHA256
        checks.append(
            DoctorCheck(
                "pinned_formula_eval",
                eval_ok,
                actual_eval if eval_ok else f"{actual_eval} != {PINNED_FORMULA_EVAL_SHA256}",
            )
        )
    else:
        checks.append(DoctorCheck("pinned_formula_eval", False, f"missing {eval_path}"))

    manifest_path = repo_root / "joint_search" / "manifests" / "snli_10k_v1.json"
    if manifest_path.is_file():
        actual_manifest = sha256_file(manifest_path)
        manifest_ok = actual_manifest == PINNED_SNLI_JOINT_MANIFEST_SHA256
        checks.append(
            DoctorCheck(
                "pinned_joint_manifest",
                manifest_ok,
                (
                    actual_manifest
                    if manifest_ok
                    else f"{actual_manifest} != {PINNED_SNLI_JOINT_MANIFEST_SHA256}"
                ),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "pinned_joint_manifest",
                False,
                f"not yet present at {manifest_path}",
            )
        )

    torch_spec = importlib.util.find_spec("torch")
    nvidia_smi = shutil.which("nvidia-smi")
    checks.append(
        DoctorCheck(
            "cuda_tooling",
            torch_spec is not None and nvidia_smi is not None,
            f"torch_installed={torch_spec is not None}, nvidia_smi={nvidia_smi or 'absent'}",
        )
    )

    return tuple(checks)
