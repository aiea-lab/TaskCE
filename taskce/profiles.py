"""Declarative TOML study profiles with single-level extends and strict validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING, Any

from taskce.runtime import LayerName, SearchBackendName, Sha256Hex, sha256_json

if TYPE_CHECKING:
    from taskce.data import DataRoot


BUILTIN_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

_VALID_BACKENDS = frozenset({"vectorized_search_2a", "new_search", "search"})
_ALLOWED_TOP_KEYS = frozenset(
    {"name", "description", "extends", "model", "dataset", "search", "artifacts"}
)
_ALLOWED_MODEL_KEYS = frozenset(
    {"model_id", "lora_repo", "lora_task", "checkpoint_name", "layers", "layer_aliases"}
)
_ALLOWED_DATASET_KEYS = frozenset(
    {
        "name",
        "hf_id",
        "split",
        "feature_columns",
        "label_column",
        "labels",
        "class_token_ids",
        "prompt",
        "tokenizer_name",
        "tokenizer_uses_pos",
    }
)
_ALLOWED_PROMPT_KEYS = frozenset({"system", "user_template"})
_ALLOWED_SEARCH_KEYS = frozenset(
    {"alpha", "min_acts", "beam", "length", "max_iterations", "seed", "backend"}
)
_ALLOWED_ARTIFACT_KEYS = frozenset(
    {
        "baseline_commit",
        "prepared_inputs_relpaths",
        "joint_manifest_name",
        "joint_parent_csv_relpath",
        "joint_baseline_results_relpath",
        "recurrence_results_relpath",
        "recurrence_layers",
        "recurrence_later_start",
        "recurrence_later_end",
    }
)


@dataclass(frozen=True)
class PromptSpec:
    """Declarative chat prompt template shared by capture and ablation."""

    system: str
    user_template: str

    def render(self, example: Mapping[str, Any]) -> list[dict[str, str]]:
        """Render one example into HF chat-template messages."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(**example)},
        ]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    lora_repo: str | None = None
    lora_task: str | None = None
    checkpoint_name: str | None = None
    layers: tuple[LayerName, ...] = ()
    layer_aliases: Mapping[str, LayerName] = field(default_factory=dict)

    def resolve_module_path(self, layer_or_alias: str) -> LayerName:
        """Accept either a short alias ('layer.8') or full module path."""
        return self.layer_aliases.get(layer_or_alias, LayerName(layer_or_alias))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    split: str
    feature_columns: tuple[str, ...]
    label_column: str
    labels: tuple[str, ...] = ()
    class_token_ids: Mapping[str, int] = field(default_factory=dict)
    prompt: PromptSpec | None = None
    tokenizer_name: str | None = None
    tokenizer_uses_pos: bool = False

    @property
    def weight_column_names(self) -> tuple[str, ...]:
        return tuple(
            f"weight_{label.replace(' ', '_')}" for label in self.class_token_ids
        )


@dataclass(frozen=True)
class SearchDefaults:
    alpha: float = 0.055
    min_acts: int = 500
    beam: int = 40
    length: int = 48
    max_iterations: int = 48
    seed: int = 20261111
    backend: SearchBackendName = "vectorized_search_2a"

    @property
    def quantile(self) -> float:
        return round(1.0 - self.alpha, 12)


@dataclass(frozen=True)
class StudyArtifacts:
    """Relative paths under TASKCE_DATA_DIR, baseline archive, or package resources."""

    baseline_commit: str | None = None
    prepared_inputs_relpaths: tuple[str, ...] = ()
    joint_manifest_name: str | None = None
    joint_parent_csv_relpath: str | None = None
    joint_baseline_results_relpath: str | None = None
    recurrence_results_relpath: str | None = None
    recurrence_layers: tuple[str, ...] = ()
    recurrence_later_start: str | None = None
    recurrence_later_end: str | None = None


@dataclass(frozen=True)
class Profile:
    """Single source of truth for a study (snli, claim, fallacy, custom TOMLs)."""

    name: str
    description: str
    model: ModelSpec
    dataset: DatasetSpec
    search: SearchDefaults
    artifacts: StudyArtifacts
    source_path: Path | None = None
    profile_sha256: Sha256Hex = Sha256Hex("")


def list_builtin_profiles() -> tuple[str, ...]:
    """Discover bundled *.toml profile names under taskce/profiles/."""
    if not BUILTIN_PROFILES_DIR.is_dir():
        return ()
    return tuple(sorted(path.stem for path in BUILTIN_PROFILES_DIR.glob("*.toml")))


def _reject_unknown_keys(
    table: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {context}: {', '.join(unknown)}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_profile_path(
    name_or_path: str | Path,
    *,
    data_root: DataRoot | None = None,
) -> Path:
    candidate = Path(name_or_path)
    as_text = str(name_or_path)

    # 1. Explicit .toml path or existing filesystem path.
    if candidate.is_file() or as_text.endswith(".toml"):
        if not candidate.is_file():
            raise FileNotFoundError(f"Profile file not found: {candidate}")
        return candidate.resolve()

    # 2. Data-root override directory (<TASKCE_DATA_DIR>/profiles/<name>.toml).
    data_dir: Path | None = None
    if data_root is not None:
        data_dir = data_root.data_dir
    elif env_dir := os.environ.get("TASKCE_DATA_DIR"):
        data_dir = Path(env_dir)

    if data_dir is not None:
        override_path = data_dir / "profiles" / f"{as_text}.toml"
        if override_path.is_file():
            return override_path.resolve()

    # 3. Bundled profile under taskce/profiles/<name>.toml.
    builtin_path = BUILTIN_PROFILES_DIR / f"{as_text}.toml"
    if builtin_path.is_file():
        return builtin_path.resolve()

    available = ", ".join(list_builtin_profiles())
    raise FileNotFoundError(
        f"Unknown profile {as_text!r}; built-in profiles: {available}"
    )


def _read_toml_table(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Profile {path} must decode to a TOML table")
    _reject_unknown_keys(data, _ALLOWED_TOP_KEYS, str(path))
    return data


def _require_str(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_str(table: Mapping[str, Any], key: str, context: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string when provided")
    return value


def _str_tuple(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    value = table.get(key)
    if value is None:
        if required:
            raise ValueError(f"{context}.{key} is required")
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{context}.{key} must be a list of strings")
    if required and not value:
        raise ValueError(f"{context}.{key} must not be empty")
    return tuple(value)


def _validate_and_build(raw: Mapping[str, Any], *, source_path: Path) -> Profile:
    _reject_unknown_keys(raw, _ALLOWED_TOP_KEYS, str(source_path))
    name = _require_str(raw, "name", "profile")
    description = str(raw.get("description", ""))

    model_tbl = raw.get("model")
    if not isinstance(model_tbl, Mapping):
        raise ValueError(f"Profile {source_path} is missing [model] table")
    _reject_unknown_keys(model_tbl, _ALLOWED_MODEL_KEYS, "model")
    aliases_raw = model_tbl.get("layer_aliases", {})
    if not isinstance(aliases_raw, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in aliases_raw.items()
    ):
        raise ValueError("model.layer_aliases must map strings to strings")
    model = ModelSpec(
        model_id=_require_str(model_tbl, "model_id", "model"),
        lora_repo=_optional_str(model_tbl, "lora_repo", "model"),
        lora_task=_optional_str(model_tbl, "lora_task", "model"),
        checkpoint_name=_optional_str(model_tbl, "checkpoint_name", "model"),
        layers=tuple(LayerName(item) for item in _str_tuple(model_tbl, "layers", "model")),
        layer_aliases={k: LayerName(v) for k, v in aliases_raw.items()},
    )

    dataset_tbl = raw.get("dataset")
    if not isinstance(dataset_tbl, Mapping):
        raise ValueError(f"Profile {source_path} is missing [dataset] table")
    _reject_unknown_keys(dataset_tbl, _ALLOWED_DATASET_KEYS, "dataset")

    class_ids_raw = dataset_tbl.get("class_token_ids", {})
    if not isinstance(class_ids_raw, Mapping) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in class_ids_raw.items()
    ):
        raise ValueError("dataset.class_token_ids must map strings to integers")

    prompt_raw = dataset_tbl.get("prompt")
    prompt: PromptSpec | None = None
    if prompt_raw is not None:
        if not isinstance(prompt_raw, Mapping):
            raise ValueError("dataset.prompt must be a table")
        _reject_unknown_keys(prompt_raw, _ALLOWED_PROMPT_KEYS, "dataset.prompt")
        prompt = PromptSpec(
            system=_require_str(prompt_raw, "system", "dataset.prompt"),
            user_template=_require_str(prompt_raw, "user_template", "dataset.prompt"),
        )

    uses_pos = dataset_tbl.get("tokenizer_uses_pos", False)
    if not isinstance(uses_pos, bool):
        raise ValueError("dataset.tokenizer_uses_pos must be a boolean")

    dataset = DatasetSpec(
        name=_require_str(dataset_tbl, "name", "dataset"),
        hf_id=_require_str(dataset_tbl, "hf_id", "dataset"),
        split=_require_str(dataset_tbl, "split", "dataset"),
        feature_columns=_str_tuple(
            dataset_tbl, "feature_columns", "dataset", required=True
        ),
        label_column=_require_str(dataset_tbl, "label_column", "dataset"),
        labels=_str_tuple(dataset_tbl, "labels", "dataset"),
        class_token_ids=dict(class_ids_raw),
        prompt=prompt,
        tokenizer_name=_optional_str(dataset_tbl, "tokenizer_name", "dataset"),
        tokenizer_uses_pos=uses_pos,
    )

    search_tbl = raw.get("search", {})
    if not isinstance(search_tbl, Mapping):
        raise ValueError("search must be a table")
    _reject_unknown_keys(search_tbl, _ALLOWED_SEARCH_KEYS, "search")
    alpha = float(search_tbl.get("alpha", SearchDefaults.alpha))
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"search.alpha must be strictly between 0 and 1, got {alpha}")
    min_acts = search_tbl.get("min_acts", SearchDefaults.min_acts)
    beam = search_tbl.get("beam", SearchDefaults.beam)
    length = search_tbl.get("length", SearchDefaults.length)
    max_iterations = search_tbl.get("max_iterations", SearchDefaults.max_iterations)
    seed = search_tbl.get("seed", SearchDefaults.seed)
    for field_name, field_value in (
        ("beam", beam),
        ("length", length),
        ("max_iterations", max_iterations),
    ):
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value < 1
        ):
            raise ValueError(f"search.{field_name} must be a positive integer")
    if not isinstance(min_acts, int) or isinstance(min_acts, bool) or min_acts < 0:
        raise ValueError("search.min_acts must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("search.seed must be an integer")
    backend = search_tbl.get("backend", SearchDefaults.backend)
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"search.backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}"
        )
    search = SearchDefaults(
        alpha=alpha,
        min_acts=min_acts,
        beam=beam,
        length=length,
        max_iterations=max_iterations,
        seed=seed,
        backend=backend,
    )

    artifacts_tbl = raw.get("artifacts", {})
    if not isinstance(artifacts_tbl, Mapping):
        raise ValueError("artifacts must be a table")
    _reject_unknown_keys(artifacts_tbl, _ALLOWED_ARTIFACT_KEYS, "artifacts")
    artifacts = StudyArtifacts(
        baseline_commit=_optional_str(artifacts_tbl, "baseline_commit", "artifacts"),
        prepared_inputs_relpaths=_str_tuple(
            artifacts_tbl, "prepared_inputs_relpaths", "artifacts"
        ),
        joint_manifest_name=_optional_str(
            artifacts_tbl, "joint_manifest_name", "artifacts"
        ),
        joint_parent_csv_relpath=_optional_str(
            artifacts_tbl, "joint_parent_csv_relpath", "artifacts"
        ),
        joint_baseline_results_relpath=_optional_str(
            artifacts_tbl, "joint_baseline_results_relpath", "artifacts"
        ),
        recurrence_results_relpath=_optional_str(
            artifacts_tbl, "recurrence_results_relpath", "artifacts"
        ),
        recurrence_layers=_str_tuple(artifacts_tbl, "recurrence_layers", "artifacts"),
        recurrence_later_start=_optional_str(
            artifacts_tbl, "recurrence_later_start", "artifacts"
        ),
        recurrence_later_end=_optional_str(
            artifacts_tbl, "recurrence_later_end", "artifacts"
        ),
    )

    canonical_payload = {key: value for key, value in raw.items() if key != "extends"}
    return Profile(
        name=name,
        description=description,
        model=model,
        dataset=dataset,
        search=search,
        artifacts=artifacts,
        source_path=source_path,
        profile_sha256=sha256_json(canonical_payload),
    )


def load_profile(
    name_or_path: str | Path,
    *,
    data_root: DataRoot | None = None,
) -> Profile:
    """Load, merge optional single-level `extends`, and validate one TOML profile."""
    source_path = _resolve_profile_path(name_or_path, data_root=data_root)
    raw = _read_toml_table(source_path)

    parent_ref = raw.get("extends")
    if parent_ref is not None:
        if not isinstance(parent_ref, str) or not parent_ref.strip():
            raise ValueError(f"extends in {source_path} must be a non-empty string")
        parent_path = _resolve_profile_path(parent_ref, data_root=data_root)
        if parent_path == source_path:
            raise ValueError(f"Profile {source_path} cannot extend itself")
        parent_raw = _read_toml_table(parent_path)
        if "extends" in parent_raw:
            raise ValueError(
                f"Chained extends is not supported: {source_path} -> {parent_path}"
            )
        override = {key: value for key, value in raw.items() if key != "extends"}
        raw = _deep_merge(parent_raw, override)

    return _validate_and_build(raw, source_path=source_path)
