"""Cryptographic and module-name pin loader for the graduated 2.a backend."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from taskce.runtime import (
    PINNED_2A_COMMIT,
    PINNED_2A_HASHES,
    PINNED_2A_MODULE_KEYS,
    PINNED_FORMULA_EVAL_SHA256,
    Sha256Hex,
    sha256_file,
)


COMMIT = PINNED_2A_COMMIT
HASHES = dict(PINNED_2A_HASHES)

_MODULE_FILES: tuple[tuple[str, str], ...] = (
    ("vectorized_search", "__init__.py"),
    ("vectorized_search.formula", "formula.py"),
    ("vectorized_search.algorithm", "algorithm.py"),
)


@dataclass(frozen=True)
class ModuleProof:
    path: str
    sha256: Sha256Hex


@dataclass(frozen=True)
class BackendProof:
    """Matches exact `verify_2a.verify()` schema expected by saved manifests."""

    commit: str
    modules: Mapping[str, ModuleProof]
    entry: str

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize to dict matching historical `validation.json` format."""
        return {
            "commit": self.commit,
            "modules": {
                name: {"path": proof.path, "sha256": str(proof.sha256)}
                for name, proof in self.modules.items()
            },
            "entry": self.entry,
        }

    def module_hashes(self) -> dict[str, str]:
        """Return `{module_name: sha256}` matching historical `backend_sha256`."""
        return {name: str(proof.sha256) for name, proof in self.modules.items()}


@dataclass(frozen=True)
class VerifiedBackend2a:
    proof: BackendProof
    evaluator_sha256: Sha256Hex
    search_config: type[Any] | None
    search_all: Callable[..., list[Any]] | None


def verify_formula_eval() -> Sha256Hex:
    """Verify `formula_search/formula_eval.py` matches `PINNED_FORMULA_EVAL_SHA256`."""
    eval_path = Path(__file__).resolve().parent / "formula_eval.py"
    if not eval_path.is_file():
        raise RuntimeError(f"Pinned formula_eval.py not found: {eval_path}")
    actual = sha256_file(eval_path)
    if actual != PINNED_FORMULA_EVAL_SHA256:
        raise RuntimeError(
            f"Pinned formula_eval.py hash mismatch: {actual} != {PINNED_FORMULA_EVAL_SHA256}"
        )
    return actual


def _verify_raw_hashes(pkg_dir: Path) -> dict[str, tuple[Path, Sha256Hex]]:
    verified: dict[str, tuple[Path, Sha256Hex]] = {}
    for filename, expected_sha in PINNED_2A_HASHES.items():
        file_path = (pkg_dir / filename).resolve()
        if not file_path.is_file():
            raise RuntimeError(f"Missing pinned 2.a file: {file_path}")
        actual = sha256_file(file_path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Pinned 2.a hash mismatch for {filename}: {actual} != {expected_sha}"
            )
        verified[filename] = (file_path, actual)
    return verified


def _guard_existing_modules(pkg_dir: Path) -> None:
    for mod_name, filename in _MODULE_FILES:
        existing = sys.modules.get(mod_name)
        if existing is None:
            continue
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != (pkg_dir / filename).resolve():
            raise RuntimeError(
                f"Conflicting {mod_name} already loaded from {existing_file}"
            )


def _load_package_without_syspath(pkg_dir: Path) -> None:
    if "vectorized_search" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "vectorized_search",
        pkg_dir / "__init__.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not construct module spec for {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vectorized_search"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("vectorized_search", None)
        raise


def _has_top_level_def(source_path: Path, name: str) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in tree.body
    )


def verify(*, require_runtime: bool = False) -> dict[str, Any]:
    """Verify and load pinned 2.a under exact module names without mutating `sys.path`.

    Hashes all three files before executing any code. When Triton is installed
    (or `require_runtime=True`), imports `vectorized_search.algorithm` and checks
    `search_all.__module__`; on CPU-only environments without Triton, verifies
    all file digests and loads `vectorized_search` and `vectorized_search.formula`.
    """
    pkg_dir = (Path(__file__).resolve().parent / "vectorized_search").resolve()
    verified_files = _verify_raw_hashes(pkg_dir)
    _guard_existing_modules(pkg_dir)
    _load_package_without_syspath(pkg_dir)

    importlib.import_module("vectorized_search.formula")

    triton_available = importlib.util.find_spec("triton") is not None
    if triton_available or require_runtime:
        algorithm = importlib.import_module("vectorized_search.algorithm")
        if algorithm.search_all.__module__ != "vectorized_search.algorithm":
            raise RuntimeError("Unexpected search_all entry point")
        entry = algorithm.search_all.__qualname__
    else:
        algo_path = verified_files["algorithm.py"][0]
        if not _has_top_level_def(algo_path, "search_all"):
            raise RuntimeError("Pinned algorithm.py is missing search_all")
        entry = "search_all"

    modules = {
        mod_name: {
            "path": str(verified_files[filename][0]),
            "sha256": str(verified_files[filename][1]),
        }
        for mod_name, filename in _MODULE_FILES
    }
    if set(modules) != set(PINNED_2A_MODULE_KEYS):
        raise RuntimeError(f"Unexpected verified module keys: {sorted(modules)}")

    return {"commit": PINNED_2A_COMMIT, "modules": modules, "entry": entry}


def load_verified_2a(*, require_runtime: bool = False) -> VerifiedBackend2a:
    """Verify pinned `vectorized_search` (3 files) and `formula_eval.py`, returning typed handles."""
    raw_proof = verify(require_runtime=require_runtime)
    eval_sha = verify_formula_eval()
    proof = BackendProof(
        commit=raw_proof["commit"],
        modules={
            name: ModuleProof(path=item["path"], sha256=Sha256Hex(item["sha256"]))
            for name, item in raw_proof["modules"].items()
        },
        entry=raw_proof["entry"],
    )
    algorithm = sys.modules.get("vectorized_search.algorithm")
    return VerifiedBackend2a(
        proof=proof,
        evaluator_sha256=eval_sha,
        search_config=getattr(algorithm, "SearchConfig", None) if algorithm is not None else None,
        search_all=getattr(algorithm, "search_all", None) if algorithm is not None else None,
    )


if __name__ == "__main__":
    print(json.dumps(verify()), flush=True)
