"""Fail before fitting if Python resolves anything but the pinned 2.a package."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

COMMIT = "6f2503fe78a05197943c7349d065224dd84648b0"
HASHES = {
    "__init__.py": "31940eb8d8fd563544b6a10603e77306e981077c23fe18aa92f40f527a34eead",
    "algorithm.py": "b6625658a1268ca07f96da6734da980eb43cfe43b39c38462fa09f52382d9683",
    "formula.py": "14d24e6ccefe62914a7d4462362caae99a698ba929be03d7a42303ac4b884f7b",
}


def verify():
    directory = Path(__file__).resolve().parents[2] / "theoretical/compositional_explanations"
    sys.path.insert(0, str(directory))
    modules = {}
    for name, expected in HASHES.items():
        package = "vectorized_search"
        module = importlib.import_module(package if name == "__init__.py" else f"{package}.{name[:-3]}")
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path != directory / package / name or digest != expected:
            raise RuntimeError(f"Unexpected import {path} {digest}")
        modules[module.__name__] = dict(path=str(path), sha256=digest)
    algorithm = sys.modules["vectorized_search.algorithm"]
    if algorithm.search_all.__module__ != algorithm.__name__:
        raise RuntimeError("Unexpected search_all entry point")
    return dict(commit=COMMIT, modules=modules, entry=algorithm.search_all.__qualname__)


if __name__ == "__main__":
    print(json.dumps(verify()), flush=True)
