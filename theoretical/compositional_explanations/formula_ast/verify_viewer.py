"""Check that an embedded formula viewer still matches its source CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from html.parser import HTMLParser
from pathlib import Path

from .analysis import parse_formula
from .viewer import display_ast


class _EmbeddedRecords(HTMLParser):
    """Collect the JSON payload from the page's records script element."""

    def __init__(self) -> None:
        super().__init__()
        self._in_records = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "records":
            self._in_records = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_records:
            self._in_records = False

    def handle_data(self, data: str) -> None:
        if self._in_records:
            self._parts.append(data)


def _embedded_payload(html_path: Path) -> dict:
    parser = _EmbeddedRecords()
    parser.feed(html_path.read_text(encoding="utf-8"))
    if not parser._parts:
        raise ValueError(f"{html_path}: missing <script id=\"records\"> payload")
    try:
        payload = json.loads("".join(parser._parts))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{html_path}: invalid records JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"{html_path}: payload must contain a datasets array")
    return payload


def verify_viewer(html_path: Path, csv_paths: list[Path]) -> list[tuple[str, int, str]]:
    """Compare embedded dataset metadata and records against raw CSV contents."""
    if not csv_paths:
        raise ValueError("provide at least one CSV")
    payload = _embedded_payload(html_path)
    datasets = payload["datasets"]
    if len(datasets) != len(csv_paths):
        raise ValueError(f"{html_path}: found {len(datasets)} datasets for {len(csv_paths)} CSV files")

    summaries = []
    for index, csv_path in enumerate(csv_paths):
        dataset = datasets[index]
        if not isinstance(dataset, dict):
            raise ValueError(f"{html_path}: dataset {index + 1} must be an object")
        layer = csv_path.parent.name if csv_path.parent.name.startswith("model.") else csv_path.stem
        raw_csv = csv_path.read_bytes()
        expected_hash = hashlib.sha256(raw_csv).hexdigest()
        if dataset.get("source") != csv_path.resolve().as_posix():
            raise ValueError(f"{html_path}: dataset {index + 1} source mismatch; expected {csv_path.resolve().as_posix()!r}, found {dataset.get('source')!r}")
        if dataset.get("id") != layer or dataset.get("layer") != layer:
            raise ValueError(f"{html_path}: dataset {index + 1} layer/id mismatch; expected {layer!r}")
        if dataset.get("sha256") != expected_hash:
            raise ValueError(f"{html_path}: {layer} CSV hash mismatch; expected {expected_hash}, found {dataset.get('sha256')!r}")

        records = dataset.get("records")
        if not isinstance(records, list):
            raise ValueError(f"{html_path}: {layer} records must be an array")
        with io.StringIO(raw_csv.decode("utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"neuron", "formula", "iou"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path}: expected columns neuron, formula, iou")
            source_rows = list(reader)
        if len(records) != len(source_rows):
            raise ValueError(f"{html_path}: {layer} has {len(records)} records; {csv_path} has {len(source_rows)} rows")
        for row_number, (record, row) in enumerate(zip(records, source_rows), start=2):
            location = f"{html_path}: {layer}, CSV row {row_number}"
            if not isinstance(record, dict):
                raise ValueError(f"{location}: record must be an object")
            try:
                neuron_id = int(row["neuron"])
                iou = float(row["iou"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}, row {row_number}: invalid neuron or iou value: {exc}") from exc
            if not math.isfinite(iou):
                raise ValueError(f"{csv_path}, row {row_number}: IoU must be finite")
            for key, expected in (("neuronId", neuron_id), ("formula", row["formula"])):
                if record.get(key) != expected:
                    raise ValueError(f"{location}: {key} mismatch; expected {expected!r}, found {record.get(key)!r}")
            actual_iou = record.get("iou")
            if isinstance(actual_iou, bool) or not isinstance(actual_iou, (int, float)) or actual_iou != iou:
                raise ValueError(f"{location}: iou mismatch; expected {iou!r}, found {actual_iou!r}")
            if row["formula"] == "LOW_ACTS_PRUNED":
                if record.get("ast", object()) is not None:
                    raise ValueError(f"{location}: LOW_ACTS_PRUNED must have ast=null")
            else:
                ast = record.get("ast")
                if not isinstance(ast, dict):
                    raise ValueError(f"{location}: non-pruned formula must have an AST object")
                expected_ast = display_ast(parse_formula(row["formula"]))
                if ast != expected_ast:
                    raise ValueError(f"{location}: AST mismatch; expected {expected_ast!r}, found {ast!r}")
        summaries.append((layer, len(records), expected_hash))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True, help="generated viewer HTML file")
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="source CSV files in viewer dataset order")
    args = parser.parse_args()
    try:
        summaries = verify_viewer(args.html, args.csv)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    for layer, count, digest in summaries:
        print(f"{layer}: {count:,} records matched; sha256 {digest}")
    print(f"Verified {sum(count for _, count, _ in summaries):,} records across {len(summaries)} CSV files")


if __name__ == "__main__":
    main()
