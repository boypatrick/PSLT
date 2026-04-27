#!/usr/bin/env python3
"""
Presubmit-style hygiene checks for the reviewer artifact registry.

This check is intentionally about packaging, not theorem content.  It verifies
that the generated registry is current, that reviewer-ledger certificate paths
are registered, and that comparator/diagnostic/candidate artifacts cannot be
mistaken for canonical baseline outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_BUILDER = ROOT / "code" / "build_artifact_status_registry.py"
OUTPUT_REGISTRY = ROOT / "output" / "repro" / "artifact_status.csv"
PAPER_REGISTRY = ROOT / "paper" / "artifact_status.csv"
MAIN_TEX = ROOT / "paper" / "main.tex"

ALLOWED_STATUSES = {
    "canonical_baseline",
    "canonical_certificate",
    "diagnostic_gate",
    "diagnostic_variant",
    "monitor_artifact",
    "legacy_comparator",
}

EXPECTED_STATUS_BY_PATH = {
    "paper/main.pdf": "canonical_baseline",
    "output/cap_resonance_1d/static_width_whittaker_special_interval_certificate_d6rc400_r60_r80_c640_summary.csv": "canonical_certificate",
    "output/cap_resonance_1d/static_width_whittaker_special_interval_certificate_d6rc400_r60_r80_c640_aggregate_summary.csv": "canonical_certificate",
    "output/hll_absolute_normalization/hll_absolute_normalization_synthesis_source_summary.csv": "canonical_certificate",
    "output/hll_absolute_normalization/hll_y2raw_adoption_impact_gate_summary.csv": "diagnostic_gate",
    "output/hll_absolute_normalization/hll_y2raw_projector_continuation_summary.csv": "canonical_certificate",
    "output/hll_absolute_normalization/hll_y2raw_topology_adoption_policy_summary.csv": "canonical_certificate",
    "output/hll_absolute_normalization/hll_y2raw_topology_margin_certificate_summary.csv": "canonical_certificate",
    "paper/hll_uv_action_loop_eymh_feshbach_complement_summary_D21E21_fix.csv": "canonical_certificate",
    "output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_summary.csv": "canonical_certificate",
    "output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_summary.csv": "canonical_certificate",
    "output/superrad_fp_1d/channel_resolved_A_l_tensor_D4-20full_eta3_N1-2-3_l1-2_summary.csv": "monitor_artifact",
    "output/superrad_fp_1d/channel_A_tensor_impact_Dgrid60_Egrid21_summary.csv": "monitor_artifact",
    "output/tcoh_fp_1d/tcoh_impact_gate_Dgrid60_Egrid21_summary.csv": "diagnostic_gate",
    "output/eta_fp_1d/eta_impact_gate_Dgrid60_Egrid21_summary.csv": "monitor_artifact",
    "paper/open_system_micro_baseline_candidate.csv": "diagnostic_variant",
    "paper/kappa_env_anchor_calibration.csv": "diagnostic_variant",
    "paper/kappa_env_anchor_holdout.csv": "diagnostic_variant",
    "output/kinetic_action_chain/action_chain_consistency_summary.csv": "legacy_comparator",
    "output/kinetic_action_chain/action_chain_consistency.png": "legacy_comparator",
}

BASELINE_FORBIDDEN_PATH_RE = re.compile(
    r"(legacy|diagnostic|candidate|holdout|stress|smoke|comparator|extreme|"
    r"kappa_env|open_system_micro|source_projection|nonhermitian)",
    re.IGNORECASE,
)

REQUIRED_MAIN_TEXT = [
    "For reviewer-facing reading, the status words below are intentional",
    "U9 is structurally closed, U10 is diagnostic-only under the strict no-topology gate, and U11 closes the projected-kernel complement",
    "U14 certifies the direct branch margin",
    "S25 is adopted as a local strict parent-tube theorem",
    "It is not a global complex-pole theorem and not a new global pole family",
    "tensor extraction is a monitor artifact",
    "the reviewer-facing exact-bridge problem is closed",
    "It is therefore diagnostic-only rather than a baseline replacement",
    r"\(\eta_{\rm fp}\) is scaled-only adoption-safe",
]


def fail(errors: List[str], message: str) -> None:
    errors.append(message)


def load_builder_rows() -> List[Dict[str, object]]:
    spec = importlib.util.spec_from_file_location("artifact_status_builder", REGISTRY_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REGISTRY_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rows()


def expected_csv_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in load_builder_rows():
        materialized = {**row, "exists": bool((ROOT / str(row["artifact_path"])).exists())}
        rows.append({key: str(value) for key, value in materialized.items()})
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def path_set(rows: Iterable[Dict[str, str]]) -> set[str]:
    return {row["artifact_path"] for row in rows}


def check_csv_matches(expected: List[Dict[str, str]], errors: List[str]) -> None:
    for csv_path in (OUTPUT_REGISTRY, PAPER_REGISTRY):
        if not csv_path.exists():
            fail(errors, f"missing generated registry: {csv_path}")
            continue
        actual = read_csv_rows(csv_path)
        if actual != expected:
            fail(
                errors,
                f"stale registry: {csv_path} does not match code/build_artifact_status_registry.py; rerun the builder",
            )


def check_registry_semantics(rows: List[Dict[str, str]], errors: List[str]) -> None:
    counts = Counter(row["artifact_path"] for row in rows)
    duplicates = sorted(path for path, count in counts.items() if count > 1)
    if duplicates:
        fail(errors, "duplicate registry paths: " + ", ".join(duplicates))

    by_path = {row["artifact_path"]: row for row in rows}

    for row in rows:
        status = row["status"]
        path = row["artifact_path"]
        if status not in ALLOWED_STATUSES:
            fail(errors, f"unknown status {status!r} for {path}")
        if row["exists"] != "True":
            fail(errors, f"registered artifact does not exist: {path}")
        if status == "canonical_baseline" and BASELINE_FORBIDDEN_PATH_RE.search(path):
            fail(errors, f"comparator-like path is marked canonical_baseline: {path}")

    for path, expected_status in EXPECTED_STATUS_BY_PATH.items():
        row = by_path.get(path)
        if row is None:
            fail(errors, f"required reviewer artifact is missing from registry: {path}")
        elif row["status"] != expected_status:
            fail(
                errors,
                f"wrong status for {path}: expected {expected_status}, got {row['status']}",
            )


def check_ledger_paths(rows: List[Dict[str, str]], errors: List[str]) -> None:
    text = MAIN_TEX.read_text()
    start = text.find(r"\subsection{Reviewer-Facing Appendix Ledger}")
    end = text.find(r"\subsection{Single Entry Point}", start)
    if start < 0 or end < 0:
        fail(errors, "could not locate reviewer-facing appendix ledger block")
        return

    ledger = text[start:end]
    ledger_paths = sorted(set(re.findall(r"\\path\{([^}]+)\}", ledger)))
    registered = path_set(rows)
    missing = [path for path in ledger_paths if path not in registered]
    if missing:
        fail(errors, "ledger paths missing from registry: " + ", ".join(missing))

    missing_files = [path for path in ledger_paths if not (ROOT / path).exists()]
    if missing_files:
        fail(errors, "ledger paths missing on disk: " + ", ".join(missing_files))


def check_main_text_status_words(errors: List[str]) -> None:
    text = MAIN_TEX.read_text()
    for needle in REQUIRED_MAIN_TEXT:
        if needle not in text:
            fail(errors, f"main-text status wording missing: {needle}")


def main() -> int:
    errors: List[str] = []
    expected = expected_csv_rows()

    check_csv_matches(expected, errors)
    check_registry_semantics(expected, errors)
    check_ledger_paths(expected, errors)
    check_main_text_status_words(errors)

    if errors:
        print("[artifact hygiene] FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    status_counts = Counter(row["status"] for row in expected)
    print("[artifact hygiene] PASS")
    print(f"  registry rows: {len(expected)}")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")
    print("  generated registries match builder")
    print("  reviewer-ledger certificate paths are registered and present")
    print("  comparator/diagnostic/candidate paths are not canonical baselines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
