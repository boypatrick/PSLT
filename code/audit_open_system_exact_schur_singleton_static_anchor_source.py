#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def _read_singleton_parent() -> dict[str, float]:
    row = pd.read_csv(OUT_DIR / "chi_open_system_exact_schur_singleton_parent_source_summary.csv").iloc[0]
    return {k: float(v) for k, v in row.to_dict().items()}


def _read_self_core_anchor() -> dict[str, float]:
    df = pd.read_csv(OUT_DIR / "chi_open_system_parity_contrast_self_core_anchor_source_detail.csv")
    row = df[df["D"] == 11.0].iloc[0]
    return {k: float(v) for k, v in row.to_dict().items()}


def _read_anchor_static() -> dict[str, float]:
    df = pd.read_csv(OUT_DIR / "chi_open_system_parity_contrast_anchor_static_source_detail.csv")
    row = df[df["D"] == 11.0].iloc[0]
    out = {}
    for k, v in row.to_dict().items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def main() -> None:
    singleton = _read_singleton_parent()
    core = _read_self_core_anchor()
    anchor = _read_anchor_static()

    required_core_global = float(singleton["required_core_floor_global"])
    required_core_local = float(singleton["required_core_floor_local"])

    self_pair_core = abs(float(core["self_core_exact"]))
    self_anchor = abs(float(core["self_anchor"]))
    self_shell = abs(float(core["self_shell"]))

    diag_anchor = abs(float(anchor["diag_anchor"]))
    mirror_anchor = abs(float(anchor["mirror_anchor"]))
    anchor_exact = abs(float(anchor["anchor_exact"]))
    mu_anchor_11 = mirror_anchor / max(diag_anchor, 1e-300)
    induced_anchor_from_diag = (1.0 - mu_anchor_11) * diag_anchor

    required_diag_global = required_core_global / max(1.0 - mu_anchor_11, 1e-300)
    required_diag_local = required_core_local / max(1.0 - mu_anchor_11, 1e-300)

    summary = {
        "D": 11.0,
        "R_star": float(core["R_star"]),
        "rho_anchor": float(anchor["rho_anchor"]),
        "zeta_anchor": float(anchor["zeta_anchor"]),
        "self_pair_core": self_pair_core,
        "self_anchor": self_anchor,
        "self_shell": self_shell,
        "anchor_same_sign_as_core": int(core["anchor_same_sign_as_core"]),
        "shell_same_sign_as_core": int(core["shell_same_sign_as_core"]),
        "anchor_abs_share_of_core": self_anchor / max(self_pair_core, 1e-300),
        "shell_abs_share_of_core": self_shell / max(self_pair_core, 1e-300),
        "diag_anchor": diag_anchor,
        "mirror_anchor": mirror_anchor,
        "anchor_exact": anchor_exact,
        "diag_same_sign_as_anchor": int(anchor["diag_same_sign_as_anchor"]),
        "mirror_same_sign_as_anchor": int(anchor["mirror_same_sign_as_anchor"]),
        "mu_anchor_11": mu_anchor_11,
        "induced_anchor_from_diag": induced_anchor_from_diag,
        "anchor_reconstruction_residual": anchor_exact - induced_anchor_from_diag,
        "required_core_floor_global": required_core_global,
        "required_core_floor_local": required_core_local,
        "required_diag_floor_global": required_diag_global,
        "required_diag_floor_local": required_diag_local,
        "anchor_over_required_core_global": self_anchor / max(required_core_global, 1e-300),
        "anchor_over_required_core_local": self_anchor / max(required_core_local, 1e-300),
        "diag_over_required_global": diag_anchor / max(required_diag_global, 1e-300),
        "diag_over_required_local": diag_anchor / max(required_diag_local, 1e-300),
    }

    detail_rows = [
        {"term": "self_pair_core", "value": self_pair_core, "role": "parent_target"},
        {"term": "self_anchor", "value": self_anchor, "role": "same_sign_core_subanchor"},
        {"term": "self_shell", "value": self_shell, "role": "same_sign_completion"},
        {"term": "diag_anchor", "value": diag_anchor, "role": "static_parent_candidate"},
        {"term": "mirror_anchor", "value": mirror_anchor, "role": "local_mirror_loss"},
        {"term": "induced_anchor_from_diag", "value": induced_anchor_from_diag, "role": "diag_to_anchor_lower_bound"},
        {"term": "required_core_floor_global", "value": required_core_global, "role": "global_target"},
        {"term": "required_core_floor_local", "value": required_core_local, "role": "local_target"},
        {"term": "required_diag_floor_global", "value": required_diag_global, "role": "global_static_target"},
        {"term": "required_diag_floor_local", "value": required_diag_local, "role": "local_static_target"},
    ]

    summary_path = OUT_DIR / "chi_open_system_exact_schur_singleton_static_anchor_source_summary.csv"
    detail_path = OUT_DIR / "chi_open_system_exact_schur_singleton_static_anchor_source_detail.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
