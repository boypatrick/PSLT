#!/usr/bin/env python3
"""
Search D-only risk-weighted profile anchoring candidates for the promoted
runtime-direct visibility branch.

The search uses the instability D-profile produced by
`scan_runtime_direct_visibility_instability.py` and generates candidate
alpha(D) profiles of the form

  alpha(D) = alpha_min + (alpha_max - alpha_min) * risk_norm(D) ** power

where risk_norm(D) is the cross-surface maximum of per-surface normalized
instability. Each candidate is validated against both release gates:

  - D21xE41
  - D60xE21

The best candidate is the one that passes both gates with the smallest
mean(alpha), then the smallest p90(alpha), then the smallest worst max|Δμ|.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
AUDIT_SCRIPT = ROOT / "code" / "scan_chain_mode_cell_direct_audit.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search risk-weighted alpha(D) candidates for runtime-direct visibility.")
    ap.add_argument(
        "--instability-d-profile",
        type=Path,
        default=OUTDIR / "runtime_direct_visibility_instability_D_profile.csv",
    )
    ap.add_argument("--alpha-max", type=float, default=0.99)
    ap.add_argument("--alpha-min-grid", type=str, default="0.85,0.90,0.93,0.95")
    ap.add_argument("--power-grid", type=str, default="1.0,1.5,2.0")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("Empty grid specification.")
    return vals


def build_risk_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    need = {"surface", "D", "max_risk_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {sorted(need - set(df.columns))}")
    pieces = []
    for surface, grp in df.groupby("surface", sort=True):
        g = grp.copy()
        denom = float(g["max_risk_score"].max())
        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
        g["risk_norm_surface"] = (g["max_risk_score"] / denom).clip(lower=0.0, upper=1.0)
        pieces.append(g[["surface", "D", "risk_norm_surface", "max_risk_score"]])
    norm_df = pd.concat(pieces, ignore_index=True)
    agg = (
        norm_df.groupby("D", as_index=False)
        .agg(
            risk_norm=("risk_norm_surface", "max"),
            max_risk_score=("max_risk_score", "max"),
            supporting_surfaces=("surface", lambda s: ",".join(sorted(set(str(x) for x in s)))),
        )
        .sort_values("D")
        .reset_index(drop=True)
    )
    return agg


def candidate_label(alpha_min: float, power: float) -> str:
    a = str(alpha_min).replace(".", "p")
    p = str(power).replace(".", "p")
    return f"amin{a}_pow{p}"


def write_alpha_profile(df_risk: pd.DataFrame, alpha_min: float, alpha_max: float, power: float, out_csv: Path) -> pd.DataFrame:
    out = df_risk.copy()
    out["alpha"] = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * np.power(
        np.asarray(out["risk_norm"], dtype=float), float(power)
    )
    out["alpha"] = out["alpha"].clip(lower=0.0, upper=1.0)
    out.to_csv(out_csv, index=False)
    return out


def run_audit(
    d_num: int,
    eta_num: int,
    blend_csv: Path,
    full_tag: str,
    cell_tag: str,
) -> pd.Series:
    cmd = [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--d-min",
        "4.0",
        "--d-max",
        "20.0",
        "--d-num",
        str(int(d_num)),
        "--eta-min",
        "0.2",
        "--eta-max",
        "4.0",
        "--eta-num",
        str(int(eta_num)),
        "--observable-mode",
        "eft_wilson_uv_rge",
        "--ref-mode",
        "fixed",
        "--ref-d",
        "10.0",
        "--ref-eta",
        "1.0",
        "--cell-chain-mode",
        "cell_direct_runtime_release_tuned",
        "--full-direct-tag",
        str(full_tag),
        "--cell-direct-tag",
        str(cell_tag),
        "--runtime-direct-b-release-profile-blend-csv-override",
        str(blend_csv),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    out_csv = OUTDIR / f"chain_mode_cell_direct_audit_Dgrid{int(d_num)}_Egrid{int(eta_num)}_cell_direct_runtime_release_tuned.csv"
    return pd.read_csv(out_csv).iloc[0]


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    alpha_min_grid = parse_float_grid(args.alpha_min_grid)
    power_grid = parse_float_grid(args.power_grid)
    risk_df = build_risk_profile(args.instability_d_profile)

    rows: List[Dict[str, object]] = []
    best_row: Dict[str, object] | None = None
    best_profile_path: Path | None = None

    for alpha_min in alpha_min_grid:
        for power in power_grid:
            label = candidate_label(alpha_min, power)
            profile_path = OUTDIR / f"runtime_direct_visibility_alphaD_profile_{label}.csv"
            alpha_df = write_alpha_profile(
                risk_df,
                alpha_min=float(alpha_min),
                alpha_max=float(args.alpha_max),
                power=float(power),
                out_csv=profile_path,
            )

            row: Dict[str, object] = {
                "label": label,
                "alpha_min": float(alpha_min),
                "alpha_max": float(args.alpha_max),
                "power": float(power),
                "mean_alpha": float(alpha_df["alpha"].mean()),
                "p90_alpha": float(alpha_df["alpha"].quantile(0.9)),
                "min_alpha": float(alpha_df["alpha"].min()),
                "max_alpha": float(alpha_df["alpha"].max()),
                "profile_csv": str(profile_path),
            }

            audit21 = run_audit(
                d_num=21,
                eta_num=41,
                blend_csv=profile_path,
                full_tag=f"alphaD_{label}_full_D21E41",
                cell_tag=f"alphaD_{label}_cell_D21E41",
            )
            audit60 = run_audit(
                d_num=60,
                eta_num=21,
                blend_csv=profile_path,
                full_tag=f"alphaD_{label}_full_D60E21",
                cell_tag=f"alphaD_{label}_cell_D60E21",
            )

            row.update(
                {
                    "d21_frac_acceptance_mismatch": float(audit21["frac_acceptance_mismatch"]),
                    "d21_max_abs_delta_mu_mumu": float(audit21["max_abs_delta_mu_mumu"]),
                    "d60_frac_acceptance_mismatch": float(audit60["frac_acceptance_mismatch"]),
                    "d60_max_abs_delta_mu_mumu": float(audit60["max_abs_delta_mu_mumu"]),
                }
            )
            row["passes_release_gate"] = bool(
                row["d21_frac_acceptance_mismatch"] <= 0.01
                and row["d21_max_abs_delta_mu_mumu"] <= 1.0
                and row["d60_frac_acceptance_mismatch"] <= 0.01
                and row["d60_max_abs_delta_mu_mumu"] <= 1.0
            )
            row["worst_gate_delta_mu_mumu"] = float(
                max(row["d21_max_abs_delta_mu_mumu"], row["d60_max_abs_delta_mu_mumu"])
            )
            rows.append(row)

            if row["passes_release_gate"]:
                if best_row is None:
                    best_row = row
                    best_profile_path = profile_path
                else:
                    current_key = (
                        float(row["mean_alpha"]),
                        float(row["p90_alpha"]),
                        float(row["worst_gate_delta_mu_mumu"]),
                    )
                    best_key = (
                        float(best_row["mean_alpha"]),
                        float(best_row["p90_alpha"]),
                        float(best_row["worst_gate_delta_mu_mumu"]),
                    )
                    if current_key < best_key:
                        best_row = row
                        best_profile_path = profile_path

    result_df = pd.DataFrame(rows).sort_values(
        ["passes_release_gate", "mean_alpha", "p90_alpha", "worst_gate_delta_mu_mumu"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    summary = {
        "instability_d_profile": str(args.instability_d_profile),
        "alpha_min_grid": alpha_min_grid,
        "power_grid": power_grid,
        "alpha_max": float(args.alpha_max),
        "n_candidates": int(len(result_df)),
        "n_gate_pass": int(result_df["passes_release_gate"].sum()),
        "best_label": None if best_row is None else str(best_row["label"]),
        "best_profile_csv": None if best_profile_path is None else str(best_profile_path),
    }
    if best_row is not None:
        summary.update(
            {
                "best_mean_alpha": float(best_row["mean_alpha"]),
                "best_p90_alpha": float(best_row["p90_alpha"]),
                "best_min_alpha": float(best_row["min_alpha"]),
                "best_d21_max_abs_delta_mu_mumu": float(best_row["d21_max_abs_delta_mu_mumu"]),
                "best_d60_max_abs_delta_mu_mumu": float(best_row["d60_max_abs_delta_mu_mumu"]),
            }
        )

    out_scan = OUTDIR / "runtime_direct_visibility_alphaD_scan.csv"
    out_summary = OUTDIR / "runtime_direct_visibility_alphaD_summary.json"
    result_df.to_csv(out_scan, index=False)
    out_summary.write_text(json.dumps(summary, indent=2))

    if best_profile_path is not None:
        best_copy = OUTDIR / "runtime_direct_visibility_alphaD_profile_best.csv"
        best_copy.write_bytes(best_profile_path.read_bytes())
        if not args.skip_paper_copy:
            (PAPER_DIR / best_copy.name).write_bytes(best_copy.read_bytes())

    if not args.skip_paper_copy:
        for src in [out_scan, out_summary]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(result_df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_scan}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
