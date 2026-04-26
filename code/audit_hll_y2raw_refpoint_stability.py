#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the reference-point stability of y_2^{raw}(D_*), separating "
            "the exact overlap identity from branch-continuation and interpolation floors."
        )
    )
    parser.add_argument(
        "--canonical-profile-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        ),
    )
    parser.add_argument(
        "--single-dstar-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D9.6.csv"
        ),
    )
    parser.add_argument(
        "--local-window-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D9-9.6-10.csv"
        ),
    )
    parser.add_argument(
        "--inserted-profile-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-9.6-10-11-12-13-14-15-16-17-18-19-20.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--tree-constants-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_tree_mumu_parent_source_constants.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    return parser.parse_args()


def fine_rows(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    return df[df["level"].astype(str).str.lower() == "fine"].sort_values("D").reset_index(drop=True)


def all_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).copy().sort_values(["D", "level"]).reset_index(drop=True)


def ycols(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            float(row["y_eff_raw_1"]),
            float(row["y_eff_raw_2"]),
            float(row["y_eff_raw_3"]),
        ],
        dtype=float,
    )


def branch_cols(row: pd.Series) -> tuple[int, int, int]:
    return (
        int(row["track_idx_1"]),
        int(row["track_idx_2"]),
        int(row["track_idx_3"]),
    )


def rel(a: float, b: float) -> float:
    return float((a - b) / max(abs(b), 1e-300))


def main() -> None:
    args = parse_args()
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    tree_constants = pd.read_csv(args.tree_constants_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta_summary = float(summary_row["ref_eta"])

    canonical = fine_rows(args.canonical_profile_input)
    single = all_rows(args.single_dstar_input)
    local = all_rows(args.local_window_input)
    inserted = all_rows(args.inserted_profile_input)
    inserted_fine = inserted[
        (inserted["level"].astype(str).str.lower() == "fine")
        & np.isclose(inserted["D"].astype(float), ref_d)
    ]
    if inserted_fine.empty:
        raise ValueError("Inserted profile must contain a fine row at D_*.")
    inserted_ref = inserted_fine.iloc[0]
    y_direct = float(inserted_ref["y_eff_raw_2"])

    y9 = float(canonical.loc[np.isclose(canonical["D"], 9.0), "y_eff_raw_2"].iloc[0])
    y10 = float(canonical.loc[np.isclose(canonical["D"], 10.0), "y_eff_raw_2"].iloc[0])
    y_linear = float(np.interp(ref_d, [9.0, 10.0], [y9, y10]))
    y_log = float(np.exp(np.interp(ref_d, [9.0, 10.0], np.log([y9, y10]))))

    detail_rows: list[dict[str, object]] = []
    for source, df in [
        ("single_point_eigen_order", single),
        ("local_9_9p6_10_history", local),
        ("canonical_inserted_history", inserted),
    ]:
        sub = df[np.isclose(df["D"].astype(float), ref_d)].copy()
        for _, row in sub.iterrows():
            ys = ycols(row)
            closest = int(np.argmin(np.abs(ys - y_direct)))
            detail_rows.append(
                {
                    "source": source,
                    "D": float(row["D"]),
                    "level": str(row["level"]),
                    "track_idx_1": int(row["track_idx_1"]),
                    "track_idx_2": int(row["track_idx_2"]),
                    "track_idx_3": int(row["track_idx_3"]),
                    "branch_indices": "/".join(str(x) for x in branch_cols(row)),
                    "y_eff_raw_1": float(ys[0]),
                    "y_eff_raw_2": float(ys[1]),
                    "y_eff_raw_3": float(ys[2]),
                    "canonical_target_branch": closest + 1,
                    "canonical_target_y": float(ys[closest]),
                    "rel_target_vs_inserted_fine": rel(float(ys[closest]), y_direct),
                    "reported_y2_rel_vs_inserted_fine": rel(float(ys[1]), y_direct),
                }
            )
    detail = pd.DataFrame(detail_rows)

    inserted_target = detail[detail["source"] == "canonical_inserted_history"].copy()
    inserted_target["abs_target_rel_vs_fine"] = inserted_target["rel_target_vs_inserted_fine"].abs()
    max_grid_rel = float(inserted_target["abs_target_rel_vs_fine"].max())
    mid_grid_rel = float(
        inserted_target.loc[
            inserted_target["level"].astype(str).str.lower() == "mid",
            "abs_target_rel_vs_fine",
        ].iloc[0]
    )
    coarse_grid_rel = float(
        inserted_target.loc[
            inserted_target["level"].astype(str).str.lower() == "coarse",
            "abs_target_rel_vs_fine",
        ].iloc[0]
    )

    coeff_ref = float(tree_constants["coeff2_kin_over_m2_ref"])
    c_tree_current = float(tree_constants["C_tree_mumu_ref"])
    a_tree_current = float(tree_constants["A_tree_ref"])
    a_star_current = float(tree_constants["A_star_ref"])
    z_diag_ref = float(tree_constants["z_diag_ref"])
    c_tree_direct = y_direct * coeff_ref
    a_tree_direct = 1.0 / max(c_tree_direct * c_tree_direct, 1e-300)
    a_star_direct = a_tree_direct / max(z_diag_ref * z_diag_ref, 1e-300)

    summary = pd.DataFrame(
        [
            {
                "section": "reference_profile_interpolation",
                "ref_D": ref_d,
                "ref_eta_summary": ref_eta_summary,
                "ref_eta_tree_nearest": float(tree_constants["ref_eta"]),
                "y2_raw_D9": y9,
                "y2_raw_D10": y10,
                "y2_raw_linear_interp_Dstar": y_linear,
                "y2_raw_log_interp_Dstar": y_log,
                "y2_raw_inserted_history_fine_Dstar": y_direct,
                "rel_inserted_vs_linear_interp": rel(y_direct, y_linear),
                "rel_inserted_vs_log_interp": rel(y_direct, y_log),
            },
            {
                "section": "branch_history_sensitivity",
                "single_point_reported_y2": float(
                    detail.loc[
                        (detail["source"] == "single_point_eigen_order")
                        & (detail["level"].astype(str).str.lower() == "fine"),
                        "y_eff_raw_2",
                    ].iloc[0]
                ),
                "single_point_closest_branch": int(
                    detail.loc[
                        (detail["source"] == "single_point_eigen_order")
                        & (detail["level"].astype(str).str.lower() == "fine"),
                        "canonical_target_branch",
                    ].iloc[0]
                ),
                "local_window_reported_y2": float(
                    detail.loc[
                        (detail["source"] == "local_9_9p6_10_history")
                        & (detail["level"].astype(str).str.lower() == "fine"),
                        "y_eff_raw_2",
                    ].iloc[0]
                ),
                "local_window_closest_branch": int(
                    detail.loc[
                        (detail["source"] == "local_9_9p6_10_history")
                        & (detail["level"].astype(str).str.lower() == "fine"),
                        "canonical_target_branch",
                    ].iloc[0]
                ),
                "canonical_inserted_reported_y2": y_direct,
                "max_abs_target_grid_rel_vs_fine": max_grid_rel,
                "mid_abs_target_grid_rel_vs_fine": mid_grid_rel,
                "coarse_abs_target_grid_rel_vs_fine": coarse_grid_rel,
            },
            {
                "section": "normalization_impact_if_adopted",
                "coeff2_kin_over_m2_ref": coeff_ref,
                "z_diag_ref": z_diag_ref,
                "C_tree_mumu_current": c_tree_current,
                "C_tree_mumu_direct_y2": c_tree_direct,
                "rel_C_tree_direct_vs_current": rel(c_tree_direct, c_tree_current),
                "A_tree_current": a_tree_current,
                "A_tree_direct_y2": a_tree_direct,
                "rel_A_tree_direct_vs_current": rel(a_tree_direct, a_tree_current),
                "A_star_current": a_star_current,
                "A_star_direct_y2": a_star_direct,
                "rel_A_star_direct_vs_current": rel(a_star_direct, a_star_current),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "current_release_reference",
                "description": "linear interpolation of canonical integer-D y_eff profile",
                "ref_D": ref_d,
                "ref_eta_summary": ref_eta_summary,
                "y2_raw_ref": y_linear,
                "status": "current release value; not parent-point audited",
            },
            {
                "candidate": "canonical_history_direct_reference",
                "description": "direct D_* solve inserted into the full D4--20 branch history",
                "ref_D": ref_d,
                "ref_eta_summary": ref_eta_summary,
                "y2_raw_ref": y_direct,
                "status": "parent-point audited candidate; requires map-impact gate before adoption",
            },
            {
                "candidate": "single_point_eigen_order",
                "description": "standalone D_* solve initialized by eigenvalue order",
                "ref_D": ref_d,
                "ref_eta_summary": ref_eta_summary,
                "y2_raw_ref": float(
                    detail.loc[
                        (detail["source"] == "single_point_eigen_order")
                        & (detail["level"].astype(str).str.lower() == "fine"),
                        "y_eff_raw_2",
                    ].iloc[0]
                ),
                "status": "diagnostic-only; wrong generation label near branch crossing",
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_refpoint_stability_summary.csv"
    detail_out = outdir / "hll_y2raw_refpoint_stability_detail.csv"
    constants_out = outdir / "hll_y2raw_refpoint_stability_constants.csv"
    meta_out = outdir / "hll_y2raw_refpoint_stability_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "canonical_profile_input": str(args.canonical_profile_input),
                "single_dstar_input": str(args.single_dstar_input),
                "local_window_input": str(args.local_window_input),
                "inserted_profile_input": str(args.inserted_profile_input),
                "summary_input": str(args.summary_input),
                "tree_constants_input": str(args.tree_constants_input),
                "ref_D": ref_d,
                "ref_eta_summary": ref_eta_summary,
                "notes": (
                    "The y2raw identity remains exact; this audit only separates "
                    "reference-point branch-continuation and interpolation floors."
                ),
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()
