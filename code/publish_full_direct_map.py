#!/usr/bin/env python3
"""
Publish-mode runner for full_direct map evidence.

This release mode produces eight reproducible artifacts:
  1) Main-map baseline in full_direct mode (D60 x E60)
  2) Small-surface complete localized-direct audit (D21 x E41)
  3) Large-surface spot-check localized-direct audit (D60 x E21)
  4) Small-surface chain parity audit (full_direct vs cell_direct_runtime, D21 x E41)
  5) Small-surface chain parity audit (full_direct vs cell_direct_runtime_release, D21 x E41)
  6) Large-surface chain parity audit (full_direct vs cell_direct_runtime, D60 x E21)
  7) Large-surface chain parity audit (full_direct vs cell_direct_runtime_release, D60 x E21)
  8) Large-surface chain parity audit (full_direct vs cell_direct_runtime_extreme, D60 x E21)

And aggregates them into one reviewer-facing summary table:
  - output/kinetic_action_chain/full_direct_map_release_summary.csv
  - paper/full_direct_map_release_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_KIN = ROOT / "output" / "kinetic_action_chain"
OUT_HLL = ROOT / "output" / "hll_signal_strength"
PAPER = ROOT / "paper"

SCAN_HLL = ROOT / "code" / "scan_hll_signal_strengths.py"
SCAN_LOC_DIRECT = ROOT / "code" / "scan_localized_direct_surface_bias.py"
SCAN_CHAIN_AUDIT = ROOT / "code" / "scan_chain_mode_cell_direct_audit.py"


def run_cmd(name: str, cmd: List[str], expected: Path | None, force: bool) -> None:
    if expected is not None and expected.exists() and not force:
        print(f"[skip] {name}: {expected} exists")
        return
    print(f"[run] {name}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_hll_mumu_summary(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df[df["channel"] == "mumu"].iloc[0]
    return {
        "f_chi2_le_4": float(row["f_chi2_le_4"]),
        "best_chi2": float(row["best_chi2"]),
        "best_D": float(row["best_D"]),
        "best_eta": float(row["best_eta"]),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish full_direct map release summary.")
    ap.add_argument("--force", action="store_true", help="Recompute all steps even if outputs already exist.")
    ap.add_argument("--main-chain-mode", choices=["full_direct", "full_direct_runtime"], default="full_direct")
    ap.add_argument("--runtime-direct-force", action="store_true", help="When main-chain-mode=full_direct_runtime, force direct profile rebuild.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT_KIN.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    tag_main = "full_direct_map_release"
    tag_full_small = "full_direct_map_full_release_D21E41"
    tag_cell_small = "full_direct_map_cell_direct_runtime_release_D21E41"
    tag_cell_small_release = "full_direct_map_cell_direct_runtime_release_candidate_D21E41"
    tag_full_large = "full_direct_map_full_release_D60E21"
    tag_cell_large_runtime = "full_direct_map_cell_direct_runtime_release_D60E21"
    tag_cell_large_release = "full_direct_map_cell_direct_runtime_release_candidate_D60E21"
    tag_cell_large_extreme = "full_direct_map_cell_direct_runtime_extreme_release_D60E21"

    # 1) Main-map full_direct baseline.
    main_summary = OUT_HLL / f"hll_signal_strength_summary_{tag_main}.csv"
    run_cmd(
        name="scan_hll_signal_strengths_full_direct_main",
        cmd=[
            sys.executable,
            str(SCAN_HLL),
            "--chain-mode",
            str(args.main_chain_mode),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "60",
            "--tag",
            tag_main,
        ]
        + (["--runtime-direct-force"] if (str(args.main_chain_mode) == "full_direct_runtime" and bool(args.runtime_direct_force)) else []),
        expected=main_summary,
        force=bool(args.force),
    )

    # 2) Small-surface complete direct audit.
    small_summary = OUT_KIN / "localized_direct_surface_summary_Dgrid21_Egrid41.csv"
    run_cmd(
        name="localized_direct_surface_bias_small_complete",
        cmd=[
            sys.executable,
            str(SCAN_LOC_DIRECT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
        ],
        expected=small_summary,
        force=bool(args.force),
    )

    # 3) Large-surface spot-check direct audit.
    large_summary = OUT_KIN / "localized_direct_surface_summary_Dgrid60_Egrid21.csv"
    run_cmd(
        name="localized_direct_surface_bias_large_spotcheck",
        cmd=[
            sys.executable,
            str(SCAN_LOC_DIRECT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
        ],
        expected=large_summary,
        force=bool(args.force),
    )

    # 4) Small-surface chain-mode parity audit (full_direct vs cell_direct_runtime).
    parity_small_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid21_Egrid41.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_small_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
            "--full-direct-tag",
            tag_full_small,
            "--cell-direct-tag",
            tag_cell_small,
        ],
        expected=parity_small_summary,
        force=bool(args.force),
    )

    # 5) Small-surface chain-mode parity audit (full_direct vs cell_direct_runtime_release).
    parity_small_release_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_release_small_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
            "--full-direct-tag",
            tag_full_small,
            "--cell-direct-tag",
            tag_cell_small_release,
            "--cell-chain-mode",
            "cell_direct_runtime_release",
        ],
        expected=parity_small_release_summary,
        force=bool(args.force),
    )

    # 6) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime).
    parity_large_runtime_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_runtime,
            "--cell-chain-mode",
            "cell_direct_runtime",
        ],
        expected=parity_large_runtime_summary,
        force=bool(args.force),
    )

    # 7) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime_release).
    parity_large_release_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_release_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_release,
            "--cell-chain-mode",
            "cell_direct_runtime_release",
        ],
        expected=parity_large_release_summary,
        force=bool(args.force),
    )

    # 8) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime_extreme).
    parity_large_extreme_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_extreme.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_extreme_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_extreme,
            "--cell-chain-mode",
            "cell_direct_runtime_extreme",
        ],
        expected=parity_large_extreme_summary,
        force=bool(args.force),
    )

    main_mumu = read_hll_mumu_summary(main_summary)
    small = pd.read_csv(small_summary).iloc[0].to_dict()
    large = pd.read_csv(large_summary).iloc[0].to_dict()
    parity_small = pd.read_csv(parity_small_summary).iloc[0].to_dict()
    parity_small_release = pd.read_csv(parity_small_release_summary).iloc[0].to_dict()
    parity_large_runtime = pd.read_csv(parity_large_runtime_summary).iloc[0].to_dict()
    parity_large_release = pd.read_csv(parity_large_release_summary).iloc[0].to_dict()
    parity_large_extreme = pd.read_csv(parity_large_extreme_summary).iloc[0].to_dict()

    rows: List[Dict[str, object]] = [
        {
            "scenario": "main_map_full_direct_baseline",
            "grid": "D60xE60",
            "n_points": int(60 * 60),
            "f_chi2_mumu_le_4": float(main_mumu["f_chi2_le_4"]),
            "best_chi2_mumu": float(main_mumu["best_chi2"]),
            "best_D": float(main_mumu["best_D"]),
            "best_eta": float(main_mumu["best_eta"]),
            "frac_winner_mismatch": "",
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": "",
            "delta_f_chi2_mumu_le_4": "",
            "source": str(main_summary.relative_to(ROOT)),
        },
        {
            "scenario": "small_surface_complete_direct_bias",
            "grid": "D21xE41",
            "n_points": int(small["n_points"]),
            "f_chi2_mumu_le_4": float(small["f_chi2_mumu_le_4_direct"]),
            "best_chi2_mumu": float(small["best_chi2_mumu_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(small["frac_winner_mismatch"]),
            "max_abs_delta_R3": float(small["max_abs_delta_R3"]),
            "max_abs_delta_mu_mumu": float(small["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(small["delta_f_chi2_mumu_le_4"]),
            "source": str(small_summary.relative_to(ROOT)),
        },
        {
            "scenario": "large_surface_spotcheck_direct_bias",
            "grid": "D60xE21",
            "n_points": int(large["n_points"]),
            "f_chi2_mumu_le_4": float(large["f_chi2_mumu_le_4_direct"]),
            "best_chi2_mumu": float(large["best_chi2_mumu_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(large["frac_winner_mismatch"]),
            "max_abs_delta_R3": float(large["max_abs_delta_R3"]),
            "max_abs_delta_mu_mumu": float(large["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(large["delta_f_chi2_mumu_le_4"]),
            "source": str(large_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_parity_full_direct_vs_cell_direct_runtime",
            "grid": "D21xE41",
            "n_points": int(parity_small["n_points"]),
            "f_chi2_mumu_le_4": float(parity_small["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_small["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_small["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_small["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_small["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_small_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime",
            "grid": "D60xE21",
            "n_points": int(parity_large_runtime["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_runtime["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_runtime["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_runtime["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_runtime["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_runtime["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_runtime_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_parity_full_direct_vs_cell_direct_runtime_release",
            "grid": "D21xE41",
            "n_points": int(parity_small_release["n_points"]),
            "f_chi2_mumu_le_4": float(parity_small_release["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_small_release["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_small_release["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_small_release["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_small_release["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_small_release_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_release",
            "grid": "D60xE21",
            "n_points": int(parity_large_release["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_release["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_release["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_release["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_release["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_release["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_release_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_extreme",
            "grid": "D60xE21",
            "n_points": int(parity_large_extreme["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_extreme["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_extreme["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_extreme["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_extreme["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_extreme["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_extreme_summary.relative_to(ROOT)),
        },
    ]

    out_csv = OUT_KIN / "full_direct_map_release_summary.csv"
    paper_csv = PAPER / out_csv.name
    out_json = OUT_KIN / "full_direct_map_release_meta.json"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paper_csv.write_text(out_csv.read_text())
    out_json.write_text(
        json.dumps(
            {
                "main_tag": tag_main,
                "main_chain_mode": str(args.main_chain_mode),
                "runtime_direct_force": bool(args.runtime_direct_force),
                "small_surface": str(small_summary.relative_to(ROOT)),
                "large_surface": str(large_summary.relative_to(ROOT)),
                "chain_parity_small_runtime": str(parity_small_summary.relative_to(ROOT)),
                "chain_parity_small_release": str(parity_small_release_summary.relative_to(ROOT)),
                "chain_parity_large_runtime": str(parity_large_runtime_summary.relative_to(ROOT)),
                "chain_parity_large_release": str(parity_large_release_summary.relative_to(ROOT)),
                "chain_parity_large_extreme": str(parity_large_extreme_summary.relative_to(ROOT)),
                "summary_csv": str(out_csv.relative_to(ROOT)),
            },
            indent=2,
        )
    )

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {out_json}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
