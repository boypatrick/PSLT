#!/usr/bin/env python3
"""
Compute anchor-invariant channel-ratio diagnostics from UV+LL-RG Wilson maps.

Ratios:
  R_e/mu   = |C_ir_ee / C_ir_mumu|^2
  R_tau/mu = |C_ir_tautau / C_ir_mumu|^2

Scopes:
  - all_grid
  - chi2_accept (chi2_uv_rge <= threshold)

Outputs:
  - output/hll_reference_anchor/reference_anchor_invariant_ratios.csv
  - paper/reference_anchor_invariant_ratios.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_reference_anchor"
PAPER_DIR = ROOT / "paper"


def summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor-invariant Wilson-ratio summary.")
    ap.add_argument("--in-csv", type=Path, default=ROOT / "output" / "hll_uv_matching" / "hll_uv_to_eft_map.csv")
    ap.add_argument("--chi2-threshold", type=float, default=4.0)
    args = ap.parse_args()

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    floor = 1e-30
    c_ee = np.maximum(df["C_ir_ee"].to_numpy(dtype=float), floor)
    c_mu = np.maximum(df["C_ir_mumu"].to_numpy(dtype=float), floor)
    c_tau = np.maximum(df["C_ir_tautau"].to_numpy(dtype=float), floor)
    chi2 = df["chi2_uv_rge"].to_numpy(dtype=float)

    r_e_mu = (c_ee / c_mu) ** 2
    r_tau_mu = (c_tau / c_mu) ** 2
    accept = chi2 <= float(args.chi2_threshold)

    rows: list[dict[str, float | str]] = []
    for scope, mask in [
        ("all_grid", np.ones_like(accept, dtype=bool)),
        ("chi2_accept", accept),
    ]:
        if not np.any(mask):
            continue
        s1 = summarize(r_e_mu[mask])
        s2 = summarize(r_tau_mu[mask])
        rows.append(
            {
                "scope": scope,
                "n_points": int(np.sum(mask)),
                "chi2_threshold": float(args.chi2_threshold),
                "R_e_over_mu_min": s1["min"],
                "R_e_over_mu_p10": s1["p10"],
                "R_e_over_mu_p50": s1["p50"],
                "R_e_over_mu_p90": s1["p90"],
                "R_e_over_mu_max": s1["max"],
                "R_e_over_mu_mean": s1["mean"],
                "R_e_over_mu_std": s1["std"],
                "R_tau_over_mu_min": s2["min"],
                "R_tau_over_mu_p10": s2["p10"],
                "R_tau_over_mu_p50": s2["p50"],
                "R_tau_over_mu_p90": s2["p90"],
                "R_tau_over_mu_max": s2["max"],
                "R_tau_over_mu_mean": s2["mean"],
                "R_tau_over_mu_std": s2["std"],
            }
        )

    out_csv = OUTDIR / "reference_anchor_invariant_ratios.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paper_csv = PAPER_DIR / "reference_anchor_invariant_ratios.csv"
    paper_csv.write_bytes(out_csv.read_bytes())

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
