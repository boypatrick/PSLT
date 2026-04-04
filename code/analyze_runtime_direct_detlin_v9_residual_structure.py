#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
DETAIL_PATH = OUTDIR / "runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64_v9_sobol3_bestobs.csv"
SUMMARY_PATH = OUTDIR / "runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64_v9_sobol3_bestobs.csv"
STEM = "runtime_direct_detlin_v9_residual_structure"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(DETAIL_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    d60 = detail.loc[detail["case"] == "D60E21_release"].copy()
    d60["eval_D"] = d60["eval_D"].astype(float)
    d60["eta"] = d60["eta"].astype(float)
    d60["abs_delta_mu_mumu"] = d60["abs_delta_mu_mumu"].astype(float)
    d60["acceptance_mismatch"] = d60["acceptance_mismatch"].astype(float)

    slice_summary = (
        d60.groupby("eval_D", sort=True)
        .agg(
            n_points=("abs_delta_mu_mumu", "size"),
            mean_abs_delta_mu_mumu=("abs_delta_mu_mumu", "mean"),
            p95_abs_delta_mu_mumu=("abs_delta_mu_mumu", lambda s: s.quantile(0.95, interpolation="higher")),
            max_abs_delta_mu_mumu=("abs_delta_mu_mumu", "max"),
            acceptance_mismatch=("acceptance_mismatch", "mean"),
        )
        .reset_index()
        .sort_values("eval_D")
        .reset_index(drop=True)
    )

    total_p95 = float(
        summary.loc[summary["case"] == "D60E21_release", "p95_abs_delta_mu_mumu"].iloc[0]
    )
    slice_summary["share_of_total_p95"] = slice_summary["p95_abs_delta_mu_mumu"] / total_p95
    dominant = slice_summary.sort_values(
        ["p95_abs_delta_mu_mumu", "max_abs_delta_mu_mumu"], ascending=False
    ).iloc[0]
    dominant_d = float(dominant["eval_D"])
    dominant_eta = (
        d60.loc[d60["eval_D"] == dominant_d, ["eta", "abs_delta_mu_mumu"]]
        .sort_values("eta")
        .reset_index(drop=True)
    )

    decision = {
        "source_detail": str(DETAIL_PATH),
        "source_summary": str(SUMMARY_PATH),
        "dominant_eval_D": dominant_d,
        "dominant_slice_p95_abs_delta_mu_mumu": float(dominant["p95_abs_delta_mu_mumu"]),
        "dominant_slice_max_abs_delta_mu_mumu": float(dominant["max_abs_delta_mu_mumu"]),
        "dominant_slice_share_of_total_p95": float(dominant["share_of_total_p95"]),
        "secondary_eval_D": float(slice_summary.sort_values("p95_abs_delta_mu_mumu", ascending=False).iloc[1]["eval_D"]),
        "secondary_slice_p95_abs_delta_mu_mumu": float(
            slice_summary.sort_values("p95_abs_delta_mu_mumu", ascending=False).iloc[1]["p95_abs_delta_mu_mumu"]
        ),
        "all_zero_acceptance_mismatch": bool((slice_summary["acceptance_mismatch"] == 0.0).all()),
        "note": "Dominant remaining broader-grid strict all-direct residual is concentrated on the D60 release slice near eval_D≈6.44; D≈4.81 is the next largest but materially smaller.",
    }

    slice_path = OUTDIR / f"{STEM}_slices.csv"
    eta_path = OUTDIR / f"{STEM}_eta_profile.csv"
    decision_path = OUTDIR / f"{STEM}_decision.json"
    png_path = OUTDIR / f"{STEM}.png"

    slice_summary.to_csv(slice_path, index=False)
    dominant_eta.to_csv(eta_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.4, 7.4), constrained_layout=True)

    ax0.plot(
        slice_summary["eval_D"],
        slice_summary["p95_abs_delta_mu_mumu"],
        marker="o",
        color="#e15759",
        label="slice p95",
    )
    ax0.plot(
        slice_summary["eval_D"],
        slice_summary["mean_abs_delta_mu_mumu"],
        marker="o",
        color="#4e79a7",
        label="slice mean",
    )
    ax0.plot(
        slice_summary["eval_D"],
        slice_summary["max_abs_delta_mu_mumu"],
        marker="o",
        color="#59a14f",
        label="slice max",
    )
    ax0.axvline(dominant_d, color="#f28e2b", ls="--", alpha=0.5)
    ax0.set_title("Strict All-Direct v9 Residual Structure (D60 x E21 release)")
    ax0.set_xlabel("eval D")
    ax0.set_ylabel("|Δμ_μμ|")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False)

    ax1.plot(
        dominant_eta["eta"],
        dominant_eta["abs_delta_mu_mumu"],
        marker="o",
        color="#f28e2b",
    )
    ax1.set_title(f"Dominant slice eta profile at eval D={dominant_d:.6f}")
    ax1.set_xlabel("eta")
    ax1.set_ylabel("|Δμ_μμ|")
    ax1.grid(True, alpha=0.25)

    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    for path in (slice_path, eta_path, decision_path, png_path):
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(slice_summary.to_string(index=False))
    print(f"\nWrote slices:   {slice_path}")
    print(f"Wrote eta:      {eta_path}")
    print(f"Wrote decision: {decision_path}")
    print(f"Wrote figure:   {png_path}")


if __name__ == "__main__":
    main()
