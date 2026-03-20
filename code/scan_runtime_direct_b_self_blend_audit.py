#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
HLL_OUTDIR = ROOT / "output" / "hll_signal_strength"

FULL_TAG = "chain_mode_full_direct_D21E41"
COMBO_TAG = "chain_mode_cell_direct_runtime_release_combo_D21E41"


def _slice_stats(full_map: pd.DataFrame, combo_map: pd.DataFrame) -> pd.DataFrame:
    merged = full_map.merge(combo_map, on=["D", "eta"], suffixes=("_full", "_combo"))
    merged["abs_delta_mu_mumu"] = np.abs(merged["mu_mumu_combo"] - merged["mu_mumu_full"])
    merged["acceptance_mismatch"] = (
        (merged["chi2_mumu_full"] <= 4.0) ^ (merged["chi2_mumu_combo"] <= 4.0)
    ).astype(float)

    rows = []
    for d_val, sub in merged.groupby("D", sort=True):
        rows.append(
            {
                "D": float(d_val),
                "slice_p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                "slice_max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                "slice_acceptance_mismatch": float(sub["acceptance_mismatch"].mean()),
                "slice_mu_mumu_combo_p95": float(np.percentile(sub["mu_mumu_combo"], 95.0)),
                "slice_mu_mumu_full_p95": float(np.percentile(sub["mu_mumu_full"], 95.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    full_map = pd.read_csv(HLL_OUTDIR / f"hll_signal_strength_map_{FULL_TAG}.csv")
    combo_map = pd.read_csv(HLL_OUTDIR / f"hll_signal_strength_map_{COMBO_TAG}.csv")

    kinetics = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_combo",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )

    d_vals = np.linspace(4.0, 20.0, 21)
    rows = []
    for d_val in d_vals:
        direct = kinetics._runtime_direct_b_operator_inputs(float(d_val))
        direct_b123 = np.maximum(np.asarray(direct["b123"], dtype=float), kinetics.params.b_overlap_floor)
        operator_b123 = kinetics._b_eft_norm_vector(float(d_val))
        metric = float(
            np.max(
                np.maximum(
                    np.log(
                        np.maximum(operator_b123[:2], kinetics.params.b_overlap_floor)
                        / np.maximum(direct_b123[:2], kinetics.params.b_overlap_floor)
                    ),
                    0.0,
                )
            )
        )
        beta = float(kinetics._runtime_direct_b_self_blend_weight(operator_b123, direct_b123))
        rows.append(
            {
                "D": float(d_val),
                "blend_metric": metric,
                "blend_beta": beta,
                "operator_b1": float(operator_b123[0]),
                "operator_b2": float(operator_b123[1]),
                "direct_b1": float(direct_b123[0]),
                "direct_b2": float(direct_b123[1]),
            }
        )

    blend_df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    slice_df = _slice_stats(full_map, combo_map)
    audit_df = blend_df.merge(slice_df, on="D", how="left")

    worst_row = audit_df.loc[audit_df["slice_max_abs_delta_mu_mumu"].idxmax()]
    summary = pd.DataFrame(
        [
            {
                "combo_tag": COMBO_TAG,
                "max_blend_metric": float(audit_df["blend_metric"].max()),
                "max_blend_beta": float(audit_df["blend_beta"].max()),
                "num_beta_gt0": int((audit_df["blend_beta"] > 0.0).sum()),
                "current_metric_lo": float(kinetics.params.runtime_direct_b_self_blend_metric_lo),
                "current_metric_hi": float(kinetics.params.runtime_direct_b_self_blend_metric_hi),
                "current_beta_max": float(kinetics.params.runtime_direct_b_self_blend_max),
                "worst_slice_D": float(worst_row["D"]),
                "worst_slice_max_abs_delta_mu_mumu": float(worst_row["slice_max_abs_delta_mu_mumu"]),
                "worst_slice_p95_abs_delta_mu_mumu": float(worst_row["slice_p95_abs_delta_mu_mumu"]),
                "worst_slice_acceptance_mismatch": float(worst_row["slice_acceptance_mismatch"]),
                "worst_slice_blend_metric": float(worst_row["blend_metric"]),
                "worst_slice_blend_beta": float(worst_row["blend_beta"]),
                "corr_metric_vs_slice_max_abs_delta": float(
                    audit_df["blend_metric"].corr(audit_df["slice_max_abs_delta_mu_mumu"])
                ),
                "corr_metric_vs_slice_acceptance_mismatch": float(
                    audit_df["blend_metric"].corr(audit_df["slice_acceptance_mismatch"])
                ),
            }
        ]
    )

    out_map = OUTDIR / "runtime_direct_b_self_blend_audit_map.csv"
    out_summary = OUTDIR / "runtime_direct_b_self_blend_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_b_self_blend_audit.png"
    out_meta = OUTDIR / "runtime_direct_b_self_blend_audit_run_meta.json"

    audit_df.to_csv(out_map, index=False)
    summary.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(audit_df["D"], audit_df["blend_metric"], marker="o", label="blend metric")
    axes[0].plot(audit_df["D"], audit_df["blend_beta"], marker="s", label="blend beta")
    axes[0].axhline(float(kinetics.params.runtime_direct_b_self_blend_metric_lo), color="tab:red", ls="--", lw=1.0, label="metric lo")
    axes[0].axhline(float(kinetics.params.runtime_direct_b_self_blend_metric_hi), color="tab:orange", ls="--", lw=1.0, label="metric hi")
    axes[0].set_ylabel("metric / beta")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(audit_df["D"], audit_df["slice_p95_abs_delta_mu_mumu"], marker="o", label="slice p95 |Δμ_μμ|")
    axes[1].plot(audit_df["D"], audit_df["slice_max_abs_delta_mu_mumu"], marker="s", label="slice max |Δμ_μμ|")
    axes[1].plot(audit_df["D"], audit_df["slice_acceptance_mismatch"], marker="^", label="slice mismatch")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("gate drift")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    meta = {
        "full_tag": FULL_TAG,
        "combo_tag": COMBO_TAG,
        "d_num": int(len(d_vals)),
        "current_metric_lo": float(kinetics.params.runtime_direct_b_self_blend_metric_lo),
        "current_metric_hi": float(kinetics.params.runtime_direct_b_self_blend_metric_hi),
        "current_beta_max": float(kinetics.params.runtime_direct_b_self_blend_max),
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for src in [out_map, out_summary, out_fig, out_meta]:
        dst = PAPER_DIR / src.name
        if src.suffix == ".png":
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text(src.read_text())

    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
