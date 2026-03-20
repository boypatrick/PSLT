#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publish_full_direct_map import _extract_b_metrics
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SIGDIR = ROOT / "output" / "hll_signal_strength"


def _slice_stats(full_map: pd.DataFrame, cand_map: pd.DataFrame, prefix: str) -> pd.DataFrame:
    merged = full_map.merge(cand_map, on=["D", "eta"], suffixes=("_full", "_cand"))
    merged["abs_delta_mu_mumu"] = np.abs(merged["mu_mumu_cand"] - merged["mu_mumu_full"])
    merged["acceptance_mismatch"] = (
        (merged["chi2_mumu_full"] <= 4.0) ^ (merged["chi2_mumu_cand"] <= 4.0)
    ).astype(float)
    rows = []
    for d_val, sub in merged.groupby("D", sort=True):
        rows.append(
            {
                "D": float(d_val),
                f"{prefix}_slice_max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                f"{prefix}_slice_p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                f"{prefix}_slice_acceptance_mismatch": float(sub["acceptance_mismatch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    full_map = pd.read_csv(SIGDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv")
    release_map = pd.read_csv(SIGDIR / "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_D21E41.csv")
    band_map = pd.read_csv(SIGDIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_bandcombo_D21E41.csv")

    common = dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **common)
    kin_release = make_baseline_kinetics(chain_mode="cell_direct_runtime_release", **common)
    kin_band = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_bandcombo", **common)

    eta_probe = 0.2
    ref_d = 9.6
    ref_eta = 1.0
    d_focus = [7.2, 8.0]
    rows = []
    for d_val in d_focus:
        metrics_full = _extract_b_metrics(kin_full, d_val, eta_probe, ref_d=ref_d, ref_eta=ref_eta)
        metrics_release = _extract_b_metrics(kin_release, d_val, eta_probe, ref_d=ref_d, ref_eta=ref_eta)
        metrics_band = _extract_b_metrics(kin_band, d_val, eta_probe, ref_d=ref_d, ref_eta=ref_eta)
        rows.append(
            {
                "D": float(d_val),
                "eta_probe": float(eta_probe),
                "full_B2": float(metrics_full["B2"]),
                "release_B2": float(metrics_release["B2"]),
                "band_B2": float(metrics_band["B2"]),
                "delta_B2_band_vs_release": float(metrics_band["B2"] - metrics_release["B2"]),
                "full_yraw2": float(metrics_full["yraw2"]),
                "release_yraw2": float(metrics_release["yraw2"]),
                "band_yraw2": float(metrics_band["yraw2"]),
                "full_m2_2": float(metrics_full["m2_2"]),
                "release_m2_2": float(metrics_release["m2_2"]),
                "band_m2_2": float(metrics_band["m2_2"]),
                "full_width_ratio": float(metrics_full["width_ratio"]),
                "release_width_ratio": float(metrics_release["width_ratio"]),
                "band_width_ratio": float(metrics_band["width_ratio"]),
                "delta_width_ratio_band_vs_release": float(metrics_band["width_ratio"] - metrics_release["width_ratio"]),
                "full_mu_mumu": float(metrics_full["mu_mumu_model"]),
                "release_mu_mumu": float(metrics_release["mu_mumu_model"]),
                "band_mu_mumu": float(metrics_band["mu_mumu_model"]),
                "delta_mu_band_vs_release": float(metrics_band["mu_mumu_model"] - metrics_release["mu_mumu_model"]),
            }
        )

    focus_df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)

    release_slice = _slice_stats(full_map, release_map, "release")
    band_slice = _slice_stats(full_map, band_map, "band")
    slice_df = release_slice.merge(band_slice, on="D", how="outer").sort_values("D").reset_index(drop=True)
    band_focus = slice_df[slice_df["D"].between(7.2, 8.0)].copy()

    summary = pd.DataFrame(
        [
            {
                "eta_probe": float(eta_probe),
                "max_abs_delta_B2_band_vs_release_midband": float(np.max(np.abs(focus_df["delta_B2_band_vs_release"]))),
                "max_abs_delta_width_ratio_band_vs_release_midband": float(
                    np.max(np.abs(focus_df["delta_width_ratio_band_vs_release"]))
                ),
                "max_abs_delta_mu_band_vs_release_midband": float(np.max(np.abs(focus_df["delta_mu_band_vs_release"]))),
                "release_midband_max_slice_mismatch": float(band_focus["release_slice_acceptance_mismatch"].max()),
                "band_midband_max_slice_mismatch": float(band_focus["band_slice_acceptance_mismatch"].max()),
                "release_midband_max_slice_abs_delta": float(band_focus["release_slice_max_abs_delta_mu_mumu"].max()),
                "band_midband_max_slice_abs_delta": float(band_focus["band_slice_max_abs_delta_mu_mumu"].max()),
            }
        ]
    )

    out_focus = OUTDIR / "runtime_direct_midband_component_audit_focus.csv"
    out_slices = OUTDIR / "runtime_direct_midband_component_audit_slices.csv"
    out_summary = OUTDIR / "runtime_direct_midband_component_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_midband_component_audit.png"
    out_meta = OUTDIR / "runtime_direct_midband_component_audit_run_meta.json"

    focus_df.to_csv(out_focus, index=False)
    slice_df.to_csv(out_slices, index=False)
    summary.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(focus_df["D"], focus_df["release_B2"], marker="o", label="release B2")
    axes[0].plot(focus_df["D"], focus_df["band_B2"], marker="s", label="bandcombo B2")
    axes[0].plot(focus_df["D"], focus_df["release_mu_mumu"], marker="^", label="release mu_mumu")
    axes[0].plot(focus_df["D"], focus_df["band_mu_mumu"], marker="d", label="bandcombo mu_mumu")
    axes[0].set_ylabel("focused midband")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    mid = slice_df[slice_df["D"].between(6.4, 8.8)].copy()
    axes[1].plot(mid["D"], mid["release_slice_acceptance_mismatch"], marker="o", label="release mismatch")
    axes[1].plot(mid["D"], mid["band_slice_acceptance_mismatch"], marker="s", label="bandcombo mismatch")
    axes[1].plot(mid["D"], mid["release_slice_max_abs_delta_mu_mumu"], marker="^", label="release max |Δμ_μμ|")
    axes[1].plot(mid["D"], mid["band_slice_max_abs_delta_mu_mumu"], marker="d", label="bandcombo max |Δμ_μμ|")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("slice drift")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    meta = {
        "eta_probe": float(eta_probe),
        "ref_d": float(ref_d),
        "ref_eta": float(ref_eta),
        "statement": "In the D~7.2-8.0 mid-band, the band-limited branch changes B2 strongly but leaves width_ratio and mu_mumu essentially unchanged, so the residual acceptance mismatch is not controlled by the current B self-blend path.",
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for src in [out_focus, out_slices, out_summary, out_fig, out_meta]:
        dst = PAPER_DIR / src.name
        if src.suffix == ".png":
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text(src.read_text())

    print(f"[saved] {out_focus}")
    print(f"[saved] {out_slices}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
