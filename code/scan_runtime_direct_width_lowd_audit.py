#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
HLL_OUTDIR = ROOT / "output" / "hll_signal_strength"

FULL_TAG = "chain_mode_full_direct_D21E41"
RELEASE_MAP = "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_D21E41.csv"
COMBO_TAG = "chain_mode_cell_direct_runtime_release_combo_D21E41"


def _slice_delta_by_d(full_map: pd.DataFrame, cand_map: pd.DataFrame, label: str) -> pd.DataFrame:
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
                f"{label}_slice_p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                f"{label}_slice_max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                f"{label}_slice_acceptance_mismatch": float(sub["acceptance_mismatch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _probe_mu_by_d(map_df: pd.DataFrame, eta_probe: float) -> pd.DataFrame:
    sub = map_df[np.isclose(map_df["eta"], eta_probe)].copy()
    return sub[["D", "mu_mumu"]].rename(columns={"mu_mumu": "mu_mumu_probe"}).sort_values("D").reset_index(drop=True)


def _width_diag(kinetics, d_vals: np.ndarray, eta_probe: float, ref_d: float, ref_eta: float, t_coh: float) -> pd.DataFrame:
    cfg = kinetics._hll_match_config()
    rows = []
    for d_val in d_vals:
        c = kinetics.hll_wilson_matrix_uv_rge(D=float(d_val), eta=float(eta_probe), t_coh=t_coh, N_max=20)
        c_ref = kinetics.hll_wilson_matrix_uv_rge(D=float(ref_d), eta=float(ref_eta), t_coh=t_coh, N_max=20)
        c_diag = np.maximum(np.diag(c), cfg.floor)
        c_ref_diag = np.maximum(np.diag(c_ref), cfg.floor)
        partial = (c_diag / c_ref_diag) ** 2
        br = np.array([cfg.br_ee, cfg.br_mumu, cfg.br_tautau], dtype=float)
        raw_delta = float(np.sum(br * (partial - 1.0)))
        raw_ratio = 1.0 + float(cfg.width_scale) * raw_delta

        rho_target = float(cfg.width_power_base) + float(cfg.width_power_log_ratio_coeff) * float(
            np.log(max(float(partial[1]), cfg.floor) / max(float(partial[2]), cfg.floor))
        )
        rho_target = float(np.clip(rho_target, float(cfg.width_power_min), float(cfg.width_power_max)))
        lo = max(float(cfg.width_power_trigger_lo), 1.0 + cfg.floor)
        hi = max(float(cfg.width_power_trigger_hi), lo + cfg.floor)
        if hi <= lo * (1.0 + 1e-12):
            activation = 1.0 if raw_ratio > lo else 0.0
        else:
            activation = float(
                np.clip(
                    np.log(max(raw_ratio, 1.0 + cfg.floor) / lo) / np.log(hi / lo),
                    0.0,
                    1.0,
                )
            )
        rho = 1.0 - activation * (1.0 - rho_target)
        stabilized_partial = np.maximum(partial, cfg.floor) ** rho
        stabilized_ratio = 1.0 + float(cfg.width_scale) * float(np.sum(br * (stabilized_partial - 1.0)))
        rows.append(
            {
                "D": float(d_val),
                "raw_width_ratio": float(raw_ratio),
                "stabilized_width_ratio": float(max(stabilized_ratio, cfg.floor)),
                "width_compression_factor": float(max(stabilized_ratio, cfg.floor) / max(raw_ratio, cfg.floor)),
                "rho_target": rho_target,
                "activation": activation,
                "rho": rho,
                "partial_mu": float(partial[1]),
                "partial_tau": float(partial[2]),
                "log_partial_mu_over_tau": float(np.log(max(float(partial[1]), cfg.floor) / max(float(partial[2]), cfg.floor))),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    eta_probe = 0.2
    ref_d = 9.6
    ref_eta = 1.0
    t_coh = float(PAPER_BASELINE["t_coh"])
    d_vals = np.linspace(4.0, 20.0, 21)

    full_map = pd.read_csv(HLL_OUTDIR / f"hll_signal_strength_map_{FULL_TAG}.csv")
    release_map = pd.read_csv(HLL_OUTDIR / RELEASE_MAP)
    combo_map = pd.read_csv(HLL_OUTDIR / f"hll_signal_strength_map_{COMBO_TAG}.csv")

    release_kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    combo_kin = make_baseline_kinetics(
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

    release_mu = _probe_mu_by_d(release_map, eta_probe).rename(columns={"mu_mumu_probe": "mu_mumu_release_probe"})
    combo_mu = _probe_mu_by_d(combo_map, eta_probe).rename(columns={"mu_mumu_probe": "mu_mumu_combo_probe"})
    full_mu = _probe_mu_by_d(full_map, eta_probe).rename(columns={"mu_mumu_probe": "mu_mumu_full_probe"})

    release_slice = _slice_delta_by_d(full_map, release_map, "release")
    combo_slice = _slice_delta_by_d(full_map, combo_map, "combo")

    release_width = pd.DataFrame(
        {
            "D": d_vals,
            "release_width_ratio": [
                release_kin.hll_total_width_ratio_uv_rge(
                    D=float(d_val),
                    eta=float(eta_probe),
                    t_coh=t_coh,
                    ref_D=float(ref_d),
                    ref_eta=float(ref_eta),
                    N_max=20,
                )
                for d_val in d_vals
            ],
        }
    )
    combo_width = _width_diag(combo_kin, d_vals, eta_probe, ref_d, ref_eta, t_coh)

    audit_df = (
        release_width.merge(combo_width, on="D")
        .merge(full_mu, on="D")
        .merge(release_mu, on="D")
        .merge(combo_mu, on="D")
        .merge(release_slice, on="D")
        .merge(combo_slice, on="D")
    )

    low_d = audit_df[audit_df["D"] <= 6.4].copy()
    worst_combo = audit_df.loc[audit_df["combo_slice_max_abs_delta_mu_mumu"].idxmax()]
    summary = pd.DataFrame(
        [
            {
                "eta_probe": eta_probe,
                "max_combo_slice_abs_delta_mu_mumu": float(audit_df["combo_slice_max_abs_delta_mu_mumu"].max()),
                "worst_combo_slice_D": float(worst_combo["D"]),
                "worst_combo_slice_activation": float(worst_combo["activation"]),
                "worst_combo_slice_rho": float(worst_combo["rho"]),
                "worst_combo_slice_width_compression_factor": float(worst_combo["width_compression_factor"]),
                "worst_combo_slice_release_width_ratio": float(worst_combo["release_width_ratio"]),
                "worst_combo_slice_stabilized_width_ratio": float(worst_combo["stabilized_width_ratio"]),
                "low_d_min_width_compression_factor": float(low_d["width_compression_factor"].min()),
                "low_d_max_activation": float(low_d["activation"].max()),
                "low_d_max_combo_slice_abs_delta_mu_mumu": float(low_d["combo_slice_max_abs_delta_mu_mumu"].max()),
                "low_d_corr_activation_vs_combo_slice_abs_delta": float(
                    low_d["activation"].corr(low_d["combo_slice_max_abs_delta_mu_mumu"])
                ),
            }
        ]
    )

    out_map = OUTDIR / "runtime_direct_width_lowd_audit_map.csv"
    out_summary = OUTDIR / "runtime_direct_width_lowd_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_width_lowd_audit.png"
    out_meta = OUTDIR / "runtime_direct_width_lowd_audit_run_meta.json"

    audit_df.to_csv(out_map, index=False)
    summary.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(audit_df["D"], audit_df["release_width_ratio"], marker="o", label="release width ratio")
    axes[0].plot(audit_df["D"], audit_df["stabilized_width_ratio"], marker="s", label="combo width ratio")
    axes[0].plot(audit_df["D"], audit_df["width_compression_factor"], marker="^", label="compression factor")
    axes[0].set_ylabel("width")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(audit_df["D"], audit_df["mu_mumu_full_probe"], marker="o", label="full-direct mu_mumu")
    axes[1].plot(audit_df["D"], audit_df["mu_mumu_release_probe"], marker="s", label="release mu_mumu")
    axes[1].plot(audit_df["D"], audit_df["mu_mumu_combo_probe"], marker="^", label="combo mu_mumu")
    axes[1].plot(audit_df["D"], audit_df["activation"], marker="d", label="activation")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("probe mu / activation")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    meta = {
        "full_tag": FULL_TAG,
        "release_map": RELEASE_MAP,
        "combo_tag": COMBO_TAG,
        "eta_probe": eta_probe,
        "ref_d": ref_d,
        "ref_eta": ref_eta,
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
