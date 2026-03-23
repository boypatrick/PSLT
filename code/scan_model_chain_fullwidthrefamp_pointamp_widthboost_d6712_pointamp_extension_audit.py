#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
DETAIL_PATH = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d60_reviewer_hotspot_audit_detail.csv"

FOCUS_D = np.array(
    [
        5.898305084745763,
        6.4406779661016955,
        6.711864406779661,
        6.983050847457627,
    ],
    dtype=float,
)
CENTER_DS = (6.68, 6.71, 6.74)
BETAS = (0.20, 0.35, 0.50, 0.65)
SIGMAS = (0.03, 0.05, 0.08, 0.10)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _geom_blend(cur: float, base: float, beta: float, floor: float = 1e-30) -> float:
    return float(
        np.exp(
            (1.0 - float(beta)) * np.log(max(float(cur), floor))
            + float(beta) * np.log(max(float(base), floor))
        )
    )


def _localized_beta(beta_peak: float, D: float, center: float, sigma: float) -> float:
    return float(beta_peak * np.exp(-0.5 * ((float(D) - float(center)) / float(sigma)) ** 2))


def _effective_ref_amp(row: pd.Series) -> float:
    partial_eff = float(row["cur_mu"]) * float(row["cur_width_ratio"])
    return float(row["cur_amp"] / np.sqrt(max(partial_eff, 1e-30)))


def _build_detail(base_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in base_detail.itertuples(index=False):
        if float(row.D) not in set(float(x) for x in FOCUS_D):
            continue
        eff_ref = _effective_ref_amp(pd.Series(row._asdict()))
        for center_D in CENTER_DS:
            for sigma_D in SIGMAS:
                for beta_peak in BETAS:
                    beta_eff = _localized_beta(float(beta_peak), float(row.D), float(center_D), float(sigma_D))
                    amp_cf = _geom_blend(float(row.cur_amp), float(row.base_amp), beta_eff)
                    partial_cf = float((amp_cf / max(eff_ref, 1e-30)) ** 2)
                    mu_cf = float(partial_cf / max(float(row.cur_width_ratio), 1e-30))
                    base_mu = float(row.base_mu)
                    base_chi2 = float(row.base_chi2)
                    chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
                    rows.append(
                        {
                            "center_D": float(center_D),
                            "sigma_D": float(sigma_D),
                            "beta_peak": float(beta_peak),
                            "beta_eff": float(beta_eff),
                            "D": float(row.D),
                            "eta": float(row.eta),
                            "base_mu": base_mu,
                            "cur_mu": float(row.cur_mu),
                            "mu_cf": mu_cf,
                            "delta_mu_cf": float(mu_cf - base_mu),
                            "delta_log_mu_cf": _safe_log_ratio(mu_cf, base_mu),
                            "acceptance_mismatch_cf": float((base_chi2 <= 4.0) ^ (chi2_cf <= 4.0)),
                            "effective_ref_vs_raw_ratio": float(eff_ref / max(float(row.cur_amp_ref), 1e-30)),
                        }
                    )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (center_D, sigma_D, beta_peak), sub in detail.groupby(["center_D", "sigma_D", "beta_peak"], sort=True):
        by_d = {
            float(d_val): float(np.max(np.abs(grp["delta_mu_cf"])))
            for d_val, grp in sub.groupby("D", sort=True)
        }
        rows.append(
            {
                "center_D": float(center_D),
                "sigma_D": float(sigma_D),
                "beta_peak": float(beta_peak),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_cf"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_cf"]))),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_cf"])),
                "D5p898_max_abs_delta_mu": float(by_d.get(5.898305084745763, np.nan)),
                "D6p441_max_abs_delta_mu": float(by_d.get(6.4406779661016955, np.nan)),
                "D6p712_max_abs_delta_mu": float(by_d.get(6.711864406779661, np.nan)),
                "D6p983_max_abs_delta_mu": float(by_d.get(6.983050847457627, np.nan)),
                "joint_objective": float(
                    max(
                        by_d.get(6.711864406779661, 0.0),
                        0.5 * by_d.get(6.4406779661016955, 0.0),
                        0.25 * by_d.get(5.898305084745763, 0.0),
                    )
                ),
                "p95_effective_ref_vs_raw_ratio": float(np.percentile(sub["effective_ref_vs_raw_ratio"], 95.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["joint_objective", "D6p712_max_abs_delta_mu", "D6p441_max_abs_delta_mu"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (center_D, sigma_D, beta_peak, d_val), grp in detail.groupby(["center_D", "sigma_D", "beta_peak", "D"], sort=True):
        rows.append(
            {
                "center_D": float(center_D),
                "sigma_D": float(sigma_D),
                "beta_peak": float(beta_peak),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_cf"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_cf"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_cf"])),
                "slice_max_beta_eff": float(np.max(grp["beta_eff"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["center_D", "sigma_D", "beta_peak", "D"]).reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("D6p712_max_abs_delta_mu", "D≈6.712 max |Δμ|"),
            ("D6p441_max_abs_delta_mu", "D≈6.441 guard"),
            ("joint_objective", "joint objective"),
        ],
    ):
        pivot = (
            summary[summary["sigma_D"] == 0.05]
            .pivot_table(index="beta_peak", columns="center_D", values=metric, aggfunc="min")
            .sort_index()
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{title}\n(sigma=0.05)")
        ax.set_xlabel("center_D")
        ax.set_ylabel("beta_peak")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Missing reviewer-hotspot detail file: {DETAIL_PATH}")
    base_detail = pd.read_csv(DETAIL_PATH)
    detail = _build_detail(base_detail)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d6712_pointamp_extension_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d6712_pointamp_extension_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d6712_pointamp_extension_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d6712_pointamp_extension_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d6712_pointamp_extension_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "detail_source": str(DETAIL_PATH),
                "focus_D": [float(x) for x in FOCUS_D],
                "centers": [float(x) for x in CENTER_DS],
                "sigmas": [float(x) for x in SIGMAS],
                "betas": [float(x) for x in BETAS],
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
