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
DETAIL_PATH = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit_detail.csv"

CENTER_D = 6.4406779661016955
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIGMAS = (0.05, 0.10, 0.15, 0.20, 0.30)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _geom_blend(cur: float, base: float, beta: float, floor: float = 1e-30) -> float:
    return float(
        np.exp(
            (1.0 - float(beta)) * np.log(max(float(cur), floor))
            + float(beta) * np.log(max(float(base), floor))
        )
    )


def _localized_beta(beta_peak: float, D: float, sigma: float) -> float:
    return float(beta_peak * np.exp(-0.5 * ((float(D) - CENTER_D) / float(sigma)) ** 2))


def _effective_ref_amp(row: pd.Series) -> float:
    partial_eff = float(row["cur_mu"]) * float(row["cur_width_ratio"])
    return float(row["cur_amp"] / np.sqrt(max(partial_eff, 1e-30)))


def _build_detail(base_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in base_detail.itertuples(index=False):
        eff_ref = _effective_ref_amp(pd.Series(row._asdict()))
        for sigma_D in SIGMAS:
            for beta_peak in BETAS:
                beta_eff = _localized_beta(float(beta_peak), float(row.D), float(sigma_D))
                amp_cf = _geom_blend(float(row.cur_amp), float(row.base_amp), beta_eff)
                partial_cf = float((amp_cf / max(eff_ref, 1e-30)) ** 2)
                mu_cf = float(partial_cf / max(float(row.cur_width_ratio), 1e-30))
                base_mu = float(row.base_mu)
                base_chi2 = float(row.base_chi2)
                chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
                rows.append(
                    {
                        "beta_peak": float(beta_peak),
                        "sigma_D": float(sigma_D),
                        "beta_eff": float(beta_eff),
                        "D": float(row.D),
                        "eta": float(row.eta),
                        "effective_ref_amp": eff_ref,
                        "effective_ref_vs_raw_ratio": float(eff_ref / max(float(row.cur_amp_ref), 1e-30)),
                        "base_mu": base_mu,
                        "cur_mu": float(row.cur_mu),
                        "mu_cf": mu_cf,
                        "delta_mu_cf": float(mu_cf - base_mu),
                        "delta_log_mu_cf": _safe_log_ratio(mu_cf, base_mu),
                        "acceptance_mismatch_cf": float((base_chi2 <= 4.0) ^ (chi2_cf <= 4.0)),
                    }
                )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    hotspot_mask = np.isclose(detail["D"].to_numpy(dtype=float), CENTER_D)
    guard_mask = ~hotspot_mask
    for (beta_peak, sigma_D), sub in detail.groupby(["beta_peak", "sigma_D"], sort=True):
        hot = sub.loc[hotspot_mask[sub.index]]
        guard = sub.loc[guard_mask[sub.index]]
        by_d = {
            float(d_val): float(np.max(np.abs(grp["delta_mu_cf"])))
            for d_val, grp in sub.groupby("D", sort=True)
        }
        rows.append(
            {
                "beta_peak": float(beta_peak),
                "sigma_D": float(sigma_D),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_cf"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_cf"]))),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_cf"])),
                "hotspot_p95_abs_delta_mu": float(np.percentile(np.abs(hot["delta_mu_cf"]), 95.0)),
                "hotspot_max_abs_delta_mu": float(np.max(np.abs(hot["delta_mu_cf"]))),
                "guard_max_abs_delta_mu": float(np.max(np.abs(guard["delta_mu_cf"]))),
                "D5p898_max_abs_delta_mu": float(by_d.get(5.898305084745763, np.nan)),
                "D6p169_max_abs_delta_mu": float(by_d.get(6.169491525423728, np.nan)),
                "D6p441_max_abs_delta_mu": float(by_d.get(6.4406779661016955, np.nan)),
                "p95_effective_ref_vs_raw_ratio": float(np.percentile(sub["effective_ref_vs_raw_ratio"], 95.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["beta_peak", "sigma_D"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (beta_peak, sigma_D, d_val), grp in detail.groupby(["beta_peak", "sigma_D", "D"], sort=True):
        rows.append(
            {
                "beta_peak": float(beta_peak),
                "sigma_D": float(sigma_D),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_cf"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_cf"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_cf"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["beta_peak", "sigma_D", "D"]).reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("hotspot_max_abs_delta_mu", "D≈6.44 hotspot"),
        ("guard_max_abs_delta_mu", "guard max"),
        ("max_abs_delta_mu", "global max"),
    ]
    betas = np.array(sorted(summary["beta_peak"].unique()), dtype=float)
    sigmas = np.array(sorted(summary["sigma_D"].unique()), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            summary.pivot(index="beta_peak", columns="sigma_D", values=col)
            .reindex(index=betas, columns=sigmas)
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("sigma_D")
        ax.set_ylabel("beta_peak")
        ax.set_xticks(np.arange(len(sigmas)))
        ax.set_xticklabels([f"{x:.2f}" for x in sigmas], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(betas)))
        ax.set_yticklabels([f"{x:.2f}" for x in betas])
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
        raise FileNotFoundError(f"Missing hotspot-path detail file: {DETAIL_PATH}")
    base_detail = pd.read_csv(DETAIL_PATH)
    detail = _build_detail(base_detail)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_d60_pointamp_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_d60_pointamp_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_d60_pointamp_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_d60_pointamp_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_d60_pointamp_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "detail_source": str(DETAIL_PATH),
                "center_D": CENTER_D,
                "beta_peaks": [float(x) for x in BETAS],
                "sigmas": [float(x) for x in SIGMAS],
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
