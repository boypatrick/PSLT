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
DETAIL_PATH = OUTDIR / "model_chain_observable_path_audit_detail.csv"

FOCUS_MODES = ("release_tuned", "release_fullsumanchor", "tailm2_detlin")
FAMILIES = ("current", "width_anchor", "partial_anchor")


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _base_from_log_ratio(cur: float, delta_log: float) -> float:
    return float(cur / np.exp(float(delta_log)))


def _family_mu(base_partial: float, cur_partial: float, base_width: float, cur_width: float, family: str) -> float:
    if family == "current":
        return float(cur_partial / max(cur_width, 1e-30))
    if family == "width_anchor":
        return float(cur_partial / max(base_width, 1e-30))
    if family == "partial_anchor":
        return float(base_partial / max(cur_width, 1e-30))
    raise ValueError(f"Unsupported family={family}")


def _build_detail(base_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    sub = base_detail[base_detail["mode"].isin(FOCUS_MODES)].copy()
    for row in sub.itertuples(index=False):
        cur_partial = float(row.partial_ratio)
        cur_width = float(row.width_ratio)
        cur_mu = float(row.mu)
        base_partial = _base_from_log_ratio(cur_partial, float(row.delta_log_partial))
        base_width = _base_from_log_ratio(cur_width, float(row.delta_log_width))
        base_mu = _base_from_log_ratio(cur_mu, float(row.delta_log_mu))
        base_chi2 = float(((base_mu - 1.4) / 0.4) ** 2) if int(row.layer) == 2 else float("nan")
        for family in FAMILIES:
            mu_cf = _family_mu(base_partial, cur_partial, base_width, cur_width, family)
            delta_mu_cf = float(mu_cf - base_mu)
            delta_log_mu_cf = _safe_log_ratio(mu_cf, base_mu)
            chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2) if int(row.layer) == 2 else float("nan")
            rows.append(
                {
                    "mode": str(row.mode),
                    "family": family,
                    "layer": int(row.layer),
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "mu_cf": mu_cf,
                    "base_mu": base_mu,
                    "delta_mu_cf": delta_mu_cf,
                    "delta_log_mu_cf": delta_log_mu_cf,
                    "acceptance_mismatch_cf": float(((base_chi2 <= 4.0) ^ (chi2_cf <= 4.0)) if int(row.layer) == 2 else 0.0),
                }
            )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for layer in sorted(detail["layer"].unique()):
        for mode in FOCUS_MODES:
            for family in FAMILIES:
                sub = detail[(detail["layer"] == layer) & (detail["mode"] == mode) & (detail["family"] == family)].copy()
                rows.append(
                    {
                        "layer": int(layer),
                        "mode": mode,
                        "family": family,
                        "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_cf"]), 95.0)),
                        "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_cf"]))),
                        "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu_cf"]), 95.0)),
                        "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_cf"])) if int(layer) == 2 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail[detail["layer"] == 2].copy()
    for (mode, family, d_val), grp in sub.groupby(["mode", "family", "D"], sort=True):
        rows.append(
            {
                "mode": mode,
                "family": family,
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_cf"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_cf"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_cf"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "family", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "release_tuned": "#f28e2b",
        "release_fullsumanchor": "#9c755f",
        "tailm2_detlin": "#e15759",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for ax, family in zip(axes, FAMILIES):
        fam = slices[slices["family"] == family]
        for mode in FOCUS_MODES:
            sub = fam[fam["mode"] == mode]
            ax.plot(sub["D"], sub["slice_p95_abs_delta_mu"], marker="o", color=palette[mode], label=mode)
        ax.set_title(f"{family}: layer-2 p95 |Δμ|")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Missing prerequisite detail file: {DETAIL_PATH}")
    base_detail = pd.read_csv(DETAIL_PATH)
    detail = _build_detail(base_detail)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_observable_anchor_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_observable_anchor_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_observable_anchor_audit_slices.csv"
    png_path = OUTDIR / "model_chain_observable_anchor_audit.png"
    meta_path = OUTDIR / "model_chain_observable_anchor_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_slices(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "detail_source": str(DETAIL_PATH),
                "focus_modes": list(FOCUS_MODES),
                "families": list(FAMILIES),
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
