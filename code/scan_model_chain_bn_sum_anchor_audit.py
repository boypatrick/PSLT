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

D_GRID = np.linspace(4.0, 20.0, 21)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)

MODES = [
    ("cell_direct_runtime_release_tuned", "release_tuned"),
    ("cell_direct_runtime_release_tailm2", "tailm2_detlin"),
]


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=float(D_GRID.min()),
        d_max=float(D_GRID.max()),
        d_num=int(len(D_GRID)),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=3.0,
        runtime_direct_chi_z_margin=6.0,
        runtime_direct_chi_n_mu=120,
        runtime_direct_chi_tol=1e-8,
        runtime_direct_chi_maxiter=30000,
        runtime_direct_chi_sigma=2.5,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.sum(vec))
    if denom <= 0.0:
        return np.zeros_like(vec)
    return np.asarray(vec / denom, dtype=float)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (mode_key, family), sub in detail.groupby(["mode", "family"], sort=True):
        rows.append(
            {
                "mode": mode_key,
                "family": family,
                "p95_abs_log_B_sum_ratio": float(np.percentile(np.abs(sub["log_B_sum_ratio"]), 95.0)),
                "max_abs_log_B_sum_ratio": float(np.max(np.abs(sub["log_B_sum_ratio"]))),
                "p95_abs_log_B2_ratio": float(np.percentile(np.abs(sub["log_B2_ratio"]), 95.0)),
                "max_abs_log_B2_ratio": float(np.max(np.abs(sub["log_B2_ratio"]))),
                "p95_share_l1": float(np.percentile(sub["share_l1"], 95.0)),
                "max_share_l1": float(np.max(sub["share_l1"])),
            }
        )
    return pd.DataFrame(rows)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (mode_key, family, d_val), grp in detail.groupby(["mode", "family", "D"], sort=True):
        rows.append(
            {
                "mode": mode_key,
                "family": family,
                "D": float(d_val),
                "slice_p95_abs_log_B_sum_ratio": float(np.percentile(np.abs(grp["log_B_sum_ratio"]), 95.0)),
                "slice_p95_abs_log_B2_ratio": float(np.percentile(np.abs(grp["log_B2_ratio"]), 95.0)),
                "slice_p95_share_l1": float(np.percentile(grp["share_l1"], 95.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "family", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    colors = {
        ("release_tuned", "direct_runtime"): "#f28e2b",
        ("release_tuned", "sum_anchor"): "#edc948",
        ("tailm2_detlin", "direct_runtime"): "#e15759",
        ("tailm2_detlin", "sum_anchor"): "#76b7b2",
    }
    for (mode_key, family), grp in slices.groupby(["mode", "family"], sort=True):
        label = f"{mode_key}:{family}"
        color = colors[(mode_key, family)]
        axes[0].plot(grp["D"], grp["slice_p95_abs_log_B2_ratio"], marker="o", label=label, color=color)
        axes[1].plot(grp["D"], grp["slice_p95_abs_log_B_sum_ratio"], marker="o", label=label, color=color)
    axes[0].set_title("p95 |log B2 ratio|")
    axes[1].set_title("p95 |log ΣB ratio|")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    common = _common_kwargs()
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **common)

    rows: list[dict[str, float | str]] = []
    for chain_mode, mode_key in MODES:
        kin = make_baseline_kinetics(chain_mode=chain_mode, **common)
        for D in D_GRID:
            for eta in ETA_GRID:
                base = np.asarray([float(kin_full.B_N(n, float(D))) for n in (1, 2, 3)], dtype=float)
                cur = np.asarray([float(kin.B_N(n, float(D))) for n in (1, 2, 3)], dtype=float)
                share_cur = _normalize(cur)
                base_sum = float(np.sum(base))
                sum_anchor = share_cur * base_sum

                for family, vec in [("direct_runtime", cur), ("sum_anchor", sum_anchor)]:
                    rows.append(
                        {
                            "mode": mode_key,
                            "family": family,
                            "D": float(D),
                            "eta": float(eta),
                            "log_B_sum_ratio": _safe_log_ratio(float(np.sum(vec)), float(np.sum(base))),
                            "log_B2_ratio": _safe_log_ratio(float(vec[1]), float(base[1])),
                            "share_l1": float(np.sum(np.abs(_normalize(vec) - _normalize(base)))),
                        }
                    )

    detail = pd.DataFrame(rows)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_bn_sum_anchor_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_bn_sum_anchor_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_bn_sum_anchor_audit_slices.csv"
    png_path = OUTDIR / "model_chain_bn_sum_anchor_audit.png"
    meta_path = OUTDIR / "model_chain_bn_sum_anchor_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_slices(slices, png_path)
    meta_path.write_text(json.dumps({"d_grid": [float(x) for x in D_GRID], "eta_grid": [float(x) for x in ETA_GRID]}, indent=2))
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
