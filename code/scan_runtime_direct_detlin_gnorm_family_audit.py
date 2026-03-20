#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

FULL_MAP = MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"
TAIL_MAP = MAPDIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_tailm2_detlin_D21E41.csv"

OUT_SCAN = OUTDIR / "runtime_direct_detlin_gnorm_family_audit_scan.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_gnorm_family_audit_summary.csv"
OUT_PNG = OUTDIR / "runtime_direct_detlin_gnorm_family_audit.png"
OUT_META = OUTDIR / "runtime_direct_detlin_gnorm_family_audit_run_meta.json"

FOCUS_D = [4.0, 4.8, 6.4, 7.2, 8.0]
REF_D = float(PAPER_BASELINE["ref_D"])
REF_ETA = float(PAPER_BASELINE["ref_eta"])
T_COH = float(PAPER_BASELINE["t_coh"])


def _load_map(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    rows: dict[tuple[float, float], dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = (float(row["D"]), float(row["eta"]))
            rows[key] = {
                "mu_mumu": float(row["mu_mumu"]),
                "chi2_mumu": float(row["chi2_mumu"]),
            }
    return rows


def _safe_log(x: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(x), floor)))


def _activation(metric: float, lo: float, hi: float) -> float:
    if hi <= lo + 1e-12:
        return 1.0 if metric > lo else 0.0
    return float(np.clip((metric - lo) / (hi - lo), 0.0, 1.0))


def _uniform_scale(diag: np.ndarray, ref_diag: np.ndarray, beta: float, lo: float, hi: float) -> np.ndarray:
    diag = np.maximum(np.asarray(diag, dtype=float).reshape(3), 1e-30)
    ref_diag = np.maximum(np.asarray(ref_diag, dtype=float).reshape(3), 1e-30)
    metric = _safe_log(np.linalg.norm(diag) / np.linalg.norm(ref_diag))
    act = _activation(metric, lo, hi)
    if act <= 0.0 or beta <= 0.0:
        return diag
    scale = float(np.exp(-beta * act))
    return np.maximum(scale * diag, 1e-30)


def _side_scale(diag: np.ndarray, ref_diag: np.ndarray, beta: float, lo: float, hi: float) -> np.ndarray:
    diag = np.maximum(np.asarray(diag, dtype=float).reshape(3), 1e-30)
    ref_diag = np.maximum(np.asarray(ref_diag, dtype=float).reshape(3), 1e-30)
    side = float(np.sqrt(diag[0] * diag[2]) / diag[1])
    side_ref = float(np.sqrt(ref_diag[0] * ref_diag[2]) / ref_diag[1])
    metric = _safe_log(side / side_ref)
    act = _activation(metric, lo, hi)
    if act <= 0.0 or beta <= 0.0:
        return diag
    scale = float(np.exp(-beta * act))
    out = np.array(diag, dtype=float, copy=True)
    out[0] *= scale
    out[2] *= scale
    return np.maximum(out, 1e-30)


def _build_kinetics(mode: str):
    return make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=mode,
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )


def _evaluate_candidate(
    kin,
    base_method,
    ref_diag: np.ndarray,
    family: str,
    beta: float,
    lo: float,
    hi: float,
    full_map: dict[tuple[float, float], dict[str, float]],
    mu_obs: float,
    sigma_obs: float,
) -> dict[str, float]:
    if family == "uniform":
        transform = _uniform_scale
    elif family == "side":
        transform = _side_scale
    else:
        raise ValueError(f"unknown family {family}")

    def _patched(self, D: float):
        base = np.asarray(base_method(float(D)), dtype=float)
        diag = np.diag(base).astype(float)
        diag_corr = transform(diag, ref_diag, beta, lo, hi)
        return np.diag(diag_corr)

    kin._hll_g_uv_matrix = types.MethodType(_patched, kin)

    slice_stats: dict[float, dict[str, float]] = {}
    worst_abs = 0.0
    for D in FOCUS_D:
        deltas = []
        mismatch = 0
        rows = [key for key in full_map if abs(key[0] - float(D)) < 1e-9]
        rows.sort(key=lambda key: key[1])
        for _, eta in rows:
            ref = full_map[(float(D), float(eta))]
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(D),
                    eta=float(eta),
                    t_coh=T_COH,
                    ref_D=REF_D,
                    ref_eta=REF_ETA,
                    observable_mode="eft_wilson_uv_rge",
                    N_max=kin.params.hll_observable_nmax,
                )
            )
            chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
            delta = abs(mu - ref["mu_mumu"])
            deltas.append(delta)
            worst_abs = max(worst_abs, delta)
            mismatch += int((ref["chi2_mumu"] <= 4.0) != (chi2 <= 4.0))
        arr = np.asarray(deltas, dtype=float)
        slice_stats[float(D)] = {
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95.0)),
            "mismatch": float(mismatch / max(len(arr), 1)),
        }

    objective = (
        1.0 * slice_stats[4.8]["max"]
        + 0.5 * slice_stats[6.4]["max"]
        + 0.25 * slice_stats[7.2]["max"]
        + 5.0 * slice_stats[4.0]["mismatch"]
        + 1.0 * slice_stats[8.0]["max"]
    )
    return {
        "family": family,
        "beta": float(beta),
        "metric_lo": float(lo),
        "metric_hi": float(hi),
        "objective": float(objective),
        "global_max_abs_delta_mu_mumu": float(worst_abs),
        "d4p8_max_abs_delta_mu_mumu": slice_stats[4.8]["max"],
        "d6p4_max_abs_delta_mu_mumu": slice_stats[6.4]["max"],
        "d7p2_max_abs_delta_mu_mumu": slice_stats[7.2]["max"],
        "d8p0_max_abs_delta_mu_mumu": slice_stats[8.0]["max"],
        "d4p0_acceptance_mismatch": slice_stats[4.0]["mismatch"],
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    full_map = _load_map(FULL_MAP)
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)
    kin = _build_kinetics("cell_direct_runtime_release_tailm2")
    base_method = kin._hll_g_uv_matrix
    ref_diag = np.diag(np.asarray(base_method(REF_D), dtype=float)).astype(float)
    grids = {
        # D=6.4 is g-norm dominated in the trusted detlin audit, so spend most
        # of the budget on norm-compression candidates that only wake up in the
        # higher-lognorm band and largely spare D=7.2/8.0.
        "uniform": {
            "beta": (0.15, 0.30, 0.45),
            "lo": (0.80, 1.00),
            "hi": (1.20, 1.40),
        },
        # Keep a smaller side-ratio family as a cross-check.
        "side": {
            "beta": (0.15, 0.30),
            "lo": (0.10, 0.20),
            "hi": (0.35, 0.45),
        },
    }
    rows: list[dict[str, float]] = []
    for family, grid in grids.items():
        for beta in grid["beta"]:
            for lo in grid["lo"]:
                for hi in grid["hi"]:
                    if hi <= lo:
                        continue
                    rows.append(_evaluate_candidate(kin, base_method, ref_diag, family, beta, lo, hi, full_map, mu_obs, sigma_obs))

    scan_df = pd.DataFrame(rows).sort_values(["objective", "family", "beta", "metric_lo", "metric_hi"]).reset_index(drop=True)
    best = scan_df.iloc[0]
    constrained = scan_df[
        (scan_df["d4p0_acceptance_mismatch"] <= 0.195122 + 1e-9)
        & (scan_df["d7p2_max_abs_delta_mu_mumu"] <= 1.022839 + 1e-9)
    ].copy()
    best_constrained = constrained.iloc[0] if not constrained.empty else None

    summary = {
        "rows": int(len(scan_df)),
        "best_family": str(best["family"]),
        "best_beta": float(best["beta"]),
        "best_metric_lo": float(best["metric_lo"]),
        "best_metric_hi": float(best["metric_hi"]),
        "best_objective": float(best["objective"]),
        "best_d4p8_max_abs_delta_mu_mumu": float(best["d4p8_max_abs_delta_mu_mumu"]),
        "best_d6p4_max_abs_delta_mu_mumu": float(best["d6p4_max_abs_delta_mu_mumu"]),
        "best_d7p2_max_abs_delta_mu_mumu": float(best["d7p2_max_abs_delta_mu_mumu"]),
        "best_d8p0_max_abs_delta_mu_mumu": float(best["d8p0_max_abs_delta_mu_mumu"]),
        "best_d4p0_acceptance_mismatch": float(best["d4p0_acceptance_mismatch"]),
        "num_constrained_candidates": int(len(constrained)),
    }
    if best_constrained is not None:
        summary.update(
            {
                "best_constrained_family": str(best_constrained["family"]),
                "best_constrained_beta": float(best_constrained["beta"]),
                "best_constrained_metric_lo": float(best_constrained["metric_lo"]),
                "best_constrained_metric_hi": float(best_constrained["metric_hi"]),
                "best_constrained_objective": float(best_constrained["objective"]),
                "best_constrained_d4p8_max_abs_delta_mu_mumu": float(best_constrained["d4p8_max_abs_delta_mu_mumu"]),
                "best_constrained_d6p4_max_abs_delta_mu_mumu": float(best_constrained["d6p4_max_abs_delta_mu_mumu"]),
                "best_constrained_d7p2_max_abs_delta_mu_mumu": float(best_constrained["d7p2_max_abs_delta_mu_mumu"]),
                "best_constrained_d8p0_max_abs_delta_mu_mumu": float(best_constrained["d8p0_max_abs_delta_mu_mumu"]),
                "best_constrained_d4p0_acceptance_mismatch": float(best_constrained["d4p0_acceptance_mismatch"]),
            }
        )

    scan_df.to_csv(OUT_SCAN, index=False)
    pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, family in zip(axes, ("uniform", "side")):
        sub = scan_df[scan_df["family"] == family].copy()
        scat = ax.scatter(
            sub["d6p4_max_abs_delta_mu_mumu"],
            sub["d4p8_max_abs_delta_mu_mumu"],
            c=sub["d7p2_max_abs_delta_mu_mumu"],
            cmap="viridis",
            s=32,
            alpha=0.85,
        )
        ax.set_title(f"{family} g correction")
        ax.set_xlabel("D=6.4 max |Δμ_μμ|")
        ax.set_ylabel("D=4.8 max |Δμ_μμ|")
        ax.grid(alpha=0.25)
        best_sub = sub.iloc[np.argmin(sub["objective"].to_numpy())]
        ax.scatter(
            [best_sub["d6p4_max_abs_delta_mu_mumu"]],
            [best_sub["d4p8_max_abs_delta_mu_mumu"]],
            color="red",
            marker="x",
            s=90,
            linewidths=2.0,
        )
    cbar = fig.colorbar(scat, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("D=7.2 max |Δμ_μμ|")
    fig.suptitle("Detlin UV/g Correction Audit: D=6.4-focused direct-only families")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    meta = {
        "focus_D": FOCUS_D,
        "full_map": str(FULL_MAP),
        "tail_map": str(TAIL_MAP),
        "summary": summary,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    for path in (OUT_SCAN, OUT_SUMMARY, OUT_PNG, OUT_META):
        shutil.copy2(path, PAPERDIR / path.name)
        print(f"[saved] {path}")

    print(",".join(summary.keys()))
    print(",".join(str(summary[k]) for k in summary.keys()))


if __name__ == "__main__":
    main()
