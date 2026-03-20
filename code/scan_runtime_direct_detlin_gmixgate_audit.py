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
OUT_SCAN = OUTDIR / "runtime_direct_detlin_gmixgate_audit_scan.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_gmixgate_audit_summary.csv"
OUT_PNG = OUTDIR / "runtime_direct_detlin_gmixgate_audit.png"
OUT_META = OUTDIR / "runtime_direct_detlin_gmixgate_audit_run_meta.json"

FOCUS_D = [4.0, 4.8, 6.4, 7.2, 8.0]
REF_D = float(PAPER_BASELINE["ref_D"])
REF_ETA = float(PAPER_BASELINE["ref_eta"])
T_COH = float(PAPER_BASELINE["t_coh"])


def _load_map(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    rows = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows[(float(row["D"]), float(row["eta"]))] = {
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


def _build_kinetics():
    return make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_tailm2",
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
    beta: float,
    norm_lo: float,
    norm_hi: float,
    side_lo: float,
    side_hi: float,
    full_map: dict[tuple[float, float], dict[str, float]],
    mu_obs: float,
    sigma_obs: float,
) -> dict[str, float]:
    ref_norm = float(np.linalg.norm(ref_diag))
    ref_side = float(np.sqrt(ref_diag[0] * ref_diag[2]) / ref_diag[1])

    def _patched(self, D: float):
        base = np.asarray(base_method(float(D)), dtype=float)
        diag = np.diag(base).astype(float)
        norm_metric = _safe_log(np.linalg.norm(diag) / ref_norm)
        side_metric = _safe_log((np.sqrt(diag[0] * diag[2]) / diag[1]) / ref_side)
        act = _activation(norm_metric, norm_lo, norm_hi) * _activation(side_metric, side_lo, side_hi)
        if act <= 0.0 or beta <= 0.0:
            return np.diag(diag)
        scale = float(np.exp(-beta * act))
        return np.diag(np.maximum(scale * diag, 1e-30))

    kin._hll_g_uv_matrix = types.MethodType(_patched, kin)

    slice_stats = {}
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
            "mismatch": float(mismatch / max(len(arr), 1)),
        }

    objective = (
        1.2 * slice_stats[6.4]["max"]
        + 0.5 * slice_stats[4.8]["max"]
        + 0.5 * slice_stats[7.2]["max"]
        + 0.25 * slice_stats[8.0]["max"]
        + 5.0 * slice_stats[4.0]["mismatch"]
    )
    return {
        "beta": float(beta),
        "norm_lo": float(norm_lo),
        "norm_hi": float(norm_hi),
        "side_lo": float(side_lo),
        "side_hi": float(side_hi),
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
    kin = _build_kinetics()
    base_method = kin._hll_g_uv_matrix
    ref_diag = np.diag(np.asarray(base_method(REF_D), dtype=float)).astype(float)

    rows = []
    for beta in (0.15, 0.30, 0.45, 0.60):
        for norm_lo, norm_hi in ((0.8, 1.2), (0.8, 1.4), (1.0, 1.4)):
            for side_lo, side_hi in ((0.02, 0.12), (0.05, 0.15), (0.08, 0.20)):
                rows.append(
                    _evaluate_candidate(
                        kin,
                        base_method,
                        ref_diag,
                        beta,
                        norm_lo,
                        norm_hi,
                        side_lo,
                        side_hi,
                        full_map,
                        mu_obs,
                        sigma_obs,
                    )
                )

    scan_df = pd.DataFrame(rows).sort_values(["objective", "beta", "norm_lo", "side_lo"]).reset_index(drop=True)
    best = scan_df.iloc[0]
    constrained = scan_df[
        (scan_df["d4p0_acceptance_mismatch"] <= 0.195122 + 1e-9)
        & (scan_df["d7p2_max_abs_delta_mu_mumu"] <= 1.022839 + 1e-9)
        & (scan_df["d8p0_max_abs_delta_mu_mumu"] <= 0.1)
    ].copy()

    summary = {
        "rows": int(len(scan_df)),
        "best_beta": float(best["beta"]),
        "best_norm_lo": float(best["norm_lo"]),
        "best_norm_hi": float(best["norm_hi"]),
        "best_side_lo": float(best["side_lo"]),
        "best_side_hi": float(best["side_hi"]),
        "best_objective": float(best["objective"]),
        "best_d4p8_max_abs_delta_mu_mumu": float(best["d4p8_max_abs_delta_mu_mumu"]),
        "best_d6p4_max_abs_delta_mu_mumu": float(best["d6p4_max_abs_delta_mu_mumu"]),
        "best_d7p2_max_abs_delta_mu_mumu": float(best["d7p2_max_abs_delta_mu_mumu"]),
        "best_d8p0_max_abs_delta_mu_mumu": float(best["d8p0_max_abs_delta_mu_mumu"]),
        "best_d4p0_acceptance_mismatch": float(best["d4p0_acceptance_mismatch"]),
        "num_constrained_candidates": int(len(constrained)),
    }
    if not constrained.empty:
        c = constrained.iloc[0]
        summary.update(
            {
                "best_constrained_beta": float(c["beta"]),
                "best_constrained_norm_lo": float(c["norm_lo"]),
                "best_constrained_norm_hi": float(c["norm_hi"]),
                "best_constrained_side_lo": float(c["side_lo"]),
                "best_constrained_side_hi": float(c["side_hi"]),
                "best_constrained_objective": float(c["objective"]),
                "best_constrained_d4p8_max_abs_delta_mu_mumu": float(c["d4p8_max_abs_delta_mu_mumu"]),
                "best_constrained_d6p4_max_abs_delta_mu_mumu": float(c["d6p4_max_abs_delta_mu_mumu"]),
                "best_constrained_d7p2_max_abs_delta_mu_mumu": float(c["d7p2_max_abs_delta_mu_mumu"]),
                "best_constrained_d8p0_max_abs_delta_mu_mumu": float(c["d8p0_max_abs_delta_mu_mumu"]),
                "best_constrained_d4p0_acceptance_mismatch": float(c["d4p0_acceptance_mismatch"]),
            }
        )

    scan_df.to_csv(OUT_SCAN, index=False)
    pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    scat = ax.scatter(
        scan_df["d6p4_max_abs_delta_mu_mumu"],
        scan_df["d4p8_max_abs_delta_mu_mumu"],
        c=scan_df["d7p2_max_abs_delta_mu_mumu"],
        cmap="viridis",
        s=36,
        alpha=0.9,
    )
    ax.scatter(
        [best["d6p4_max_abs_delta_mu_mumu"]],
        [best["d4p8_max_abs_delta_mu_mumu"]],
        color="red",
        marker="x",
        s=100,
        linewidths=2.0,
        label="best",
    )
    ax.set_xlabel("D=6.4 max |Δμ_μμ|")
    ax.set_ylabel("D=4.8 max |Δμ_μμ|")
    ax.set_title("Detlin UV/g Audit: mix-gated uniform correction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label("D=7.2 max |Δμ_μμ|")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    meta = {"summary": summary, "focus_D": FOCUS_D, "full_map": str(FULL_MAP)}
    OUT_META.write_text(json.dumps(meta, indent=2))
    for path in (OUT_SCAN, OUT_SUMMARY, OUT_PNG, OUT_META):
        shutil.copy2(path, PAPERDIR / path.name)
        print(f"[saved] {path}")

    print(",".join(summary.keys()))
    print(",".join(str(summary[k]) for k in summary.keys()))


if __name__ == "__main__":
    main()
