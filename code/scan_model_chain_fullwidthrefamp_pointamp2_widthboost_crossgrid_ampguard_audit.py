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
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
LAYER = 2
REF_ETA = 1.0
REF_D60 = 9.966101694915254
REF_D21 = 9.6

D_GRID60 = np.linspace(4.0, 20.0, 60)
ETA_GRID60 = np.linspace(0.2, 4.0, 21)
D_GRID21 = np.linspace(4.0, 20.0, 21)
ETA_GRID21 = np.linspace(0.2, 4.0, 41)

HOT60_TARGETS = [6.169, 6.441]
HOT21_TARGETS = [6.4]
GUARD60_TARGETS = [4.814, 5.627, 6.712]
GUARD21_TARGETS = [4.0, 8.0]

HOT60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in HOT60_TARGETS]
HOT21_D = [float(D_GRID21[int(np.argmin(np.abs(D_GRID21 - t)))]) for t in HOT21_TARGETS]
GUARD60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in GUARD60_TARGETS]
GUARD21_D = [float(D_GRID21[int(np.argmin(np.abs(D_GRID21 - t)))]) for t in GUARD21_TARGETS]

POINT1_PEAKS = [0.50, 0.65]
POINT1_SIGMAS = [0.03, 0.04]
PARTIAL_PEAKS = [0.5, 1.0]
PARTIAL_CENTERS = [6.10, 6.17]
PARTIAL_SIGMAS = [0.05, 0.08]


def _common_kwargs(d_num: int) -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=int(d_num),
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


def _build_ref(d_num: int):
    return make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs(d_num))


def _build_cur(d_num: int, point1_peak: float, point1_sigma: float, partial_peak: float, partial_center: float, partial_sigma: float):
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost",
        **_common_kwargs(d_num),
        observable_point_amp_anchor_peak_override=float(point1_peak),
        observable_point_amp_anchor_center_D_override=6.4406779661016955,
        observable_point_amp_anchor_sigma_D_override=float(point1_sigma),
        observable_partial_anchor_peak_override=float(partial_peak),
        observable_partial_anchor_center_D_override=float(partial_center),
        observable_partial_anchor_sigma_D_override=float(partial_sigma),
    )


def _mu(kin, D: float, eta: float, ref_D: float) -> float:
    return float(
        kin.hll_mu_pred(
            LAYER,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=ref_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )


def _slice_max(kin_ref, kin_cur, Ds: list[float], etas: np.ndarray, ref_D: float) -> dict[float, float]:
    out: dict[float, float] = {}
    for D in Ds:
        vals = []
        for eta in etas:
            vals.append(abs(_mu(kin_cur, D, float(eta), ref_D) - _mu(kin_ref, D, float(eta), ref_D)))
        out[float(D)] = float(max(vals))
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin_ref60 = _build_ref(60)
    kin_ref21 = _build_ref(21)
    rows = []
    combos = [
        (p1, s1, pp, pc, ps)
        for p1 in POINT1_PEAKS
        for s1 in POINT1_SIGMAS
        for pp in PARTIAL_PEAKS
        for pc in PARTIAL_CENTERS
        for ps in PARTIAL_SIGMAS
    ]
    total = len(combos)
    for idx, (point1_peak, point1_sigma, partial_peak, partial_center, partial_sigma) in enumerate(combos, start=1):
        print(
            f"[combo {idx}/{total}] "
            f"p1={point1_peak:.2f} s1={point1_sigma:.2f} "
            f"pp={partial_peak:.2f} pc={partial_center:.2f} ps={partial_sigma:.2f}",
            flush=True,
        )
        kin_cur60 = _build_cur(60, point1_peak, point1_sigma, partial_peak, partial_center, partial_sigma)
        kin_cur21 = _build_cur(21, point1_peak, point1_sigma, partial_peak, partial_center, partial_sigma)
        hot60 = _slice_max(kin_ref60, kin_cur60, HOT60_D, ETA_GRID60, REF_D60)
        hot21 = _slice_max(kin_ref21, kin_cur21, HOT21_D, ETA_GRID21, REF_D21)
        guard60 = _slice_max(kin_ref60, kin_cur60, GUARD60_D, ETA_GRID60, REF_D60)
        guard21 = _slice_max(kin_ref21, kin_cur21, GUARD21_D, ETA_GRID21, REF_D21)
        rows.append({
            "point1_peak": float(point1_peak),
            "point1_sigma_D": float(point1_sigma),
            "partial_peak": float(partial_peak),
            "partial_center_D": float(partial_center),
            "partial_sigma_D": float(partial_sigma),
            "D60_D6p169_max_abs_delta_mu": float(hot60[HOT60_D[0]]),
            "D60_D6p441_max_abs_delta_mu": float(hot60[HOT60_D[1]]),
            "D21_D6p4_max_abs_delta_mu": float(hot21[HOT21_D[0]]),
            "D60_guard_D4p814_max_abs_delta_mu": float(guard60[GUARD60_D[0]]),
            "D60_guard_D5p627_max_abs_delta_mu": float(guard60[GUARD60_D[1]]),
            "D60_guard_D6p712_max_abs_delta_mu": float(guard60[GUARD60_D[2]]),
            "D21_guard_D4p0_max_abs_delta_mu": float(guard21[GUARD21_D[0]]),
            "D21_guard_D8p0_max_abs_delta_mu": float(guard21[GUARD21_D[1]]),
            "hotspot_objective": float(max(max(hot60.values()), max(hot21.values()))),
            "guard_objective": float(max(max(guard60.values()), max(guard21.values()))),
            "joint_objective": float(max(max(hot60.values()), max(hot21.values()), max(guard60.values()), max(guard21.values()))),
        })

    df = pd.DataFrame(rows).sort_values(
        [
            "joint_objective",
            "hotspot_objective",
            "D21_D6p4_max_abs_delta_mu",
            "D60_D6p441_max_abs_delta_mu",
            "guard_objective",
        ]
    ).reset_index(drop=True)

    stem = "model_chain_fullwidthrefamp_pointamp2_widthboost_crossgrid_ampguard_audit"
    detail_path = OUTDIR / f"{stem}_detail.csv"
    summary_path = OUTDIR / f"{stem}_summary.csv"
    slices_path = OUTDIR / f"{stem}_slices.csv"
    png_path = OUTDIR / f"{stem}.png"
    meta_path = OUTDIR / f"{stem}_run_meta.json"

    df.to_csv(detail_path, index=False)
    df.to_csv(summary_path, index=False)
    df.to_csv(slices_path, index=False)

    top = df.head(15)
    fig, ax = plt.subplots(figsize=(12.5, 5.6), constrained_layout=True)
    x = range(len(top))
    ax.plot(x, top["D60_D6p169_max_abs_delta_mu"], marker="o", label="D60 D≈6.169")
    ax.plot(x, top["D60_D6p441_max_abs_delta_mu"], marker="o", label="D60 D≈6.441")
    ax.plot(x, top["D21_D6p4_max_abs_delta_mu"], marker="o", linestyle="--", label="D21 D=6.4")
    ax.plot(x, top["D60_guard_D5p627_max_abs_delta_mu"], marker="o", linestyle=":", label="D60 guard D≈5.627")
    ax.plot(x, top["D60_guard_D6p712_max_abs_delta_mu"], marker="o", linestyle="-.", label="D60 guard D≈6.712")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [
            f"{r.point1_peak:.2f}\n{r.point1_sigma_D:.2f}\n{r.partial_peak:.2f}\n{r.partial_center_D:.2f}\n{r.partial_sigma_D:.2f}"
            for r in top.itertuples()
        ],
        fontsize=6,
    )
    ax.set_ylabel("max |Δμ|")
    ax.set_title("Cross-grid partial/point-amp guard candidates")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    meta_path.write_text(
        json.dumps(
            {
                "hot60_D": HOT60_D,
                "hot21_D": HOT21_D,
                "guard60_D": GUARD60_D,
                "guard21_D": GUARD21_D,
                "point1_peaks": POINT1_PEAKS,
                "point1_sigmas": POINT1_SIGMAS,
                "partial_peaks": PARTIAL_PEAKS,
                "partial_centers": PARTIAL_CENTERS,
                "partial_sigmas": PARTIAL_SIGMAS,
            },
            indent=2,
        )
    )

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
