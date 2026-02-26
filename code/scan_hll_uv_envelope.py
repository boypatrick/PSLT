#!/usr/bin/env python3
"""
Build a map-level UV-to-EFT systematics envelope for H->mumu.

This script runs one-at-a-time UV control variations around a nonzero
finite-match center and aggregates a pointwise mu_mumu envelope.

Outputs:
  - output/robustness/hll_uv_envelope_cases.csv
  - output/robustness/hll_uv_envelope_map.csv
  - output/robustness/hll_uv_envelope_summary.csv
  - output/robustness/hll_uv_envelope.png
  - paper/hll_uv_envelope_cases.csv
  - paper/hll_uv_envelope_map.csv
  - paper/hll_uv_envelope_summary.csv
  - paper/hll_uv_envelope.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scan_hll_signal_strengths import (
    PAPER_BASELINE,
    compute_maps,
    load_observations,
    make_baseline_kinetics,
    snap_ref_d_for_full_direct,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "robustness"
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class UVCase:
    name: str
    uv_rge_mu_low: float
    uv_rge_gamma_diag: float
    uv_rge_gamma_offdiag: float
    uv_match_kappa_diag: float
    uv_match_kappa_offdiag: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map-level UV systematics envelope for H->mumu.")
    ap.add_argument("--chain-mode", choices=["auto", "full_direct"], default="full_direct")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--d-num", type=int, default=21)
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--eta-num", type=int, default=41)
    ap.add_argument("--ref-d", type=float, default=10.0)
    ap.add_argument("--ref-eta", type=float, default=1.0)
    ap.add_argument("--center-kappa-diag", type=float, default=0.30)
    ap.add_argument("--center-kappa-offdiag", type=float, default=0.15)
    ap.add_argument("--center-gamma-diag", type=float, default=2.0)
    ap.add_argument("--center-gamma-offdiag", type=float, default=1.0)
    ap.add_argument("--center-mu-low", type=float, default=1.0)
    return ap.parse_args()


def build_cases(args: argparse.Namespace) -> List[UVCase]:
    # One-at-a-time windows around a nonzero finite-match center.
    return [
        UVCase(
            "center_nonzero",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "mu_low_minus",
            uv_rge_mu_low=0.5,
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "mu_low_plus",
            uv_rge_mu_low=2.0,
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "gamma_diag_minus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=1.5,
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "gamma_diag_plus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=2.5,
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "gamma_offdiag_minus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=0.8,
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "gamma_offdiag_plus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=1.2,
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "kappa_diag_minus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=0.10,
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "kappa_diag_plus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=0.50,
            uv_match_kappa_offdiag=float(args.center_kappa_offdiag),
        ),
        UVCase(
            "kappa_offdiag_minus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=0.05,
        ),
        UVCase(
            "kappa_offdiag_plus",
            uv_rge_mu_low=float(args.center_mu_low),
            uv_rge_gamma_diag=float(args.center_gamma_diag),
            uv_rge_gamma_offdiag=float(args.center_gamma_offdiag),
            uv_match_kappa_diag=float(args.center_kappa_diag),
            uv_match_kappa_offdiag=0.25,
        ),
        # Keep a legacy zero-match comparator for context.
        UVCase(
            "legacy_zero_match",
            uv_rge_mu_low=1.0,
            uv_rge_gamma_diag=2.0,
            uv_rge_gamma_offdiag=1.0,
            uv_match_kappa_diag=0.0,
            uv_match_kappa_offdiag=0.0,
        ),
    ]


def evaluate_case(
    case: UVCase,
    args: argparse.Namespace,
    obs_mu: float,
    obs_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float | str]]:
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=str(args.chain_mode),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        uv_blend=float(PAPER_BASELINE["hll_uv_blend"]),
        uv_m2_power=float(PAPER_BASELINE["hll_uv_m2_power"]),
        uv_match_kappa_diag=float(case.uv_match_kappa_diag),
        uv_match_kappa_offdiag=float(case.uv_match_kappa_offdiag),
        uv_rge_mu_low=float(case.uv_rge_mu_low),
        uv_rge_gamma_diag=float(case.uv_rge_gamma_diag),
        uv_rge_gamma_offdiag=float(case.uv_rge_gamma_offdiag),
        uv_rge_log_clip=float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
    )

    d_vals = np.linspace(float(args.d_min), float(args.d_max), int(args.d_num))
    ref_d_eff, snapped = snap_ref_d_for_full_direct(str(args.chain_mode), float(args.ref_d), d_vals)
    d_vals, eta_vals, maps, _ = compute_maps(
        kinetics=kin,
        ref_d=float(ref_d_eff),
        ref_eta=float(args.ref_eta),
        observable_mode="eft_wilson_uv_rge",
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
    )
    mu = np.asarray(maps["mumu"], dtype=float)
    chi2 = ((mu - float(obs_mu)) / float(obs_sigma)) ** 2

    row: Dict[str, float | str] = {
        "case": case.name,
        "chain_mode": str(args.chain_mode),
        "d_num": int(args.d_num),
        "eta_num": int(args.eta_num),
        "ref_D_input": float(args.ref_d),
        "ref_D_effective": float(ref_d_eff),
        "ref_D_snapped_to_grid": bool(snapped),
        "ref_eta": float(args.ref_eta),
        "uv_rge_mu_low": float(case.uv_rge_mu_low),
        "uv_rge_gamma_diag": float(case.uv_rge_gamma_diag),
        "uv_rge_gamma_offdiag": float(case.uv_rge_gamma_offdiag),
        "uv_match_kappa_diag": float(case.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(case.uv_match_kappa_offdiag),
        "f_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "mu_mean": float(np.mean(mu)),
        "mu_p50": float(np.percentile(mu, 50.0)),
        "mu_p90": float(np.percentile(mu, 90.0)),
        "mu_std": float(np.std(mu)),
    }
    return d_vals, eta_vals, mu, chi2, row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_envelope_map_csv(
    out_csv: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    mu_center: np.ndarray,
    case_lo: np.ndarray,
    case_hi: np.ndarray,
) -> None:
    rows: List[Dict[str, object]] = []
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            rows.append(
                {
                    "D": float(d),
                    "eta": float(eta),
                    "mu_mumu_center": float(mu_center[i, j]),
                    "mu_mumu_min": float(mu_lo[i, j]),
                    "mu_mumu_max": float(mu_hi[i, j]),
                    "mu_mumu_half_width": float(0.5 * (mu_hi[i, j] - mu_lo[i, j])),
                    "mu_mumu_full_span": float(mu_hi[i, j] - mu_lo[i, j]),
                    "case_at_min": str(case_lo[i, j]),
                    "case_at_max": str(case_hi[i, j]),
                }
            )
    write_csv(out_csv, rows)


def make_plot(
    out_png: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    mu_center: np.ndarray,
    mu_half_width: np.ndarray,
    mu_full_span: np.ndarray,
) -> None:
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        (mu_center, r"$\mu_{\mu\mu}^{\rm center}$"),
        (mu_half_width, r"Envelope half-width"),
        (mu_full_span, r"Envelope full span"),
    ]
    for ax, (data, ttl) in zip(axes, panels):
        p05, p95 = np.percentile(data, [5.0, 95.0])
        if p95 <= p05:
            p05 = float(np.min(data))
            p95 = float(np.max(data)) + 1e-12
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=float(p05),
            vmax=float(p95),
        )
        ax.set_title(ttl)
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("value")
    fig.suptitle("UV control envelope on H->mumu map", fontsize=12.5)
    fig.savefig(out_png, dpi=210)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if int(args.d_num) < 2 or int(args.eta_num) < 2:
        raise ValueError("d-num and eta-num must be >= 2.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    observations = load_observations()
    obs = observations.get("mumu")
    if obs is None:
        obs_mu, obs_sigma = 1.4, 0.4
    else:
        obs_mu, obs_sigma = float(obs.mu_obs), float(obs.sigma_obs)

    cases = build_cases(args)
    case_rows: List[Dict[str, object]] = []
    case_names: List[str] = []
    mu_stack: List[np.ndarray] = []
    chi2_stack: List[np.ndarray] = []
    d_ref: np.ndarray | None = None
    eta_ref: np.ndarray | None = None

    for case in cases:
        d_vals, eta_vals, mu, chi2, row = evaluate_case(case, args, obs_mu=obs_mu, obs_sigma=obs_sigma)
        if d_ref is None:
            d_ref = d_vals
            eta_ref = eta_vals
        else:
            if not np.allclose(d_ref, d_vals) or not np.allclose(eta_ref, eta_vals):
                raise RuntimeError("grid mismatch across UV envelope cases")
        case_rows.append(row)
        case_names.append(case.name)
        mu_stack.append(mu)
        chi2_stack.append(chi2)
        print(f"[case] {case.name} f(chi2<=4)={float(row['f_chi2_le_4']):.4f}")

    assert d_ref is not None and eta_ref is not None
    mu_arr = np.asarray(mu_stack, dtype=float)
    chi2_arr = np.asarray(chi2_stack, dtype=float)
    name_arr = np.asarray(case_names, dtype=object)
    idx_lo = np.argmin(mu_arr, axis=0)
    idx_hi = np.argmax(mu_arr, axis=0)
    mu_lo = np.min(mu_arr, axis=0)
    mu_hi = np.max(mu_arr, axis=0)

    center_idx = int(case_names.index("center_nonzero"))
    center_mu = mu_arr[center_idx]
    mu_half_width = 0.5 * (mu_hi - mu_lo)
    mu_full_span = mu_hi - mu_lo
    chi2_accept_frac_cases = np.mean(chi2_arr <= 4.0, axis=(1, 2))

    case_lo = name_arr[idx_lo]
    case_hi = name_arr[idx_hi]
    envelope_summary = {
        "chain_mode": str(args.chain_mode),
        "d_num": int(args.d_num),
        "eta_num": int(args.eta_num),
        "n_cases": int(len(case_names)),
        "center_case": "center_nonzero",
        "obs_mu": float(obs_mu),
        "obs_sigma": float(obs_sigma),
        "f_chi2_le_4_center": float(chi2_accept_frac_cases[center_idx]),
        "f_chi2_le_4_min": float(np.min(chi2_accept_frac_cases)),
        "f_chi2_le_4_max": float(np.max(chi2_accept_frac_cases)),
        "f_chi2_le_4_span": float(np.max(chi2_accept_frac_cases) - np.min(chi2_accept_frac_cases)),
        "case_at_f_chi2_le_4_min": str(case_names[int(np.argmin(chi2_accept_frac_cases))]),
        "case_at_f_chi2_le_4_max": str(case_names[int(np.argmax(chi2_accept_frac_cases))]),
        "mean_mu_half_width": float(np.mean(mu_half_width)),
        "p95_mu_half_width": float(np.percentile(mu_half_width, 95.0)),
        "max_mu_half_width": float(np.max(mu_half_width)),
        "mean_mu_full_span": float(np.mean(mu_full_span)),
        "p95_mu_full_span": float(np.percentile(mu_full_span, 95.0)),
        "max_mu_full_span": float(np.max(mu_full_span)),
    }

    out_cases = OUTDIR / "hll_uv_envelope_cases.csv"
    out_map = OUTDIR / "hll_uv_envelope_map.csv"
    out_summary = OUTDIR / "hll_uv_envelope_summary.csv"
    out_fig = OUTDIR / "hll_uv_envelope.png"
    paper_cases = PAPER_DIR / out_cases.name
    paper_map = PAPER_DIR / out_map.name
    paper_summary = PAPER_DIR / out_summary.name
    paper_fig = PAPER_DIR / out_fig.name

    write_csv(out_cases, case_rows)
    write_envelope_map_csv(
        out_csv=out_map,
        d_vals=d_ref,
        eta_vals=eta_ref,
        mu_lo=mu_lo,
        mu_hi=mu_hi,
        mu_center=center_mu,
        case_lo=case_lo,
        case_hi=case_hi,
    )
    write_csv(out_summary, [envelope_summary])
    make_plot(out_fig, d_ref, eta_ref, center_mu, mu_half_width, mu_full_span)

    paper_cases.write_text(out_cases.read_text())
    paper_map.write_text(out_map.read_text())
    paper_summary.write_text(out_summary.read_text())
    paper_fig.write_bytes(out_fig.read_bytes())

    print(f"[saved] {out_cases}")
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {paper_cases}")
    print(f"[saved] {paper_map}")
    print(f"[saved] {paper_summary}")
    print(f"[saved] {paper_fig}")
    print(envelope_summary)


if __name__ == "__main__":
    main()
