#!/usr/bin/env python3
"""
Impact gate for the localized channel-resolved A_l(D, eta, N) tensor.

The current baseline injects the N_ref=2 profile slice A_l(D; N_ref) into all
layers.  This audit builds an active-grid tensor A_l(D, eta, N), evaluates both
the all-valid strict lookup and the same lookup with a bound-sector fallback
for positive-proxy rows, and compares them against the baseline profile branch
without changing the production default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile
from extract_channel_resolved_A_l_tensor import (
    PhysicalParams,
    Level,
    _build_reference_values,
    _collapse_summary,
    _expand_eta_tensor,
    _solve_base_rows,
)
from hll_observable import HLLChannelPredictor, HLLObservableConfig
from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "superrad_fp_1d"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class EvalCase:
    name: str
    gamma_mode: str
    gamma_csv: Path
    gamma_tensor_bound_policy: str = "all_valid"


def ensure_tensor_csv(
    d_scan: np.ndarray,
    *,
    force: bool = False,
    zmax: float = 80.0,
    n_ref: int = 2,
) -> Tuple[Path, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_grid = len(d_scan)
    tensor_path = OUTDIR / f"channel_resolved_A_l_tensor_Dgrid{n_grid}_eta3_N1-2-3_l1-2.csv"
    summary_path = OUTDIR / f"channel_resolved_A_l_tensor_Dgrid{n_grid}_eta3_N1-2-3_l1-2_summary.csv"
    if tensor_path.exists() and summary_path.exists() and not force:
        return tensor_path, summary_path

    d_ref = 12.0
    required_reference_D = (10.0, d_ref)
    d_values = [float(x) for x in d_scan]
    for d_extra in required_reference_D:
        if not np.any(np.isclose(np.asarray(d_values, dtype=float), d_extra, rtol=0.0, atol=1e-10)):
            d_values.append(float(d_extra))
    d_values = sorted(d_values)
    eta_values = [0.25, 1.0, 4.0]
    n_values = [1, 2, 3]
    ell_values = [1, 2]
    level = Level("fine", 8001)
    params = PhysicalParams()
    base = _solve_base_rows(d_values, n_values, ell_values, level, zmax, params)
    refs = _build_reference_values(base, d_ref=d_ref, n_ref=n_ref)
    tensor = _expand_eta_tensor(base, eta_values, refs, n_ref)
    summary = _collapse_summary(tensor)
    tensor.to_csv(tensor_path, index=False)
    summary.to_csv(summary_path, index=False)
    return tensor_path, summary_path


def make_kinetics(case: EvalCase, chi_profile: Dict[str, object]) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode="fp_2d_full",
        g_fp_norm_mode="phase_space",
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        chi_mode=str(chi_profile["mode"]),
        chi_lr_D=tuple(float(x) for x in chi_profile["d"]),
        chi_lr_vals=tuple(float(y) for y in chi_profile["chi"]),
        A1=1.0,
        A2=1.0,
        gamma_mode=str(case.gamma_mode),
        gamma_superrad_csv=str(case.gamma_csv),
        gamma_tensor_bound_policy=str(case.gamma_tensor_bound_policy),
        b_mode="eft_operator_norm",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=20,
    )
    return PSLTKinetics(params)


def evaluate_case(case: EvalCase, kin: PSLTKinetics, d_vals: np.ndarray, eta_vals: np.ndarray) -> pd.DataFrame:
    hll_cfg = HLLObservableConfig(
        mode="eft_wilson_uv_rge",
        t_coh=1.0,
        ref_D=10.0,
        ref_eta=1.0,
        n_max=20,
    )
    hll_mumu = HLLChannelPredictor(kin, layer_n=2, cfg=hll_cfg)
    rows: List[Dict[str, float | int | str]] = []
    for eta in eta_vals:
        for D in d_vals:
            _probs, _weights, meta = kin.get_probabilities(float(D), float(eta), 1.0, N_max=20)
            mu_pred = float(hll_mumu.mu_pred(float(D), float(eta)))
            chi2 = float(((mu_pred - 1.4) / 0.4) ** 2)
            rows.append(
                {
                    "case": case.name,
                    "gamma_mode_requested": case.gamma_mode,
                    "gamma_mode_active": kin.active_gamma_mode(),
                    "gamma_tensor_bound_policy": case.gamma_tensor_bound_policy,
                    "D": float(D),
                    "eta": float(eta),
                    "R3": float(meta["generation_ratio"]),
                    "winner": int(meta["winner"]),
                    "mu_mumu": mu_pred,
                    "chi2_mumu": chi2,
                    "hmumu_accept": bool(chi2 <= 4.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_case(df: pd.DataFrame) -> Dict[str, float | int | str]:
    idx = int(df["chi2_mumu"].idxmin())
    best = df.loc[idx]
    return {
        "row_type": "case",
        "case": str(best["case"]),
        "gamma_mode_active": str(best["gamma_mode_active"]),
        "gamma_tensor_bound_policy": str(best["gamma_tensor_bound_policy"]),
        "n_points": int(len(df)),
        "f_R3_ge_0p90": float(np.mean(df["R3"].to_numpy(dtype=float) >= 0.90)),
        "f_R3_ge_0p95": float(np.mean(df["R3"].to_numpy(dtype=float) >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(df["hmumu_accept"].to_numpy(dtype=bool))),
        "best_chi2": float(best["chi2_mumu"]),
        "best_D": float(best["D"]),
        "best_eta": float(best["eta"]),
        "winner_gt_3_fraction": float(np.mean(df["winner"].to_numpy(dtype=int) > 3)),
    }


def summarize_delta(base: pd.DataFrame, candidate: pd.DataFrame) -> Tuple[Dict[str, float | int | str], pd.DataFrame]:
    b = base.sort_values(["D", "eta"]).reset_index(drop=True)
    t = candidate.sort_values(["D", "eta"]).reset_index(drop=True)
    if len(b) != len(t) or not np.allclose(b["D"], t["D"]) or not np.allclose(b["eta"], t["eta"]):
        raise RuntimeError("baseline and tensor maps are not aligned")

    candidate_name = str(t["case"].iloc[0])
    candidate_mode = str(t["gamma_mode_active"].iloc[0])
    candidate_policy = str(t["gamma_tensor_bound_policy"].iloc[0])
    detail = pd.DataFrame(
        {
            "candidate": candidate_name,
            "gamma_mode_active": candidate_mode,
            "gamma_tensor_bound_policy": candidate_policy,
            "D": b["D"],
            "eta": b["eta"],
            "R3_profile": b["R3"],
            "R3_candidate": t["R3"],
            "delta_R3": t["R3"].to_numpy(dtype=float) - b["R3"].to_numpy(dtype=float),
            "winner_profile": b["winner"],
            "winner_candidate": t["winner"],
            "mu_mumu_profile": b["mu_mumu"],
            "mu_mumu_candidate": t["mu_mumu"],
            "delta_mu_mumu": t["mu_mumu"].to_numpy(dtype=float) - b["mu_mumu"].to_numpy(dtype=float),
            "chi2_profile": b["chi2_mumu"],
            "chi2_candidate": t["chi2_mumu"],
            "accept_profile": b["hmumu_accept"],
            "accept_candidate": t["hmumu_accept"],
        }
    )
    abs_dmu = np.abs(detail["delta_mu_mumu"].to_numpy(dtype=float))
    abs_dr3 = np.abs(detail["delta_R3"].to_numpy(dtype=float))
    delta = {
        "row_type": "delta",
        "case": f"{candidate_name}_minus_profile",
        "gamma_mode_active": f"{candidate_mode}-vs-profile",
        "gamma_tensor_bound_policy": candidate_policy,
        "n_points": int(len(detail)),
        "acceptance_mismatch": float(np.mean(detail["accept_profile"].to_numpy(dtype=bool) != detail["accept_candidate"].to_numpy(dtype=bool))),
        "winner_mismatch": float(np.mean(detail["winner_profile"].to_numpy(dtype=int) != detail["winner_candidate"].to_numpy(dtype=int))),
        "max_abs_delta_mu_mumu": float(np.max(abs_dmu)),
        "p95_abs_delta_mu_mumu": float(np.percentile(abs_dmu, 95)),
        "max_abs_delta_R3": float(np.max(abs_dr3)),
        "p95_abs_delta_R3": float(np.percentile(abs_dr3, 95)),
    }
    return delta, detail


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit scan impact of A_l(D, eta, N) tensor lookup.")
    ap.add_argument("--d-num", type=int, default=60)
    ap.add_argument("--eta-num", type=int, default=21)
    ap.add_argument("--force-tensor", action="store_true")
    ap.add_argument("--copy-paper", action="store_true")
    args = ap.parse_args()

    d_scan = scan_d_values(4.0, 20.0, int(args.d_num))
    eta_scan = np.linspace(0.2, 4.0, int(args.eta_num))
    chi_profile = select_chi_profile(ROOT, d_scan)
    superrad_profile = select_superrad_profile(ROOT, d_scan)
    tensor_csv, tensor_summary = ensure_tensor_csv(d_scan, force=bool(args.force_tensor))

    cases = [
        EvalCase("profile_Nref2_baseline", str(superrad_profile["mode"]), Path(superrad_profile["path"])),
        EvalCase("tensor_N_resolved_strict", "action_tensor_grid_strict", tensor_csv),
        EvalCase("tensor_bound_only_strict", "action_tensor_grid_strict", tensor_csv, "bound_only_fallback"),
    ]
    maps = []
    summary_rows: List[Dict[str, float | int | str]] = []
    for case in cases:
        kin = make_kinetics(case, chi_profile)
        cur = evaluate_case(case, kin, d_scan, eta_scan)
        maps.append(cur)
        summary_rows.append(summarize_case(cur))

    detail_frames: List[pd.DataFrame] = []
    for candidate in maps[1:]:
        delta_row, detail = summarize_delta(maps[0], candidate)
        summary_rows.append(delta_row)
        detail_frames.append(detail)
    detail = pd.concat(detail_frames, ignore_index=True)

    tag = f"Dgrid{int(args.d_num)}_Egrid{int(args.eta_num)}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTDIR / f"channel_A_tensor_impact_{tag}_summary.csv"
    detail_path = OUTDIR / f"channel_A_tensor_impact_{tag}_detail.csv"
    meta_path = OUTDIR / f"channel_A_tensor_impact_{tag}_meta.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    pd.DataFrame(
        [
            {
                "tag": tag,
                "chi_profile_path": str(chi_profile["path"]),
                "chi_profile_mode": str(chi_profile["mode"]),
                "profile_superrad_path": str(superrad_profile["path"]),
                "profile_superrad_mode": str(superrad_profile["mode"]),
                "tensor_path": str(tensor_csv),
                "tensor_summary_path": str(tensor_summary),
            }
        ]
    ).to_csv(meta_path, index=False)

    print(f"[saved] {summary_path}")
    print(f"[saved] {detail_path}")
    print(f"[saved] {meta_path}")
    if args.copy_paper:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        for src in (summary_path, meta_path):
            dst = PAPER_DIR / src.name
            dst.write_text(src.read_text())
            print(f"[copied] {dst}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
