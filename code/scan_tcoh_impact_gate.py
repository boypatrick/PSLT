#!/usr/bin/env python3
"""
Release-style impact gate for the dephasing t_coh(D) candidate.

This audit keeps the current action-derived scan baseline fixed and changes
only the t_coh policy:
  - input_const_tcoh: t_coh = 1
  - uniform_dephasing_profile: 1D dephasing profile pi / Delta omega_12
  - uniform_dephasing_capped_1e4: same profile capped at 1e4
  - prolate_dephasing_profile: prolate-mesh cross-check profile, when present

The script is diagnostic only.  It does not change the production default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile
from hll_observable import HLLChannelPredictor, HLLObservableConfig
from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "tcoh_fp_1d"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
UNIFORM_TCOH_CSV = OUTDIR / "tcoh_dephasing_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
PROLATE_DETAIL_CSV = ROOT / "output" / "kinetic_action_chain" / "prolate_tcoh_propagation_audit_detail.csv"
PROLATE_TCOH_CSV = OUTDIR / "tcoh_dephasing_prolate_D4-20.csv"


@dataclass(frozen=True)
class EvalCase:
    name: str
    t_coh_mode: str
    t_coh_csv: Optional[Path] = None
    t_coh_cap: float = 1.0e4


def ensure_prolate_tcoh_csv() -> Optional[Path]:
    if not PROLATE_DETAIL_CSV.exists():
        return None
    detail = pd.read_csv(PROLATE_DETAIL_CSV)
    required = {"D", "prolate_t_coh_deph"}
    if not required.issubset(set(detail.columns)):
        return None
    out = pd.DataFrame(
        {
            "D": detail["D"].astype(float),
            "level": "fine",
            "t_coh_deph": detail["prolate_t_coh_deph"].astype(float),
        }
    ).sort_values("D")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROLATE_TCOH_CSV, index=False)
    return PROLATE_TCOH_CSV


def make_kinetics(case: EvalCase, chi_profile: Dict[str, object], superrad_profile: Dict[str, object]) -> PSLTKinetics:
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
        gamma_mode=str(superrad_profile["mode"]),
        gamma_superrad_csv=str(superrad_profile["path"]),
        t_coh_mode=str(case.t_coh_mode),
        t_coh_dephasing_csv=str(case.t_coh_csv) if case.t_coh_csv is not None else None,
        t_coh_cap=float(case.t_coh_cap),
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
    rows: List[Dict[str, float | int | str | bool]] = []
    for eta in eta_vals:
        for D in d_vals:
            _probs, _weights, meta = kin.get_probabilities(float(D), float(eta), 1.0, N_max=20)
            mu_pred = float(hll_mumu.mu_pred(float(D), float(eta)))
            chi2 = float(((mu_pred - 1.4) / 0.4) ** 2)
            rows.append(
                {
                    "case": case.name,
                    "t_coh_mode_requested": case.t_coh_mode,
                    "t_coh_mode_active": kin.active_t_coh_mode(),
                    "D": float(D),
                    "eta": float(eta),
                    "t_coh_eff": float(kin.t_coh_effective(float(D), 1.0)),
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
        "t_coh_mode_active": str(best["t_coh_mode_active"]),
        "n_points": int(len(df)),
        "t_coh_min": float(np.min(df["t_coh_eff"].to_numpy(dtype=float))),
        "t_coh_max": float(np.max(df["t_coh_eff"].to_numpy(dtype=float))),
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
    c = candidate.sort_values(["D", "eta"]).reset_index(drop=True)
    if len(b) != len(c) or not np.allclose(b["D"], c["D"]) or not np.allclose(b["eta"], c["eta"]):
        raise RuntimeError("baseline and t_coh maps are not aligned")

    candidate_name = str(c["case"].iloc[0])
    candidate_mode = str(c["t_coh_mode_active"].iloc[0])
    detail = pd.DataFrame(
        {
            "candidate": candidate_name,
            "t_coh_mode_active": candidate_mode,
            "D": b["D"],
            "eta": b["eta"],
            "t_coh_profile": b["t_coh_eff"],
            "t_coh_candidate": c["t_coh_eff"],
            "R3_profile": b["R3"],
            "R3_candidate": c["R3"],
            "delta_R3": c["R3"].to_numpy(dtype=float) - b["R3"].to_numpy(dtype=float),
            "winner_profile": b["winner"],
            "winner_candidate": c["winner"],
            "mu_mumu_profile": b["mu_mumu"],
            "mu_mumu_candidate": c["mu_mumu"],
            "delta_mu_mumu": c["mu_mumu"].to_numpy(dtype=float) - b["mu_mumu"].to_numpy(dtype=float),
            "chi2_profile": b["chi2_mumu"],
            "chi2_candidate": c["chi2_mumu"],
            "accept_profile": b["hmumu_accept"],
            "accept_candidate": c["hmumu_accept"],
        }
    )
    abs_dmu = np.abs(detail["delta_mu_mumu"].to_numpy(dtype=float))
    abs_dr3 = np.abs(detail["delta_R3"].to_numpy(dtype=float))
    delta = {
        "row_type": "delta",
        "case": f"{candidate_name}_minus_input",
        "t_coh_mode_active": f"{candidate_mode}-vs-input",
        "n_points": int(len(detail)),
        "t_coh_min": float(np.min(c["t_coh_eff"].to_numpy(dtype=float))),
        "t_coh_max": float(np.max(c["t_coh_eff"].to_numpy(dtype=float))),
        "acceptance_mismatch": float(np.mean(detail["accept_profile"].to_numpy(dtype=bool) != detail["accept_candidate"].to_numpy(dtype=bool))),
        "winner_mismatch": float(np.mean(detail["winner_profile"].to_numpy(dtype=int) != detail["winner_candidate"].to_numpy(dtype=int))),
        "max_abs_delta_mu_mumu": float(np.max(abs_dmu)),
        "p95_abs_delta_mu_mumu": float(np.percentile(abs_dmu, 95)),
        "max_abs_delta_R3": float(np.max(abs_dr3)),
        "p95_abs_delta_R3": float(np.percentile(abs_dr3, 95)),
    }
    return delta, detail


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit D/eta map impact of t_coh(D) profiles.")
    ap.add_argument("--d-num", type=int, default=60)
    ap.add_argument("--eta-num", type=int, default=21)
    ap.add_argument("--copy-paper", action="store_true")
    args = ap.parse_args()

    d_scan = scan_d_values(4.0, 20.0, int(args.d_num))
    eta_scan = np.linspace(0.2, 4.0, int(args.eta_num))
    chi_profile = select_chi_profile(ROOT, d_scan)
    superrad_profile = select_superrad_profile(ROOT, d_scan)
    prolate_csv = ensure_prolate_tcoh_csv()

    cases = [
        EvalCase("input_const_tcoh_1", "input"),
        EvalCase("uniform_dephasing_profile", "dephasing_profile", UNIFORM_TCOH_CSV),
        EvalCase("uniform_dephasing_capped_1e4", "dephasing_profile_capped", UNIFORM_TCOH_CSV, 1.0e4),
    ]
    if prolate_csv is not None:
        cases.append(EvalCase("prolate_dephasing_profile", "dephasing_profile", prolate_csv))
        cases.append(EvalCase("prolate_dephasing_capped_1e4", "dephasing_profile_capped", prolate_csv, 1.0e4))

    maps = []
    summary_rows: List[Dict[str, float | int | str]] = []
    for case in cases:
        kin = make_kinetics(case, chi_profile, superrad_profile)
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
    summary_path = OUTDIR / f"tcoh_impact_gate_{tag}_summary.csv"
    detail_path = OUTDIR / f"tcoh_impact_gate_{tag}_detail.csv"
    meta_path = OUTDIR / f"tcoh_impact_gate_{tag}_meta.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "chi_profile_path": str(chi_profile["path"]),
                "chi_profile_mode": str(chi_profile["mode"]),
                "superrad_profile_path": str(superrad_profile["path"]),
                "superrad_profile_mode": str(superrad_profile["mode"]),
                "uniform_tcoh_csv": str(UNIFORM_TCOH_CSV),
                "prolate_tcoh_csv": str(prolate_csv) if prolate_csv is not None else None,
                "baseline_case": "input_const_tcoh_1",
            },
            indent=2,
        )
    )

    print(f"[saved] {summary_path}")
    print(f"[saved] {detail_path}")
    print(f"[saved] {meta_path}")
    if args.copy_paper:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        for src in (summary_path, meta_path):
            dst = PAPER_DIR / src.name
            dst.write_bytes(src.read_bytes())
            print(f"[copied] {dst}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
