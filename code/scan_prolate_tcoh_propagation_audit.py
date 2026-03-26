#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_auxiliary_extraction_audit import (
    OUTDIR as KIN_OUTDIR,
    PAPER_DIR,
    P,
    Z_MAX,
    N_TOTAL,
    build_parametrized_prolate_axis_mesh,
    solve_nonuniform_bound_states,
)
from scan_tcoh_profile_impact import Case, evaluate_case, make_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
STEM = "prolate_tcoh_propagation_audit"

TCOH_UNIFORM_PATH = ROOT / "output" / "tcoh_fp_1d" / "tcoh_dephasing_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
CHI_LOCALIZED_PATH = ROOT / "output" / "chi_fp_2d" / "localized_chi_D6-12-18.csv"
GN_PATH = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_D6-12-18.csv"

D_FULL = np.arange(4.0, 21.0, 1.0)
D_WITNESS = np.array([6.0, 12.0, 18.0], dtype=float)

CALIBRATED_PROFILE = {"n_inner": 401, "eta_power": 0.75, "xi_power": 2.0}


def _rel_err(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    cur = np.asarray(cur, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return np.abs(cur - ref) / np.maximum(np.abs(ref), 1e-30)


def _anchor_ratio(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / values[0]


def _shape_l1(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float))))


def _solve_prolate_tcoh_profile() -> pd.DataFrame:
    rows = []
    for D in D_FULL:
        z = build_parametrized_prolate_axis_mesh(
            float(D),
            Z_MAX,
            N_TOTAL,
            int(CALIBRATED_PROFILE["n_inner"]),
            float(CALIBRATED_PROFILE["eta_power"]),
            float(CALIBRATED_PROFILE["xi_power"]),
        )
        evals, omega = solve_nonuniform_bound_states(z, float(D), P, n_states=3)
        delta = float(omega[1] - omega[0])
        rows.append(
            {
                "D": float(D),
                "prolate_E1": float(evals[0]),
                "prolate_E2": float(evals[1]),
                "prolate_omega1": float(omega[0]),
                "prolate_omega2": float(omega[1]),
                "prolate_delta_omega12": delta,
                "prolate_t_coh_deph": float(np.pi / max(abs(delta), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _load_uniform_tcoh_profile() -> pd.DataFrame:
    df = pd.read_csv(TCOH_UNIFORM_PATH)
    df = df.loc[df["level"].str.lower() == "fine"].copy()
    return df[["D", "omega1", "omega2", "delta_omega12", "t_coh_deph"]].rename(
        columns={
            "omega1": "uniform_omega1",
            "omega2": "uniform_omega2",
            "delta_omega12": "uniform_delta_omega12",
            "t_coh_deph": "uniform_t_coh_deph",
        }
    )


def _load_witness() -> pd.DataFrame:
    chi = pd.read_csv(CHI_LOCALIZED_PATH)
    chi = chi.loc[chi["level"].str.lower() == "fine", ["D", "chi_LR", "Gamma_ref"]].copy()
    gn = pd.read_csv(GN_PATH)
    gn = gn.loc[gn["level"].str.lower() == "fine", ["D", "g2_raw", "g3_raw"]].copy()
    return chi.merge(gn, on="D", how="inner").sort_values("D").reset_index(drop=True)


def _build_detail() -> pd.DataFrame:
    uniform = _load_uniform_tcoh_profile()
    prolate = _solve_prolate_tcoh_profile()
    detail = uniform.merge(prolate, on="D", how="inner").sort_values("D").reset_index(drop=True)
    detail["rel_delta_bias_vs_uniform"] = _rel_err(detail["prolate_delta_omega12"], detail["uniform_delta_omega12"])
    detail["rel_tcoh_bias_vs_uniform"] = _rel_err(detail["prolate_t_coh_deph"], detail["uniform_t_coh_deph"])
    return detail


def _interp_profile(df: pd.DataFrame, value_col: str):
    d = df["D"].to_numpy(dtype=float)
    v = df[value_col].to_numpy(dtype=float)

    def fn(D: float) -> float:
        return float(np.interp(D, d, v))

    return fn


def _build_summary(detail: pd.DataFrame, witness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "summary_type": "profile_bias",
            "metric": "delta_omega12",
            "uniform_reference": "uniform_fine_tcoh_profile",
            "p95_rel_bias": float(np.percentile(detail["rel_delta_bias_vs_uniform"], 95.0)),
            "max_rel_bias": float(np.max(detail["rel_delta_bias_vs_uniform"])),
        }
    )
    rows.append(
        {
            "summary_type": "profile_bias",
            "metric": "t_coh_deph",
            "uniform_reference": "uniform_fine_tcoh_profile",
            "p95_rel_bias": float(np.percentile(detail["rel_tcoh_bias_vs_uniform"], 95.0)),
            "max_rel_bias": float(np.max(detail["rel_tcoh_bias_vs_uniform"])),
        }
    )

    witness_detail = detail.loc[detail["D"].isin(D_WITNESS)].merge(witness, on="D", how="left").sort_values("D")
    u_delta = _anchor_ratio(witness_detail["uniform_delta_omega12"].to_numpy(dtype=float))
    p_delta = _anchor_ratio(witness_detail["prolate_delta_omega12"].to_numpy(dtype=float))
    chi_ratio = _anchor_ratio(witness_detail["chi_LR"].to_numpy(dtype=float))
    g2_ratio = _anchor_ratio(witness_detail["g2_raw"].to_numpy(dtype=float))
    g3_ratio = _anchor_ratio(witness_detail["g3_raw"].to_numpy(dtype=float))

    for witness_name, witness_ratio in [("chi_LR", chi_ratio), ("g2_raw", g2_ratio), ("g3_raw", g3_ratio)]:
        rows.append(
            {
                "summary_type": "trend_witness",
                "metric": witness_name,
                "uniform_anchor_l1": _shape_l1(u_delta, witness_ratio),
                "prolate_anchor_l1": _shape_l1(p_delta, witness_ratio),
                "better_method": "prolate_aux" if _shape_l1(p_delta, witness_ratio) < _shape_l1(u_delta, witness_ratio) else "uniform_aux",
            }
        )

    kin = make_kinetics()
    impact_rows = [
        evaluate_case(Case(name="constant_tcoh_1", mode="constant"), kin, _interp_profile(detail, "uniform_t_coh_deph")),
        evaluate_case(Case(name="uniform_dephasing_profile", mode="dephasing"), kin, _interp_profile(detail, "uniform_t_coh_deph")),
        evaluate_case(Case(name="prolate_dephasing_profile", mode="dephasing"), kin, _interp_profile(detail, "prolate_t_coh_deph")),
        evaluate_case(Case(name="prolate_dephasing_capped_1e4", mode="dephasing_capped", t_cap=1.0e4), kin, _interp_profile(detail, "prolate_t_coh_deph")),
    ]
    for row in impact_rows:
        row["summary_type"] = "map_impact"
        rows.append(row)

    return pd.DataFrame(rows)


def _build_slices(detail: pd.DataFrame, witness: pd.DataFrame) -> pd.DataFrame:
    witness_detail = detail.loc[detail["D"].isin(D_WITNESS)].merge(witness, on="D", how="left").sort_values("D")
    return witness_detail[
        [
            "D",
            "uniform_omega1",
            "prolate_omega1",
            "uniform_omega2",
            "prolate_omega2",
            "uniform_delta_omega12",
            "prolate_delta_omega12",
            "uniform_t_coh_deph",
            "prolate_t_coh_deph",
            "chi_LR",
            "Gamma_ref",
            "g2_raw",
            "g3_raw",
            "rel_delta_bias_vs_uniform",
            "rel_tcoh_bias_vs_uniform",
        ]
    ].copy()


def _plot(detail: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    x = detail["D"].to_numpy(dtype=float)

    ax = axes[0, 0]
    ax.plot(x, detail["uniform_delta_omega12"], lw=2, label="uniform Δω12")
    ax.plot(x, detail["prolate_delta_omega12"], lw=2, label="prolate Δω12")
    ax.set_title("Delta omega12 profile")
    ax.set_xlabel("D")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, detail["uniform_t_coh_deph"], lw=2, label="uniform t_coh")
    ax.plot(x, detail["prolate_t_coh_deph"], lw=2, label="prolate t_coh")
    ax.set_title("Dephasing t_coh profile")
    ax.set_xlabel("D")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, detail["rel_delta_bias_vs_uniform"], lw=2)
    ax.set_title("Relative delta bias vs uniform")
    ax.set_xlabel("D")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(x, detail["rel_tcoh_bias_vs_uniform"], lw=2)
    ax.set_title("Relative t_coh bias vs uniform")
    ax.set_xlabel("D")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    detail = _build_detail()
    witness = _load_witness()
    summary = _build_summary(detail, witness)
    slices = _build_slices(detail, witness)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(detail, png_path)

    meta_path.write_text(
        json.dumps(
            {
                "D_full": [float(x) for x in D_FULL],
                "D_witness": [float(x) for x in D_WITNESS],
                "calibrated_profile": CALIBRATED_PROFILE,
                "uniform_profile_csv": str(TCOH_UNIFORM_PATH),
                "chi_witness_csv": str(CHI_LOCALIZED_PATH),
                "gn_witness_csv": str(GN_PATH),
            },
            indent=2,
        )
    )

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
