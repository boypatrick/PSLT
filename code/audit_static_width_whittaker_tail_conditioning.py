#!/usr/bin/env python3
"""
Tail-conditioning audit for the Whittaker-DtN parent lift.

S9 reduces the parent-lift obstruction to a Weyl-function remainder.  This
script separates two effects:

  1. the raw Coulomb-tail forcing |U_l - U_C| beyond the matching radius, and
  2. the Riccati transfer amplification that carries the tail mismatch back to
     the finite interface R.

No new roots or branches are searched.  The script evaluates the same
Whittaker-DtN map at R_tail=400 and R_tail=800 on each already-audited S6 root,
then compares the resulting DtN displacement with the S8 local Rouché budget.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_tail_conditioning_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_tail_conditioning_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_tail_conditioning_budget.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR, u_eff
from extract_static_width_exact_dtn import ExactDtnConfig, exterior_dtn
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_DETAIL = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_tail_conditioning"


def coulomb_tail_value(z: np.ndarray, ell: int, p: PhysicalParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    A = 4.0 * p.a * p.m0 * p.m0
    B = ell * (ell + 1.0) + 4.0 * p.a * p.a * p.m0 * p.m0
    return A / z + B / (z * z)


@lru_cache(maxsize=None)
def tail_l1_remainder(
    D: float,
    ell: int,
    R_tail: float,
    zmax: float,
    n_grid: int,
) -> tuple[float, float, float]:
    p = PhysicalParams()
    z = np.geomspace(float(R_tail), float(zmax), int(n_grid))
    rem = np.abs(u_eff(z, D=float(D), ell=int(ell), p=p) - coulomb_tail_value(z, ell=int(ell), p=p))
    integral = float(np.trapezoid(rem, z))
    # The leading remainder is O(z^-3).  The terminal coefficient gives a
    # conservative numerical tail completion for the audited large zmax.
    coeff_terminal = float(rem[-1] * z[-1] ** 3)
    tail_completion = coeff_terminal / (2.0 * z[-1] ** 2)
    return integral + tail_completion, coeff_terminal, tail_completion


def argument_lookup(argument: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in argument.iterrows():
        key = (float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))
        out[key] = row
    return out


def audit_row(
    row: pd.Series,
    arg_rows: dict[tuple[float, float, float, str], pd.Series],
    cfg: ExactDtnConfig,
    p: PhysicalParams,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    D = float(row["D"])
    R = float(row["R"])
    R_tail = float(row["R_tail"])
    parity = str(row["parity"])
    ell = int(round(float(row.get("ell", cfg.ell))))
    h = float(row["h"])
    E0 = complex(float(row["E_dtn_real"]), float(row["E_dtn_imag"]))

    cfg_row = replace(cfg, ell=ell, whittaker_dps=int(args.whittaker_dps))
    m400, meta400 = exterior_dtn(E=E0, R=R, R_tail=400.0, D=D, ell=ell, cfg=cfg_row, p=p)
    m800, meta800 = exterior_dtn(E=E0, R=R, R_tail=800.0, D=D, ell=ell, cfg=cfg_row, p=p)
    dtn_delta = float(abs(m400 - m800))
    residual_delta = float((2.0 / h) * dtn_delta)

    arg = arg_rows[(D, R, R_tail, parity)]
    boundary_gap = float(arg["boundary_F_abs_min"])
    m_rouche_budget = float(0.5 * h * boundary_gap)
    dtn_delta_over_budget = float(dtn_delta / max(m_rouche_budget, 1.0e-300))
    residual_delta_over_gap = float(residual_delta / max(boundary_gap, 1.0e-300))

    tail_l1_self, coeff_self, tail_completion_self = tail_l1_remainder(
        D=D,
        ell=ell,
        R_tail=R_tail,
        zmax=float(args.zmax),
        n_grid=int(args.tail_grid),
    )
    tail_l1_400, coeff_400, _ = tail_l1_remainder(
        D=D,
        ell=ell,
        R_tail=400.0,
        zmax=float(args.zmax),
        n_grid=int(args.tail_grid),
    )
    tail_l1_800, coeff_800, _ = tail_l1_remainder(
        D=D,
        ell=ell,
        R_tail=800.0,
        zmax=float(args.zmax),
        n_grid=int(args.tail_grid),
    )
    tail_l1_pair = max(tail_l1_400, tail_l1_800)
    raw_tail_over_budget = float(tail_l1_self / max(m_rouche_budget, 1.0e-300))
    pair_tail_over_budget = float(tail_l1_pair / max(m_rouche_budget, 1.0e-300))
    empirical_conditioning_lower = float(dtn_delta / max(tail_l1_pair, 1.0e-300))
    allowed_conditioning_self = float(m_rouche_budget / max(tail_l1_self, 1.0e-300))
    allowed_conditioning_pair = float(m_rouche_budget / max(tail_l1_pair, 1.0e-300))

    return {
        "D": D,
        "R": R,
        "R_tail": R_tail,
        "parity": parity,
        "ell": float(ell),
        "h": h,
        "E0_real": float(E0.real),
        "E0_imag": float(E0.imag),
        "Gamma_dtn": float(row["Gamma_dtn"]),
        "boundary_F_abs_min": boundary_gap,
        "m_rouche_budget": m_rouche_budget,
        "m400_real": float(m400.real),
        "m400_imag": float(m400.imag),
        "m800_real": float(m800.real),
        "m800_imag": float(m800.imag),
        "m400_dtn_success": float(meta400["dtn_success"]),
        "m800_dtn_success": float(meta800["dtn_success"]),
        "dtn_delta_400_800": dtn_delta,
        "residual_delta_400_800": residual_delta,
        "dtn_delta_over_m_budget": dtn_delta_over_budget,
        "residual_delta_over_boundary_gap": residual_delta_over_gap,
        "tail_l1_self": tail_l1_self,
        "tail_l1_400": tail_l1_400,
        "tail_l1_800": tail_l1_800,
        "tail_l1_pair_max": tail_l1_pair,
        "tail_coeff_self_terminal": coeff_self,
        "tail_coeff_400_terminal": coeff_400,
        "tail_coeff_800_terminal": coeff_800,
        "tail_completion_self": tail_completion_self,
        "raw_tail_over_m_budget": raw_tail_over_budget,
        "pair_tail_over_m_budget": pair_tail_over_budget,
        "empirical_conditioning_lower_bound": empirical_conditioning_lower,
        "allowed_conditioning_self": allowed_conditioning_self,
        "allowed_conditioning_pair": allowed_conditioning_pair,
        "tail_conditioning_status": "passes_tail_conditioning_budget"
        if dtn_delta_over_budget <= 1.0
        else "not_lifted_tail_conditioning",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        rows.append(
            {
                "D": float(D),
                "n_rows": float(len(sub)),
                "n_tail_conditioning_pass": float((sub["tail_conditioning_status"] == "passes_tail_conditioning_budget").sum()),
                "m_rouche_budget_min": float(sub["m_rouche_budget"].min()),
                "m_rouche_budget_max": float(sub["m_rouche_budget"].max()),
                "tail_l1_self_max": float(sub["tail_l1_self"].max()),
                "raw_tail_over_m_budget_max": float(sub["raw_tail_over_m_budget"].max()),
                "dtn_delta_400_800_min": float(sub["dtn_delta_400_800"].min()),
                "dtn_delta_400_800_max": float(sub["dtn_delta_400_800"].max()),
                "dtn_delta_over_m_budget_min": float(sub["dtn_delta_over_m_budget"].min()),
                "dtn_delta_over_m_budget_max": float(sub["dtn_delta_over_m_budget"].max()),
                "empirical_conditioning_lower_bound_min": float(sub["empirical_conditioning_lower_bound"].min()),
                "empirical_conditioning_lower_bound_max": float(sub["empirical_conditioning_lower_bound"].max()),
                "allowed_conditioning_pair_min": float(sub["allowed_conditioning_pair"].min()),
                "allowed_conditioning_pair_max": float(sub["allowed_conditioning_pair"].max()),
                "tail_conditioning_status": "not_lifted_current_tail_conditioning"
                if float((sub["tail_conditioning_status"] == "passes_tail_conditioning_budget").sum()) == 0
                else "has_tail_conditioning_pass",
            }
        )
    rows.append(
        {
            "D": "global",
            "n_rows": float(len(detail)),
            "n_tail_conditioning_pass": float((detail["tail_conditioning_status"] == "passes_tail_conditioning_budget").sum()),
            "m_rouche_budget_min": float(detail["m_rouche_budget"].min()),
            "m_rouche_budget_max": float(detail["m_rouche_budget"].max()),
            "tail_l1_self_max": float(detail["tail_l1_self"].max()),
            "raw_tail_over_m_budget_max": float(detail["raw_tail_over_m_budget"].max()),
            "dtn_delta_400_800_min": float(detail["dtn_delta_400_800"].min()),
            "dtn_delta_400_800_max": float(detail["dtn_delta_400_800"].max()),
            "dtn_delta_over_m_budget_min": float(detail["dtn_delta_over_m_budget"].min()),
            "dtn_delta_over_m_budget_max": float(detail["dtn_delta_over_m_budget"].max()),
            "empirical_conditioning_lower_bound_min": float(detail["empirical_conditioning_lower_bound"].min()),
            "empirical_conditioning_lower_bound_max": float(detail["empirical_conditioning_lower_bound"].max()),
            "allowed_conditioning_pair_min": float(detail["allowed_conditioning_pair"].min()),
            "allowed_conditioning_pair_max": float(detail["allowed_conditioning_pair"].max()),
            "tail_conditioning_status": "not_lifted_current_tail_conditioning"
            if float((detail["tail_conditioning_status"] == "passes_tail_conditioning_budget").sum()) == 0
            else "has_tail_conditioning_pass",
        }
    )
    return pd.DataFrame(rows)


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    colors = {6.0: "#1b9e77", 12.0: "#d95f02", 18.0: "#7570b3"}

    ax = axes[0]
    for D, sub in detail.groupby("D", sort=True):
        ax.scatter(
            sub["raw_tail_over_m_budget"],
            sub["dtn_delta_over_m_budget"],
            s=42,
            alpha=0.8,
            label=f"D={D:g}",
            color=colors.get(float(D), None),
        )
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.axvline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("raw tail L1 / local m-budget")
    ax.set_ylabel(r"$|m_{400}-m_{800}|$ / local m-budget")
    ax.set_title("Raw tail is small; transfer is not")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    grouped = detail.groupby("D", sort=True)["dtn_delta_over_m_budget"].min()
    ax.bar([str(int(k)) for k in grouped.index], grouped.to_numpy(dtype=float), color="#586e75")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("best DtN delta / local m-budget")
    ax.set_title("Best tail-conditioning gap")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--zmax", type=float, default=1.0e6)
    ap.add_argument("--tail-grid", type=int, default=4096)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    detail_in = pd.read_csv(args.detail)
    argument = pd.read_csv(args.argument_detail)
    arg_rows = argument_lookup(argument)
    p = PhysicalParams()
    cfg = ExactDtnConfig(tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))

    rows = [audit_row(row, arg_rows=arg_rows, cfg=cfg, p=p, args=args) for _, row in detail_in.iterrows()]
    detail = pd.DataFrame(rows)
    summary = summarize(detail)

    detail_path = OUTDIR / f"{args.out_tag}_detail.csv"
    summary_path = OUTDIR / f"{args.out_tag}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))

    if not args.skip_plot:
        plot_path = OUTDIR / f"{args.out_tag}_budget.png"
        plot_budget(detail, plot_path)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
