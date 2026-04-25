#!/usr/bin/env python3
"""
Outgoing Weyl transfer-condition audit for the Whittaker-DtN parent lift.

This is the theorem-facing follow-up to S10.  It does not solve for new roots.
For each already-audited S6 root, it treats the finite outgoing DtN map as a
Riccati transfer map

    T_{R,Rc}: m(Rc) -> m(R),

and measures:
  - the dyadic DtN displacement between Rc and 2Rc,
  - the local empirical transfer gain from the initial mismatch at Rc,
  - the infinitesimal Riccati condition number |dT/dm(Rc)|, and
  - the local Rouché budget inherited from S8.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_transfer_condition_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_transfer_condition_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_transfer_condition_budget.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from audit_static_width_siegert_extraction import OUTDIR, u_eff
from extract_static_width_exact_dtn import (
    ExactDtnConfig,
    exterior_dtn,
    outgoing_whittaker_log_derivative,
)
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_DETAIL = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_transfer_condition"


def parse_pairs(raw: str) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        left, right = item.split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def argument_lookup(argument: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in argument.iterrows():
        out[(float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))] = row
    return out


def riccati_transfer_condition(
    E: complex,
    D: float,
    ell: int,
    R_start: float,
    R: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[complex, complex, float, float]:
    m_start, _ = outgoing_whittaker_log_derivative(
        E=E,
        z=R_start,
        ell=ell,
        p=p,
        dps=cfg.whittaker_dps,
    )

    def rhs(z: float, y: np.ndarray) -> list[float]:
        m = complex(float(y[0]), float(y[1]))
        kappa = complex(float(y[2]), float(y[3]))
        U_z = float(u_eff(np.array([z]), D=D, ell=ell, p=p)[0])
        dm = U_z - E - m * m
        dkappa = -2.0 * m * kappa
        return [float(dm.real), float(dm.imag), float(dkappa.real), float(dkappa.imag)]

    sol = solve_ivp(
        rhs,
        t_span=(R_start, R),
        y0=np.array([m_start.real, m_start.imag, 1.0, 0.0], dtype=float),
        method="DOP853",
        rtol=cfg.dtn_rtol,
        atol=cfg.dtn_atol,
        max_step=cfg.dtn_max_step,
    )
    m_R = complex(float(sol.y[0, -1]), float(sol.y[1, -1]))
    kappa_R = complex(float(sol.y[2, -1]), float(sol.y[3, -1]))
    return m_R, kappa_R, float(abs(kappa_R)), float(sol.nfev)


def audit_row_pair(
    row: pd.Series,
    arg: pd.Series,
    pair: tuple[float, float],
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    D = float(row["D"])
    R = float(row["R"])
    R_tail = float(row["R_tail"])
    parity = str(row["parity"])
    ell = int(round(float(row["ell"])))
    h = float(row["h"])
    E0 = complex(float(row["E_dtn_real"]), float(row["E_dtn_imag"]))
    R_start, R_ref = pair
    pair_label = f"{R_start:g}_to_{R_ref:g}"
    cfg_row = replace(cfg, ell=ell)

    m_start_at_R, _ = exterior_dtn(E=E0, R=R, R_tail=R_start, D=D, ell=ell, cfg=cfg_row, p=p)
    m_ref_at_R, _ = exterior_dtn(E=E0, R=R, R_tail=R_ref, D=D, ell=ell, cfg=cfg_row, p=p)
    transfer_delta = float(abs(m_start_at_R - m_ref_at_R))

    m_start_tail, _ = outgoing_whittaker_log_derivative(
        E=E0,
        z=R_start,
        ell=ell,
        p=p,
        dps=cfg.whittaker_dps,
    )
    m_ref_at_start, _ = exterior_dtn(E=E0, R=R_start, R_tail=R_ref, D=D, ell=ell, cfg=cfg_row, p=p)
    initial_delta = float(abs(m_start_tail - m_ref_at_start))
    empirical_gain = transfer_delta / max(initial_delta, 1.0e-300)

    m_var_at_R, kappa_R, variational_gain, nfev = riccati_transfer_condition(
        E=E0,
        D=D,
        ell=ell,
        R_start=R_start,
        R=R,
        cfg=cfg_row,
        p=p,
    )
    variational_residual = float(abs(m_var_at_R - m_start_at_R))

    boundary_gap = float(arg["boundary_F_abs_min"])
    m_budget = 0.5 * h * boundary_gap
    allowed_gain = m_budget / max(initial_delta, 1.0e-300)
    linearized_delta = variational_gain * initial_delta

    return {
        "D": D,
        "R": R,
        "R_tail_root": R_tail,
        "parity": parity,
        "ell": float(ell),
        "h": h,
        "E0_real": float(E0.real),
        "E0_imag": float(E0.imag),
        "Gamma_dtn": float(row["Gamma_dtn"]),
        "transfer_pair": pair_label,
        "R_start": R_start,
        "R_ref": R_ref,
        "boundary_F_abs_min": boundary_gap,
        "m_rouche_budget": m_budget,
        "initial_delta_at_R_start": initial_delta,
        "transfer_delta_at_R": transfer_delta,
        "transfer_delta_over_budget": transfer_delta / max(m_budget, 1.0e-300),
        "empirical_transfer_gain": empirical_gain,
        "allowed_transfer_gain": allowed_gain,
        "empirical_gain_over_allowed": empirical_gain / max(allowed_gain, 1.0e-300),
        "variational_condition_abs": variational_gain,
        "variational_condition_log10": float(np.log10(max(variational_gain, 1.0e-300))),
        "linearized_delta": linearized_delta,
        "linearized_delta_over_budget": linearized_delta / max(m_budget, 1.0e-300),
        "empirical_over_variational": empirical_gain / max(variational_gain, 1.0e-300),
        "variational_kappa_real": float(kappa_R.real),
        "variational_kappa_imag": float(kappa_R.imag),
        "variational_nfev": nfev,
        "variational_m_residual": variational_residual,
        "empirical_transfer_status": "passes_empirical_transfer_budget"
        if transfer_delta <= m_budget
        else "not_lifted_empirical_transfer",
        "linear_transfer_status": "passes_linear_condition_budget"
        if linearized_delta <= m_budget
        else "not_lifted_linear_condition",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (pair, D), sub in detail.groupby(["transfer_pair", "D"], sort=True):
        rows.append(summary_row(pair=pair, D=D, sub=sub))
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        rows.append(summary_row(pair=pair, D="global", sub=sub))
    rows.append(summary_row(pair="all_pairs", D="global", sub=detail))
    return pd.DataFrame(rows)


def summary_row(pair: str, D: float | str, sub: pd.DataFrame) -> dict[str, float | str]:
    return {
        "transfer_pair": pair,
        "D": float(D) if isinstance(D, (float, int, np.floating)) else str(D),
        "n_rows": float(len(sub)),
        "n_empirical_transfer_pass": float((sub["empirical_transfer_status"] == "passes_empirical_transfer_budget").sum()),
        "n_linear_condition_pass": float((sub["linear_transfer_status"] == "passes_linear_condition_budget").sum()),
        "transfer_delta_over_budget_min": float(sub["transfer_delta_over_budget"].min()),
        "transfer_delta_over_budget_median": float(sub["transfer_delta_over_budget"].median()),
        "transfer_delta_over_budget_max": float(sub["transfer_delta_over_budget"].max()),
        "empirical_transfer_gain_min": float(sub["empirical_transfer_gain"].min()),
        "empirical_transfer_gain_max": float(sub["empirical_transfer_gain"].max()),
        "allowed_transfer_gain_min": float(sub["allowed_transfer_gain"].min()),
        "allowed_transfer_gain_max": float(sub["allowed_transfer_gain"].max()),
        "variational_condition_abs_min": float(sub["variational_condition_abs"].min()),
        "variational_condition_abs_max": float(sub["variational_condition_abs"].max()),
        "linearized_delta_over_budget_min": float(sub["linearized_delta_over_budget"].min()),
        "linearized_delta_over_budget_max": float(sub["linearized_delta_over_budget"].max()),
        "empirical_over_variational_max": float(sub["empirical_over_variational"].max()),
        "transfer_condition_status": "has_uniform_empirical_pass"
        if float((sub["empirical_transfer_status"] == "passes_empirical_transfer_budget").sum()) == len(sub)
        else "not_uniform_on_current_grid",
    }


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    colors = {"400_to_800": "#d95f02", "800_to_1600": "#1b9e77"}

    ax = axes[0]
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        x = np.arange(len(sub))
        ax.scatter(
            x,
            sub.sort_values("transfer_delta_over_budget")["transfer_delta_over_budget"],
            s=28,
            alpha=0.78,
            label=pair,
            color=colors.get(pair, None),
        )
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("rows sorted within each pair")
    ax.set_ylabel("dyadic transfer delta / local m-budget")
    ax.set_title("Dyadic Weyl transfer budget")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        ax.scatter(
            sub["empirical_transfer_gain"],
            sub["variational_condition_abs"],
            s=38,
            alpha=0.75,
            label=pair,
            color=colors.get(pair, None),
        )
    lims = [
        min(detail["empirical_transfer_gain"].min(), detail["variational_condition_abs"].min()),
        max(detail["empirical_transfer_gain"].max(), detail["variational_condition_abs"].max()),
    ]
    ax.plot(lims, lims, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("empirical finite transfer gain")
    ax.set_ylabel("infinitesimal condition number")
    ax.set_title("Linear condition is much more pessimistic")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--pairs", default="400:800,800:1600")
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    detail_in = pd.read_csv(args.detail)
    argument = pd.read_csv(args.argument_detail)
    arg_rows = argument_lookup(argument)
    p = PhysicalParams()
    cfg = ExactDtnConfig(tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    pairs = parse_pairs(args.pairs)

    rows: list[dict[str, float | str]] = []
    for _, row in detail_in.iterrows():
        arg = arg_rows[(float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))]
        for pair in pairs:
            rows.append(audit_row_pair(row=row, arg=arg, pair=pair, cfg=cfg, p=p))
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
