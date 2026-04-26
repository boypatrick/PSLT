#!/usr/bin/env python3
"""
Cauchy/slab derivative bound for the final Whittaker-DtN flowbox.

S22 showed that a 5% inflated upper-Darboux envelope closes the final
D=6, R_tail=400, R=60->80 parent tube.  This S23 audit turns that inflation
into the explicit analytic derivative condition

    (Delta R_j / 2) L_j <= 0.05 M_j,

where M_j is the endpoint node upper bound for |G| and L_j bounds the fixed-z
R derivative of

    G(R,z) = (F_R(R,gamma(R)+z) + gamma'(R) F_E(R,gamma(R)+z))
             / F_E(R,gamma(R)+z).

If the condition holds on each slab, then the 5% S22 inflated envelope is a
true slab envelope.  The numerical part below samples G_R at left/mid/right
slab stations and applies a configurable Cauchy guard factor.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_adapted_flow import (
    derivative_estimates,
    make_eval,
    parse_floats,
    parse_strings,
    sample_points,
)
from audit_static_width_whittaker_curved_flowbox import polynomial_centerline
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_S22_SLABS = OUTDIR / "static_width_whittaker_slab_envelope_d6rc400_r60_r80_c640_slabs.csv"
DEFAULT_TAG = "static_width_whittaker_cauchy_slab_bound_d6rc400_r60_r80_c640"


def parse_segments(raw: str) -> tuple[tuple[float, float], ...]:
    if raw.strip() == "":
        return ()
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        out.append((float(left), float(right)))
    return tuple(out)


class HFlowEvaluator:
    def __init__(
        self,
        D: float,
        R_tail: float,
        parity: str,
        n_half: int,
        ell: int,
        gamma: callable,
        gamma_prime: callable,
        cfg: ExactDtnConfig,
        p: PhysicalParams,
        dR_step: float,
        dE_step: float,
    ) -> None:
        self.D = float(D)
        self.R_tail = float(R_tail)
        self.parity = str(parity)
        self.n_half = int(n_half)
        self.ell = int(ell)
        self.gamma = gamma
        self.gamma_prime = gamma_prime
        self.cfg = cfg
        self.p = p
        self.dR_step = float(dR_step)
        self.dE_step = float(dE_step)

    @lru_cache(maxsize=None)
    def _eval_triplet(self, R_key: float) -> tuple[callable, callable, callable]:
        R = float(R_key)
        return (
            make_eval(self.D, R - self.dR_step, self.R_tail, self.parity, self.n_half, self.ell, self.cfg, self.p),
            make_eval(self.D, R, self.R_tail, self.parity, self.n_half, self.ell, self.cfg, self.p),
            make_eval(self.D, R + self.dR_step, self.R_tail, self.parity, self.n_half, self.ell, self.cfg, self.p),
        )

    def H(self, R: float, z: complex) -> tuple[complex, float, float, float]:
        R = float(R)
        eval_minus, eval_mid, eval_plus = self._eval_triplet(round(R, 12))
        E = self.gamma(R) + z
        deriv = derivative_estimates(
            E=complex(E),
            eval_minus=eval_minus,
            eval_mid=eval_mid,
            eval_plus=eval_plus,
            dR_step=self.dR_step,
            dE_step=self.dE_step,
        )
        dF_dR = complex(deriv["dF_dR"])
        dF_dE = complex(deriv["dF_dE"])
        denom = max(abs(dF_dE), 1.0e-300)
        H_val = (dF_dR + self.gamma_prime(R) * dF_dE) / dF_dE
        return (
            complex(H_val),
            float(denom),
            float(abs(dF_dR + self.gamma_prime(R) * dF_dE)),
            float(deriv["dF_dE_CR_rel_mismatch"]),
        )

    def dH_dR_abs(self, R: float, z: complex, cauchy_R_step: float) -> dict[str, float]:
        h = float(cauchy_R_step)
        if R - h - self.dR_step <= 0.0 or R + h + self.dR_step >= self.R_tail:
            raise ValueError(f"Bad Cauchy R stencil: R={R}, h={h}, R_tail={self.R_tail}")
        H_minus, den_minus, num_minus, cr_minus = self.H(R - h, z)
        H_plus, den_plus, num_plus, cr_plus = self.H(R + h, z)
        dH = (H_plus - H_minus) / (2.0 * h)
        return {
            "dH_dR_abs": float(abs(dH)),
            "H_minus_abs": float(abs(H_minus)),
            "H_plus_abs": float(abs(H_plus)),
            "dF_dE_abs_min": float(min(den_minus, den_plus)),
            "curved_numerator_abs_max": float(max(num_minus, num_plus)),
            "dF_dE_CR_rel_mismatch_max": float(max(cr_minus, cr_plus)),
        }


def audit_parity(
    path: pd.DataFrame,
    slab_rows: pd.DataFrame,
    center_mode: str,
    tube_factor: float,
    n_per_side: int,
    dR_step: float,
    dE_step_fraction: float,
    cauchy_R_step_fraction: float,
    cauchy_guard_factor: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    path = path.sort_values("R").reset_index(drop=True)
    parity = str(path.iloc[0]["parity"])
    D = float(path.iloc[0]["D"])
    R_tail = float(path.iloc[0]["R_tail"])
    ell = int(round(float(path.iloc[0].get("ell", cfg.ell))))
    n_half = int(round(float(path.iloc[0].get("n_half", cfg.n_half))))
    R0 = float(slab_rows["R0"].iloc[0])
    R1 = float(slab_rows["R1"].iloc[0])
    gamma, gamma_prime, anchors = polynomial_centerline(path=path, R0=R0, R1=R1, mode=center_mode)

    radius = float(slab_rows["tube_radius_min"].iloc[0])
    dE_step = max(radius * float(dE_step_fraction), 1.0e-8)
    h_slab = float(slab_rows["delta_R"].iloc[0])
    cauchy_R_step = h_slab * float(cauchy_R_step_fraction)
    evaluator = HFlowEvaluator(
        D=D,
        R_tail=R_tail,
        parity=parity,
        n_half=n_half,
        ell=ell,
        gamma=gamma,
        gamma_prime=gamma_prime,
        cfg=cfg,
        p=p,
        dR_step=dR_step,
        dE_step=dE_step,
    )

    check_rows: list[dict[str, float | str]] = []
    slab_out: list[dict[str, float | str]] = []
    for _, slab in slab_rows.sort_values("slab_index").iterrows():
        R_left = float(slab["R_left"])
        R_right = float(slab["R_right"])
        R_mid = 0.5 * (R_left + R_right)
        M_j = float(slab["node_upper_beta"])
        reserve = 0.05 * M_j
        sample_max = 0.0
        den_min = np.inf
        num_max = 0.0
        cr_max = 0.0

        for station_name, R in (("left", R_left), ("mid", R_mid), ("right", R_right)):
            center = gamma(R)
            pts = sample_points(center=center, half_re=radius, half_im=radius, n_per_side=n_per_side, include_center=True)
            for idx, E in enumerate(pts):
                z = complex(E - center)
                vals = evaluator.dH_dR_abs(R=R, z=z, cauchy_R_step=cauchy_R_step)
                sample_max = max(sample_max, vals["dH_dR_abs"])
                den_min = min(den_min, vals["dF_dE_abs_min"])
                num_max = max(num_max, vals["curved_numerator_abs_max"])
                cr_max = max(cr_max, vals["dF_dE_CR_rel_mismatch_max"])
                check_rows.append(
                    {
                        "D": D,
                        "R_tail": R_tail,
                        "parity": parity,
                        "R0": R0,
                        "R1": R1,
                        "center_mode": center_mode,
                        "center_anchors": ";".join(f"{x:g}" for x in anchors),
                        "tube_factor": float(tube_factor),
                        "tube_radius_min": radius,
                        "slab_index": float(slab["slab_index"]),
                        "station": station_name,
                        "R": float(R),
                        "point_index": float(idx),
                        "z_real": float(z.real),
                        "z_imag": float(z.imag),
                        **vals,
                    }
                )

        L_bound = float(cauchy_guard_factor) * sample_max
        remainder = 0.5 * h_slab * L_bound
        beta_cauchy = M_j + remainder
        slab_out.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_factor": float(tube_factor),
                "tube_radius_min": radius,
                "slab_index": float(slab["slab_index"]),
                "R_left": R_left,
                "R_right": R_right,
                "delta_R": h_slab,
                "node_upper_beta": M_j,
                "sampled_dH_dR_abs_max": sample_max,
                "cauchy_guard_factor": float(cauchy_guard_factor),
                "cauchy_L_bound": L_bound,
                "five_percent_reserve": reserve,
                "cauchy_remainder_bound": remainder,
                "remainder_over_reserve": remainder / max(reserve, 1.0e-300),
                "beta_cauchy": beta_cauchy,
                "beta_s22_inflated": float(slab["inflated_beta"]),
                "beta_cauchy_over_s22_inflated": beta_cauchy / max(float(slab["inflated_beta"]), 1.0e-300),
                "cauchy_contribution": h_slab * beta_cauchy,
                "s22_inflated_contribution": float(slab["inflated_contribution"]),
                "dF_dE_abs_min": float(den_min),
                "curved_numerator_abs_max": float(num_max),
                "dF_dE_CR_rel_mismatch_max": float(cr_max),
                "cauchy_slab_pass": float(beta_cauchy <= float(slab["inflated_beta"])),
            }
        )

    check_df = pd.DataFrame(check_rows)
    slab_df = pd.DataFrame(slab_out)
    cauchy_ratio = float(slab_df["cauchy_contribution"].sum() / radius)
    inflated_ratio = float(slab_df["s22_inflated_contribution"].sum() / radius)
    summary = {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "tube_factor": float(tube_factor),
        "tube_radius_min": radius,
        "n_slabs": float(len(slab_df)),
        "n_cauchy_check_points": float(len(check_df)),
        "cauchy_R_step": float(cauchy_R_step),
        "dR_step": float(dR_step),
        "dE_step": float(dE_step),
        "cauchy_guard_factor": float(cauchy_guard_factor),
        "max_sampled_dH_dR_abs": float(slab_df["sampled_dH_dR_abs_max"].max()),
        "max_cauchy_L_bound": float(slab_df["cauchy_L_bound"].max()),
        "max_remainder_over_reserve": float(slab_df["remainder_over_reserve"].max()),
        "median_remainder_over_reserve": float(slab_df["remainder_over_reserve"].median()),
        "max_beta_cauchy_over_s22_inflated": float(slab_df["beta_cauchy_over_s22_inflated"].max()),
        "cauchy_integral_ratio": cauchy_ratio,
        "s22_inflated_integral_ratio": inflated_ratio,
        "min_dF_dE_abs": float(slab_df["dF_dE_abs_min"].min()),
        "max_dF_dE_CR_rel_mismatch": float(slab_df["dF_dE_CR_rel_mismatch_max"].max()),
        "n_cauchy_slab_pass": float(slab_df["cauchy_slab_pass"].sum()),
        "cauchy_slab_status": "cauchy_slab_bound_closed"
        if float(slab_df["cauchy_slab_pass"].sum()) == len(slab_df) and cauchy_ratio < 1.0
        else "cauchy_slab_bound_not_closed",
    }
    return check_df, slab_df, summary


def audit_node_slope_parity(
    slab_rows: pd.DataFrame,
    center_mode: str,
    tube_factor: float,
    cauchy_guard_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Fast S23 gate using the S22 node slopes as the Cauchy derivative proxy.

    The parent theorem is stated with the exact derivative envelope L_j.  This
    audit records the concrete sufficient numerical target L_j <= guard *
    node_slope_abs and checks that this guarded value still fits inside the
    5% S22 reserve.
    """

    slab_rows = slab_rows.sort_values("slab_index").reset_index(drop=True)
    D = float(slab_rows["D"].iloc[0])
    R_tail = float(slab_rows["R_tail"].iloc[0])
    parity = str(slab_rows["parity"].iloc[0])
    R0 = float(slab_rows["R0"].iloc[0])
    R1 = float(slab_rows["R1"].iloc[0])
    radius = float(slab_rows["tube_radius_min"].iloc[0])
    check_rows: list[dict[str, float | str]] = []
    slab_out: list[dict[str, float | str]] = []

    for _, slab in slab_rows.iterrows():
        h_slab = float(slab["delta_R"])
        M_j = float(slab["node_upper_beta"])
        reserve = 0.05 * M_j
        observed_slope = float(slab["node_slope_abs"])
        L_bound = float(cauchy_guard_factor) * observed_slope
        remainder = 0.5 * h_slab * L_bound
        beta_cauchy = M_j + remainder
        check_rows.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_factor": float(tube_factor),
                "tube_radius_min": radius,
                "slab_index": float(slab["slab_index"]),
                "station": "node_slope_proxy",
                "R": 0.5 * (float(slab["R_left"]) + float(slab["R_right"])),
                "point_index": 0.0,
                "z_real": 0.0,
                "z_imag": 0.0,
                "dH_dR_abs": observed_slope,
                "H_minus_abs": float(slab["q_left"]),
                "H_plus_abs": float(slab["q_right"]),
                "dF_dE_abs_min": np.nan,
                "curved_numerator_abs_max": np.nan,
                "dF_dE_CR_rel_mismatch_max": np.nan,
            }
        )
        slab_out.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_factor": float(tube_factor),
                "tube_radius_min": radius,
                "slab_index": float(slab["slab_index"]),
                "R_left": float(slab["R_left"]),
                "R_right": float(slab["R_right"]),
                "delta_R": h_slab,
                "node_upper_beta": M_j,
                "sampled_dH_dR_abs_max": observed_slope,
                "cauchy_guard_factor": float(cauchy_guard_factor),
                "cauchy_L_bound": L_bound,
                "five_percent_reserve": reserve,
                "cauchy_remainder_bound": remainder,
                "remainder_over_reserve": remainder / max(reserve, 1.0e-300),
                "beta_cauchy": beta_cauchy,
                "beta_s22_inflated": float(slab["inflated_beta"]),
                "beta_cauchy_over_s22_inflated": beta_cauchy / max(float(slab["inflated_beta"]), 1.0e-300),
                "cauchy_contribution": h_slab * beta_cauchy,
                "s22_inflated_contribution": float(slab["inflated_contribution"]),
                "dF_dE_abs_min": np.nan,
                "curved_numerator_abs_max": np.nan,
                "dF_dE_CR_rel_mismatch_max": np.nan,
                "cauchy_slab_pass": float(beta_cauchy <= float(slab["inflated_beta"])),
            }
        )

    check_df = pd.DataFrame(check_rows)
    slab_df = pd.DataFrame(slab_out)
    cauchy_ratio = float(slab_df["cauchy_contribution"].sum() / radius)
    inflated_ratio = float(slab_df["s22_inflated_contribution"].sum() / radius)
    summary = {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "tube_factor": float(tube_factor),
        "tube_radius_min": radius,
        "n_slabs": float(len(slab_df)),
        "n_cauchy_check_points": float(len(check_df)),
        "cauchy_R_step": np.nan,
        "dR_step": np.nan,
        "dE_step": np.nan,
        "cauchy_guard_factor": float(cauchy_guard_factor),
        "max_sampled_dH_dR_abs": float(slab_df["sampled_dH_dR_abs_max"].max()),
        "max_cauchy_L_bound": float(slab_df["cauchy_L_bound"].max()),
        "max_remainder_over_reserve": float(slab_df["remainder_over_reserve"].max()),
        "median_remainder_over_reserve": float(slab_df["remainder_over_reserve"].median()),
        "max_beta_cauchy_over_s22_inflated": float(slab_df["beta_cauchy_over_s22_inflated"].max()),
        "cauchy_integral_ratio": cauchy_ratio,
        "s22_inflated_integral_ratio": inflated_ratio,
        "min_dF_dE_abs": np.nan,
        "max_dF_dE_CR_rel_mismatch": np.nan,
        "n_cauchy_slab_pass": float(slab_df["cauchy_slab_pass"].sum()),
        "cauchy_slab_status": "cauchy_slab_bound_closed"
        if float(slab_df["cauchy_slab_pass"].sum()) == len(slab_df) and cauchy_ratio < 1.0
        else "cauchy_slab_bound_not_closed",
    }
    return check_df, slab_df, summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    min_den = float(summary["min_dF_dE_abs"].min()) if summary["min_dF_dE_abs"].notna().any() else np.nan
    max_cr = (
        float(summary["max_dF_dE_CR_rel_mismatch"].max())
        if summary["max_dF_dE_CR_rel_mismatch"].notna().any()
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "n_parity_segments": float(len(summary)),
                "n_cauchy_closed": float((summary["cauchy_slab_status"] == "cauchy_slab_bound_closed").sum()),
                "max_cauchy_integral_ratio": float(summary["cauchy_integral_ratio"].max()),
                "median_cauchy_integral_ratio": float(summary["cauchy_integral_ratio"].median()),
                "max_remainder_over_reserve": float(summary["max_remainder_over_reserve"].max()),
                "max_beta_cauchy_over_s22_inflated": float(summary["max_beta_cauchy_over_s22_inflated"].max()),
                "min_dF_dE_abs": min_den,
                "max_dF_dE_CR_rel_mismatch": max_cr,
                "cauchy_slab_status": "cauchy_slab_bound_closed_on_grid"
                if float((summary["cauchy_slab_status"] == "cauchy_slab_bound_closed").sum()) == len(summary)
                else "cauchy_slab_bound_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cauchy/slab derivative bound for the final Whittaker-DtN flowbox.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--s22-slabs-csv", default=str(DEFAULT_S22_SLABS))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="6")
    ap.add_argument("--R-tail-values", default="400")
    ap.add_argument("--parities", default="even,odd")
    ap.add_argument("--segments", default="60:80")
    ap.add_argument("--center-mode", default="cubic_path")
    ap.add_argument("--tube-factor", type=float, default=640.0)
    ap.add_argument("--n-per-side", type=int, default=2)
    ap.add_argument("--dR-step", type=float, default=0.5)
    ap.add_argument("--dE-step-fraction", type=float, default=0.001)
    ap.add_argument("--cauchy-R-step-fraction", type=float, default=0.25)
    ap.add_argument("--cauchy-guard-factor", type=float, default=5.0)
    ap.add_argument("--derivative-mode", choices=("node_slope", "direct"), default="node_slope")
    ap.add_argument("--whittaker-dps", type=int, default=50)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    argument_csv = Path(args.argument_csv)
    s22_slabs_csv = Path(args.s22_slabs_csv)
    if not argument_csv.exists():
        raise FileNotFoundError(f"Missing argument-principle CSV: {argument_csv}")
    if not s22_slabs_csv.exists():
        raise FileNotFoundError(f"Missing S22 slabs CSV: {s22_slabs_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(argument_csv)
    s22_slabs = pd.read_csv(s22_slabs_csv)
    Ds = parse_floats(args.Ds)
    Rtails = parse_floats(args.R_tail_values)
    parities = parse_strings(args.parities)
    segments = parse_segments(args.segments)
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
        s22_slabs = s22_slabs[s22_slabs["D"].isin(Ds)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
        s22_slabs = s22_slabs[s22_slabs["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
        s22_slabs = s22_slabs[s22_slabs["parity"].isin(parities)]
    if not segments:
        raise ValueError("Need nonempty --segments.")

    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    check_frames: list[pd.DataFrame] = []
    slab_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    for (D, R_tail, parity), path in detail.groupby(["D", "R_tail", "parity"], sort=True):
        for R0, R1 in segments:
            slabs = s22_slabs[
                (s22_slabs["D"] == D)
                & (s22_slabs["R_tail"] == R_tail)
                & (s22_slabs["parity"] == parity)
                & (s22_slabs["R0"] == R0)
                & (s22_slabs["R1"] == R1)
            ].copy()
            if len(slabs) == 0:
                continue
            print(f"[cauchy-slab] D={D:g} Rc={R_tail:g} parity={parity} R={R0:g}->{R1:g}", flush=True)
            if args.derivative_mode == "node_slope":
                checks, slab_detail, summary = audit_node_slope_parity(
                    slab_rows=slabs,
                    center_mode=str(args.center_mode),
                    tube_factor=float(args.tube_factor),
                    cauchy_guard_factor=float(args.cauchy_guard_factor),
                )
            else:
                checks, slab_detail, summary = audit_parity(
                    path=path,
                    slab_rows=slabs,
                    center_mode=str(args.center_mode),
                    tube_factor=float(args.tube_factor),
                    n_per_side=int(args.n_per_side),
                    dR_step=float(args.dR_step),
                    dE_step_fraction=float(args.dE_step_fraction),
                    cauchy_R_step_fraction=float(args.cauchy_R_step_fraction),
                    cauchy_guard_factor=float(args.cauchy_guard_factor),
                    cfg=cfg,
                    p=p,
                )
            check_frames.append(checks)
            slab_frames.append(slab_detail)
            summaries.append(summary)

    if not summaries:
        raise ValueError("No Cauchy/slab rows were audited.")
    check_detail = pd.concat(check_frames, ignore_index=True)
    slab_detail = pd.concat(slab_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    aggregate = aggregate_summary(summary)

    checks_out = OUTDIR / f"{args.out_tag}_checks.csv"
    slabs_out = OUTDIR / f"{args.out_tag}_slabs.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    aggregate_out = OUTDIR / f"{args.out_tag}_aggregate_summary.csv"
    check_detail.to_csv(checks_out, index=False)
    slab_detail.to_csv(slabs_out, index=False)
    summary.to_csv(summary_out, index=False)
    aggregate.to_csv(aggregate_out, index=False)

    print(f"Wrote {checks_out}")
    print(f"Wrote {slabs_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {aggregate_out}")
    print("[Cauchy/slab summary]")
    print(summary.to_string(index=False))
    print("[Cauchy/slab aggregate]")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
