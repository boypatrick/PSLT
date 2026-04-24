#!/usr/bin/env python3
"""
Physical static-width extractor for the CAP and local Robin/Siegert routes.

This script deliberately separates "a complex width can be computed" from
"a physical static width has been extracted."  It runs, or reuses, the two
available finite-box diagnostics for the same action-derived one-dimensional
operator

    H_{D,l} = -d^2/dz^2 + U(z;D) + l(l+1)/(z^2+eps^2),

and then applies promotion gates:

  CAP route
    H_cap(eta,z_cap) = H_{D,l} - i eta W_{z_cap}.
    A physical resonance width must be stationary in eta and in z_cap, while
    Gamma_cap/eta must not stay constant across the eta window.  The latter
    rejects absorber-linear leakage.

  Local Robin/Siegert route
    The half-line parity roots satisfy psi'(R)=i q_R(E) psi(R), with
    q_R(E)=sqrt(E-U_l(R;D)).  A physical local-pole candidate must have tiny
    residuals, negligible parity splitting, and an R-stable width.

Only if both independent routes pass their internal gates and agree with each
other is `physical_width` populated.  Otherwise the script reports the best
candidate metrics and an explicit non-promotion reason.

Outputs:
  - output/cap_resonance_1d/static_width_physical_extraction_detail.csv
  - output/cap_resonance_1d/static_width_physical_extraction_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import (
    AuditConfig,
    relative_span,
    solve_siegert_case,
)
from extract_cap_resonance_1d import (
    CapParams,
    DEFAULT_OUTDIR,
    solve_cap_case,
    tag_from_floats,
)
from extract_superrad_prefactor_1d import PhysicalParams


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = DEFAULT_OUTDIR


@dataclass(frozen=True)
class PhysicalExtractionConfig:
    ell: int = 1
    n_target: int = 3
    d_values: tuple[float, ...] = (6.0, 12.0, 18.0)

    cap_zmax: float = 80.0
    cap_Nz: int = 1201
    cap_power: float = 2.0
    cap_k_eigs: int = 6
    cap_etas: tuple[float, ...] = (5.0e-5, 5.53e-5, 6.12e-5, 6.78e-5, 7.5e-5)
    cap_z_cap_fracs: tuple[float, ...] = (0.8, 0.825, 0.85, 0.875, 0.9)

    siegert_n_half: int = 1200
    siegert_r_values: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0)
    siegert_parity_values: tuple[str, ...] = ("even", "odd")

    # Promotion gates.  These are intentionally stricter than diagnostic gates:
    # a promoted static width should be insensitive to the absorber and to the
    # artificial Robin radius.
    cap_eta_gamma_rel_span_max: float = 0.15
    cap_gamma_over_eta_rel_span_min: float = 0.25
    cap_zcap_gamma_rel_span_max: float = 0.25
    cap_candidate_overlap_min: float = 0.99
    cap_eta_zcap_candidate_rel_diff_max: float = 0.25

    siegert_R_gamma_rel_span_max: float = 0.15
    siegert_R120_over_R80_min: float = 0.90
    siegert_R120_over_R80_max: float = 1.10
    siegert_parity_gamma_rel_split_max: float = 1.0e-5
    siegert_root_residual_abs_max: float = 1.0e-8

    cross_method_width_rel_diff_max: float = 0.25


def finite_median(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def rel_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(abs(a - b) / max(0.5 * (abs(a) + abs(b)), 1.0e-300))


def bool01(flag: bool) -> float:
    return 1.0 if flag else 0.0


def fail_join(reasons: list[str]) -> str:
    return "pass" if len(reasons) == 0 else ";".join(reasons)


def cap_scan_base_name(cfg: PhysicalExtractionConfig) -> str:
    d_values = list(cfg.d_values)
    eta_values = list(cfg.cap_etas)
    zcap_values = list(cfg.cap_z_cap_fracs)
    tag_D = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_values)
    tag_eta = tag_from_floats(eta_values, prefix="eta")
    tag_zcap = tag_from_floats(zcap_values, prefix="zcap")
    return f"cap_resonance_scan_D{tag_D}_l{cfg.ell}_N{cfg.n_target}_{tag_eta}_{tag_zcap}"


def expected_cap_scan_path(cfg: PhysicalExtractionConfig) -> Path:
    return OUTDIR / f"{cap_scan_base_name(cfg)}.csv"


def cap_scan_matches(df: pd.DataFrame, cfg: PhysicalExtractionConfig) -> bool:
    required = {
        "D",
        "ell",
        "N_target",
        "eta_cap",
        "z_cap_frac",
        "Gamma_cap",
        "Gamma_cap_over_eta",
        "candidate_overlap",
    }
    if not required.issubset(df.columns):
        return False

    def rounded(vals: Iterable[float]) -> set[float]:
        return {round(float(v), 12) for v in vals}

    return (
        rounded(df["D"].unique()) == rounded(cfg.d_values)
        and rounded(df["eta_cap"].unique()) == rounded(cfg.cap_etas)
        and rounded(df["z_cap_frac"].unique()) == rounded(cfg.cap_z_cap_fracs)
        and rounded(df["ell"].unique()) == {round(float(cfg.ell), 12)}
        and rounded(df["N_target"].unique()) == {round(float(cfg.n_target), 12)}
    )


def load_or_run_cap_scan(
    cfg: PhysicalExtractionConfig,
    p: PhysicalParams,
    force: bool,
) -> tuple[pd.DataFrame, str]:
    scan_path = expected_cap_scan_path(cfg)
    if not force and scan_path.exists():
        df = pd.read_csv(scan_path)
        if cap_scan_matches(df, cfg):
            return df.sort_values(["D", "z_cap_frac", "eta_cap"]).reset_index(drop=True), str(scan_path)

    base = CapParams(
        zmax=cfg.cap_zmax,
        Nz=cfg.cap_Nz,
        z_cap_frac=cfg.cap_z_cap_fracs[0],
        cap_power=cfg.cap_power,
        n_target=cfg.n_target,
        k_eigs=cfg.cap_k_eigs,
    )
    rows: list[dict[str, float]] = []
    for D in cfg.d_values:
        for z_cap_frac in cfg.cap_z_cap_fracs:
            case_cfg = replace(base, z_cap_frac=z_cap_frac)
            for eta in cfg.cap_etas:
                rows.append(solve_cap_case(D=D, ell=cfg.ell, eta=eta, cfg=case_cfg, p=p))

    df = pd.DataFrame(rows).sort_values(["D", "z_cap_frac", "eta_cap"]).reset_index(drop=True)
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(scan_path, index=False)
    return df, str(scan_path)


def siegert_detail_matches(df: pd.DataFrame, cfg: PhysicalExtractionConfig) -> bool:
    required = {
        "D",
        "ell",
        "N_target",
        "R",
        "parity",
        "Gamma_siegert",
        "E_siegert_real",
        "E_siegert_imag",
        "residual_abs",
    }
    if not required.issubset(df.columns):
        return False

    def rounded(vals: Iterable[float]) -> set[float]:
        return {round(float(v), 12) for v in vals}

    return (
        rounded(df["D"].unique()) == rounded(cfg.d_values)
        and rounded(df["R"].unique()) == rounded(cfg.siegert_r_values)
        and set(str(v) for v in df["parity"].unique()) == set(cfg.siegert_parity_values)
        and rounded(df["ell"].unique()) == {round(float(cfg.ell), 12)}
        and rounded(df["N_target"].unique()) == {round(float(cfg.n_target), 12)}
    )


def load_or_run_siegert_detail(
    cfg: PhysicalExtractionConfig,
    p: PhysicalParams,
    force: bool,
) -> tuple[pd.DataFrame, str]:
    detail_path = OUTDIR / "static_width_siegert_audit_detail.csv"
    if not force and detail_path.exists():
        df = pd.read_csv(detail_path)
        if siegert_detail_matches(df, cfg):
            return df.sort_values(["D", "R", "parity"]).reset_index(drop=True), str(detail_path)

    audit_cfg = AuditConfig(
        ell=cfg.ell,
        n_target=cfg.n_target,
        n_half=cfg.siegert_n_half,
        d_values=cfg.d_values,
        r_values=cfg.siegert_r_values,
        parity_values=cfg.siegert_parity_values,
    )
    rows = [
        solve_siegert_case(D=D, R=R, parity=parity, cfg=audit_cfg, p=p)
        for D in cfg.d_values
        for R in cfg.siegert_r_values
        for parity in cfg.siegert_parity_values
    ]
    df = pd.DataFrame(rows).sort_values(["D", "R", "parity"]).reset_index(drop=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_path, index=False)
    return df, str(detail_path)


def cap_gate_defect_eta(row: pd.Series, cfg: PhysicalExtractionConfig) -> float:
    span_defect = max(0.0, float(row["Gamma_cap_rel_span"]) - cfg.cap_eta_gamma_rel_span_max)
    linear_defect = max(
        0.0,
        cfg.cap_gamma_over_eta_rel_span_min - float(row["Gamma_cap_over_eta_rel_span"]),
    )
    overlap_defect = max(0.0, cfg.cap_candidate_overlap_min - float(row["candidate_overlap_min"]))
    return float(
        span_defect / cfg.cap_eta_gamma_rel_span_max
        + linear_defect / cfg.cap_gamma_over_eta_rel_span_min
        + overlap_defect / cfg.cap_candidate_overlap_min
    )


def cap_gate_defect_zcap(row: pd.Series, cfg: PhysicalExtractionConfig) -> float:
    span_defect = max(0.0, float(row["Gamma_cap_rel_span"]) - cfg.cap_zcap_gamma_rel_span_max)
    overlap_defect = max(0.0, cfg.cap_candidate_overlap_min - float(row["candidate_overlap_min"]))
    return float(
        span_defect / cfg.cap_zcap_gamma_rel_span_max
        + overlap_defect / cfg.cap_candidate_overlap_min
    )


def summarize_cap_D(df: pd.DataFrame, cfg: PhysicalExtractionConfig) -> dict[str, float | str]:
    eta_rows: list[dict[str, float]] = []
    for z_cap_frac, sub in df.groupby("z_cap_frac", sort=True):
        sub = sub.sort_values("eta_cap")
        eta_rows.append(
            {
                "z_cap_frac": float(z_cap_frac),
                "n_eta": float(len(sub)),
                "Gamma_cap_eta_median": finite_median(sub["Gamma_cap"]),
                "Gamma_cap_rel_span": relative_span(sub["Gamma_cap"]),
                "Gamma_cap_over_eta_rel_span": relative_span(sub["Gamma_cap_over_eta"]),
                "candidate_overlap_min": float(sub["candidate_overlap"].min()),
            }
        )
    eta_summary = pd.DataFrame(eta_rows)
    eta_summary["gate_defect"] = eta_summary.apply(
        lambda row: cap_gate_defect_eta(row, cfg=cfg),
        axis=1,
    )
    eta_best = eta_summary.sort_values(
        ["gate_defect", "Gamma_cap_rel_span", "Gamma_cap_over_eta_rel_span"],
        ascending=[True, True, False],
    ).iloc[0]

    zcap_rows: list[dict[str, float]] = []
    for eta, sub in df.groupby("eta_cap", sort=True):
        sub = sub.sort_values("z_cap_frac")
        zcap_rows.append(
            {
                "eta_cap": float(eta),
                "n_z_cap_frac": float(len(sub)),
                "Gamma_cap_zcap_median": finite_median(sub["Gamma_cap"]),
                "Gamma_cap_rel_span": relative_span(sub["Gamma_cap"]),
                "Gamma_cap_over_eta_rel_span": relative_span(sub["Gamma_cap_over_eta"]),
                "candidate_overlap_min": float(sub["candidate_overlap"].min()),
            }
        )
    zcap_summary = pd.DataFrame(zcap_rows)
    zcap_summary["gate_defect"] = zcap_summary.apply(
        lambda row: cap_gate_defect_zcap(row, cfg=cfg),
        axis=1,
    )
    zcap_best = zcap_summary.sort_values(
        ["gate_defect", "Gamma_cap_rel_span"],
        ascending=[True, True],
    ).iloc[0]

    cap_width_eta = float(eta_best["Gamma_cap_eta_median"])
    cap_width_zcap = float(zcap_best["Gamma_cap_zcap_median"])
    cap_internal_diff = rel_diff(cap_width_eta, cap_width_zcap)

    reasons: list[str] = []
    if float(eta_best["Gamma_cap_rel_span"]) > cfg.cap_eta_gamma_rel_span_max:
        reasons.append("eta_width_not_stationary")
    if float(eta_best["Gamma_cap_over_eta_rel_span"]) < cfg.cap_gamma_over_eta_rel_span_min:
        reasons.append("eta_absorber_linear")
    if float(zcap_best["Gamma_cap_rel_span"]) > cfg.cap_zcap_gamma_rel_span_max:
        reasons.append("zcap_boundary_not_stationary")
    if min(float(eta_best["candidate_overlap_min"]), float(zcap_best["candidate_overlap_min"])) < cfg.cap_candidate_overlap_min:
        reasons.append("candidate_overlap_low")
    if (
        np.isfinite(cap_internal_diff)
        and cap_internal_diff > cfg.cap_eta_zcap_candidate_rel_diff_max
    ):
        reasons.append("eta_zcap_candidates_disagree")

    cap_pass = len(reasons) == 0
    cap_candidate = finite_median([cap_width_eta, cap_width_zcap])
    return {
        "cap_pass": bool01(cap_pass),
        "cap_status": "physical_cap_plateau" if cap_pass else "not_promoted",
        "cap_fail_reasons": fail_join(reasons),
        "Gamma_cap_physical": cap_candidate if cap_pass else np.nan,
        "Gamma_cap_candidate": cap_candidate,
        "Gamma_cap_eta_candidate": cap_width_eta,
        "Gamma_cap_zcap_candidate": cap_width_zcap,
        "Gamma_cap_eta_zcap_candidate_rel_diff": cap_internal_diff,
        "cap_best_z_cap_frac_for_eta": float(eta_best["z_cap_frac"]),
        "cap_best_eta_for_zcap": float(zcap_best["eta_cap"]),
        "cap_eta_Gamma_rel_span_best": float(eta_best["Gamma_cap_rel_span"]),
        "cap_eta_Gamma_over_eta_rel_span_best": float(eta_best["Gamma_cap_over_eta_rel_span"]),
        "cap_zcap_Gamma_rel_span_best": float(zcap_best["Gamma_cap_rel_span"]),
        "cap_overlap_min_best": float(
            min(float(eta_best["candidate_overlap_min"]), float(zcap_best["candidate_overlap_min"]))
        ),
    }


def summarize_siegert_D(df: pd.DataFrame, cfg: PhysicalExtractionConfig) -> dict[str, float | str]:
    by_R = df.groupby("R", sort=True)["Gamma_siegert"].mean()
    gamma_R80 = float(by_R.loc[80.0]) if 80.0 in by_R.index else np.nan
    gamma_R120 = float(by_R.loc[120.0]) if 120.0 in by_R.index else np.nan
    ratio_120_80 = (
        float(gamma_R120 / gamma_R80)
        if np.isfinite(gamma_R80) and abs(gamma_R80) > 0.0
        else np.nan
    )

    even = df[df["parity"] == "even"].set_index("R")
    odd = df[df["parity"] == "odd"].set_index("R")
    common_R = sorted(set(even.index).intersection(set(odd.index)))
    parity_gamma_rel = []
    parity_E_abs = []
    for R in common_R:
        ge = float(even.loc[R, "Gamma_siegert"])
        go = float(odd.loc[R, "Gamma_siegert"])
        ee = complex(float(even.loc[R, "E_siegert_real"]), float(even.loc[R, "E_siegert_imag"]))
        eo = complex(float(odd.loc[R, "E_siegert_real"]), float(odd.loc[R, "E_siegert_imag"]))
        parity_gamma_rel.append(abs(ge - go) / max(0.5 * (abs(ge) + abs(go)), 1.0e-300))
        parity_E_abs.append(abs(ee - eo))

    gamma_R_rel_span = relative_span(by_R.to_numpy(dtype=float))
    parity_gamma_rel_max = float(max(parity_gamma_rel)) if parity_gamma_rel else np.nan
    parity_E_abs_max = float(max(parity_E_abs)) if parity_E_abs else np.nan
    residual_abs_max = float(df["residual_abs"].max())
    siegert_candidate = finite_median(by_R.to_numpy(dtype=float))

    reasons: list[str] = []
    if residual_abs_max > cfg.siegert_root_residual_abs_max:
        reasons.append("root_residual_large")
    if parity_gamma_rel_max > cfg.siegert_parity_gamma_rel_split_max:
        reasons.append("parity_split_large")
    if gamma_R_rel_span > cfg.siegert_R_gamma_rel_span_max:
        reasons.append("R_width_not_stationary")
    if not (
        np.isfinite(ratio_120_80)
        and cfg.siegert_R120_over_R80_min <= ratio_120_80 <= cfg.siegert_R120_over_R80_max
    ):
        reasons.append("R120_R80_ratio_outside_gate")

    siegert_pass = len(reasons) == 0
    return {
        "siegert_pass": bool01(siegert_pass),
        "siegert_status": "physical_local_robin_pole" if siegert_pass else "not_promoted",
        "siegert_fail_reasons": fail_join(reasons),
        "Gamma_siegert_physical": siegert_candidate if siegert_pass else np.nan,
        "Gamma_siegert_candidate": siegert_candidate,
        "Gamma_siegert_R_rel_span": gamma_R_rel_span,
        "Gamma_siegert_R120_over_R80": ratio_120_80,
        "parity_Gamma_rel_split_max": parity_gamma_rel_max,
        "parity_E_abs_split_max": parity_E_abs_max,
        "root_residual_abs_max": residual_abs_max,
    }


def summarize_physical(
    cap_detail: pd.DataFrame,
    siegert_detail: pd.DataFrame,
    cfg: PhysicalExtractionConfig,
    cap_source: str,
    siegert_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D in cfg.d_values:
        cap_D = cap_detail[np.isclose(cap_detail["D"].to_numpy(dtype=float), D)].copy()
        siegert_D = siegert_detail[np.isclose(siegert_detail["D"].to_numpy(dtype=float), D)].copy()

        cap_row = summarize_cap_D(cap_D, cfg=cfg)
        siegert_row = summarize_siegert_D(siegert_D, cfg=cfg)
        method_rel_diff = rel_diff(
            float(cap_row["Gamma_cap_candidate"]),
            float(siegert_row["Gamma_siegert_candidate"]),
        )

        cap_pass = bool(cap_row["cap_pass"])
        siegert_pass = bool(siegert_row["siegert_pass"])
        if cap_pass and siegert_pass and method_rel_diff <= cfg.cross_method_width_rel_diff_max:
            status = "physical_width_extracted"
            physical_width = finite_median(
                [
                    float(cap_row["Gamma_cap_physical"]),
                    float(siegert_row["Gamma_siegert_physical"]),
                ]
            )
            recommendation = "promote_static_width_with_CAP_and_local_Robin_cross_check"
        elif cap_pass and siegert_pass:
            status = "method_disagreement_not_promoted"
            physical_width = np.nan
            recommendation = "tighten_exterior_DtN_or_ECS_before_promotion"
        elif cap_pass or siegert_pass:
            status = "single_method_candidate_not_promoted"
            physical_width = np.nan
            recommendation = "obtain_an_independent_matching_static_width_before_promotion"
        else:
            status = "not_promoted"
            physical_width = np.nan
            recommendation = "replace_boundary_surrogates_by_exact_DtN_or_ECS_then_retest"

        row: dict[str, float | str] = {
            "D": float(D),
            "ell": float(cfg.ell),
            "N_target": float(cfg.n_target),
            "physical_width_status": status,
            "Gamma_static_physical": physical_width,
            "Gamma_CAP_vs_Siegert_candidate_rel_diff": method_rel_diff,
            "recommended_next_step": recommendation,
            "cap_source": cap_source,
            "siegert_source": siegert_source,
        }
        row.update(cap_row)
        row.update(siegert_row)
        rows.append(row)

    global_status = (
        "physical_width_extracted_all_D"
        if all(r["physical_width_status"] == "physical_width_extracted" for r in rows)
        else "not_promoted_global"
    )
    global_row: dict[str, float | str] = {
        "D": "global",
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "physical_width_status": global_status,
        "Gamma_static_physical": np.nan,
        "Gamma_CAP_vs_Siegert_candidate_rel_diff": float(
            max(r["Gamma_CAP_vs_Siegert_candidate_rel_diff"] for r in rows)
        ),
        "recommended_next_step": (
            "publish_no_static_width_promotion_until_parent_side_outgoing_boundary_is_upgraded"
        ),
        "cap_source": cap_source,
        "siegert_source": siegert_source,
        "cap_pass": bool01(all(r["cap_pass"] == 1.0 for r in rows)),
        "cap_status": "mixed",
        "cap_fail_reasons": ";".join(sorted(set(str(r["cap_fail_reasons"]) for r in rows))),
        "Gamma_cap_physical": np.nan,
        "Gamma_cap_candidate": float(max(r["Gamma_cap_candidate"] for r in rows)),
        "Gamma_cap_eta_candidate": float(max(r["Gamma_cap_eta_candidate"] for r in rows)),
        "Gamma_cap_zcap_candidate": float(max(r["Gamma_cap_zcap_candidate"] for r in rows)),
        "Gamma_cap_eta_zcap_candidate_rel_diff": float(
            max(r["Gamma_cap_eta_zcap_candidate_rel_diff"] for r in rows)
        ),
        "cap_best_z_cap_frac_for_eta": np.nan,
        "cap_best_eta_for_zcap": np.nan,
        "cap_eta_Gamma_rel_span_best": float(max(r["cap_eta_Gamma_rel_span_best"] for r in rows)),
        "cap_eta_Gamma_over_eta_rel_span_best": float(
            min(r["cap_eta_Gamma_over_eta_rel_span_best"] for r in rows)
        ),
        "cap_zcap_Gamma_rel_span_best": float(max(r["cap_zcap_Gamma_rel_span_best"] for r in rows)),
        "cap_overlap_min_best": float(min(r["cap_overlap_min_best"] for r in rows)),
        "siegert_pass": bool01(all(r["siegert_pass"] == 1.0 for r in rows)),
        "siegert_status": "mixed",
        "siegert_fail_reasons": ";".join(sorted(set(str(r["siegert_fail_reasons"]) for r in rows))),
        "Gamma_siegert_physical": np.nan,
        "Gamma_siegert_candidate": float(max(r["Gamma_siegert_candidate"] for r in rows)),
        "Gamma_siegert_R_rel_span": float(max(r["Gamma_siegert_R_rel_span"] for r in rows)),
        "Gamma_siegert_R120_over_R80": float(min(r["Gamma_siegert_R120_over_R80"] for r in rows)),
        "parity_Gamma_rel_split_max": float(max(r["parity_Gamma_rel_split_max"] for r in rows)),
        "parity_E_abs_split_max": float(max(r["parity_E_abs_split_max"] for r in rows)),
        "root_residual_abs_max": float(max(r["root_residual_abs_max"] for r in rows)),
    }
    rows.append(global_row)
    return pd.DataFrame(rows)


def combined_detail(cap_detail: pd.DataFrame, siegert_detail: pd.DataFrame) -> pd.DataFrame:
    cap = cap_detail.copy()
    cap.insert(0, "method", "CAP")
    siegert = siegert_detail.copy()
    siegert.insert(0, "method", "local_robin_siegert")
    return pd.concat([cap, siegert], ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Gate CAP and local Robin/Siegert candidates into a physical static-width extraction."
    )
    ap.add_argument("--force-cap-scan", action="store_true", help="Recompute the CAP grid instead of reusing the matching CSV.")
    ap.add_argument("--force-siegert-scan", action="store_true", help="Recompute local Robin/Siegert roots instead of reusing the matching CSV.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = PhysicalExtractionConfig()
    p = PhysicalParams()

    cap_detail, cap_source = load_or_run_cap_scan(cfg=cfg, p=p, force=args.force_cap_scan)
    siegert_detail, siegert_source = load_or_run_siegert_detail(
        cfg=cfg,
        p=p,
        force=args.force_siegert_scan,
    )
    summary = summarize_physical(
        cap_detail=cap_detail,
        siegert_detail=siegert_detail,
        cfg=cfg,
        cap_source=cap_source,
        siegert_source=siegert_source,
    )
    detail = combined_detail(cap_detail=cap_detail, siegert_detail=siegert_detail)

    detail_out = OUTDIR / "static_width_physical_extraction_detail.csv"
    summary_out = OUTDIR / "static_width_physical_extraction_summary.csv"
    detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print("[static-width physical extraction summary]")
    display_cols = [
        "D",
        "physical_width_status",
        "Gamma_static_physical",
        "Gamma_cap_candidate",
        "cap_eta_Gamma_rel_span_best",
        "cap_eta_Gamma_over_eta_rel_span_best",
        "cap_zcap_Gamma_rel_span_best",
        "Gamma_siegert_candidate",
        "Gamma_siegert_R_rel_span",
        "Gamma_siegert_R120_over_R80",
        "Gamma_CAP_vs_Siegert_candidate_rel_diff",
        "recommended_next_step",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
