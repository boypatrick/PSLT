#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _load_constants() -> tuple[float, float]:
    df = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv")
    row = df.iloc[0]
    return float(row["R_star"]), float(row["lambda_star"])


def _anchor_masks(rr: np.ndarray, zz: np.ndarray, D: float, R_star: float, rho_anchor: float, zeta_anchor: float):
    rp = np.sqrt(rr * rr + (zz - D / 2.0) ** 2)
    rm = np.sqrt(rr * rr + (zz + D / 2.0) ** 2)
    core = (rp <= R_star) | (rm <= R_star)
    plus_anchor = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor) & core
    minus_anchor = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_anchor) & core
    return plus_anchor, minus_anchor


def _evaluate_box(
    fields_by_D: dict[float, dict[str, np.ndarray]],
    Ds: list[float],
    R_star: float,
    rho_anchor: float,
    zeta_anchor: float,
    rho_box: float,
    zeta_box: float,
) -> tuple[bool, pd.DataFrame]:
    rows = []
    safe = True
    for D in Ds:
        f = fields_by_D[D]
        rr = f["rr"]
        zz = f["zz"]
        C = f["contrast_density"]
        plus_anchor, minus_anchor = _anchor_masks(rr, zz, D, R_star, rho_anchor, zeta_anchor)

        anchor_exact = float(
            np.sum(C[plus_anchor] * (f["delta_plus"][plus_anchor] + f["delta_minus"][plus_anchor]))
            + np.sum(C[minus_anchor] * (f["delta_plus"][minus_anchor] + f["delta_minus"][minus_anchor]))
        )
        sign_anchor = 1.0 if anchor_exact >= 0.0 else -1.0

        diag_anchor = float(
            np.sum(C[plus_anchor] * f["delta_plus"][plus_anchor])
            + np.sum(C[minus_anchor] * f["delta_minus"][minus_anchor])
        )
        mirror_anchor = float(
            np.sum(C[plus_anchor] * f["delta_minus"][plus_anchor])
            + np.sum(C[minus_anchor] * f["delta_plus"][minus_anchor])
        )

        plus_box = (rr <= rho_box) & (np.abs(zz - D / 2.0) <= zeta_box) & plus_anchor
        minus_box = (rr <= rho_box) & (np.abs(zz + D / 2.0) <= zeta_box) & minus_anchor

        diag_box = float(
            np.sum(C[plus_box] * f["delta_plus"][plus_box])
            + np.sum(C[minus_box] * f["delta_minus"][minus_box])
        )
        diag_shell = diag_anchor - diag_box

        box_same = diag_box == 0.0 or np.sign(diag_box) == sign_anchor
        shell_same = diag_shell == 0.0 or np.sign(diag_shell) == sign_anchor
        safe = safe and box_same and shell_same
        rows.append(
            {
                "D": D,
                "R_star": R_star,
                "rho_anchor": rho_anchor,
                "zeta_anchor": zeta_anchor,
                "rho_box": rho_box,
                "zeta_box": zeta_box,
                "anchor_exact": anchor_exact,
                "diag_anchor": diag_anchor,
                "mirror_anchor": mirror_anchor,
                "diag_box": diag_box,
                "diag_shell": diag_shell,
                "diag_same_sign_as_anchor": int(diag_anchor == 0.0 or np.sign(diag_anchor) == sign_anchor),
                "mirror_same_sign_as_anchor": int(mirror_anchor == 0.0 or np.sign(mirror_anchor) == sign_anchor),
                "box_same_sign_as_anchor": int(box_same),
                "shell_same_sign_as_anchor": int(shell_same),
                "diag_abs_share_of_anchor": abs(diag_anchor) / max(abs(anchor_exact), 1e-300),
                "mirror_abs_share_of_anchor": abs(mirror_anchor) / max(abs(anchor_exact), 1e-300),
                "box_abs_share_of_diag": abs(diag_box) / max(abs(diag_anchor), 1e-300),
                "shell_abs_share_of_diag": abs(diag_shell) / max(abs(diag_anchor), 1e-300),
                "mirror_over_diag_abs": abs(mirror_anchor) / max(abs(diag_anchor), 1e-300),
            }
        )
    return safe, pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-anchor", type=float, default=1.21)
    ap.add_argument("--rho-min", type=float, default=0.40)
    ap.add_argument("--rho-max", type=float, default=0.70)
    ap.add_argument("--rho-step", type=float, default=0.01)
    ap.add_argument("--zeta-min", type=float, default=0.80)
    ap.add_argument("--zeta-max", type=float, default=1.21)
    ap.add_argument("--zeta-step", type=float, default=0.01)
    ap.add_argument("--rho-max-grid", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    R_star, lambda_star = _load_constants()
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))

    fields_by_D = {}
    for D in Ds:
        fields_by_D[D] = solve_fields(
            D=D,
            p=p,
            level=level,
            rho_max=float(args.rho_max_grid),
            z_margin=float(args.z_margin),
            sigma=float(args.sigma),
            tol=float(args.tol),
            maxiter=int(args.maxiter),
        )

    scan_rows = []
    best = None
    best_df = None
    rho_values = np.arange(float(args.rho_min), float(args.rho_max) + 0.5 * float(args.rho_step), float(args.rho_step))
    zeta_values = np.arange(float(args.zeta_min), float(args.zeta_max) + 0.5 * float(args.zeta_step), float(args.zeta_step))
    for rho_box in rho_values:
        for zeta_box in zeta_values:
            safe, df = _evaluate_box(
                fields_by_D,
                Ds,
                R_star,
                float(args.rho_anchor),
                float(args.zeta_anchor),
                float(rho_box),
                float(zeta_box),
            )
            if not safe:
                scan_rows.append(
                    {
                        "rho_box": float(rho_box),
                        "zeta_box": float(zeta_box),
                        "safe_box": 0,
                        "diag_box_floor_abs": np.nan,
                        "mu_mirror_over_box": np.nan,
                        "induced_anchor_lower_bound": np.nan,
                    }
                )
                continue
            diag_box_floor = float(df["diag_box"].abs().min())
            mu = float(df["mirror_anchor"].abs().div(df["diag_box"].abs().clip(lower=1e-300)).max())
            induced_anchor_lb = float((1.0 - mu) * diag_box_floor) if mu < 1.0 else np.nan
            row = {
                "rho_box": float(rho_box),
                "zeta_box": float(zeta_box),
                "safe_box": 1,
                "diag_box_floor_abs": diag_box_floor,
                "mu_mirror_over_box": mu,
                "induced_anchor_lower_bound": induced_anchor_lb,
            }
            scan_rows.append(row)
            if np.isfinite(induced_anchor_lb):
                cand = (induced_anchor_lb, diag_box_floor, -mu, float(rho_box), float(zeta_box))
                if best is None or cand > best:
                    best = cand
                    best_df = df.copy()

    if best is None or best_df is None:
        raise RuntimeError("No safe same-center box found inside the canonical anchor.")

    safe_canonical, canonical_df = _evaluate_box(
        fields_by_D,
        Ds,
        R_star,
        float(args.rho_anchor),
        float(args.zeta_anchor),
        float(args.rho_anchor),
        float(args.zeta_anchor),
    )
    if not safe_canonical:
        raise RuntimeError("Canonical same-center anchor box is not safe on the audited knot set.")

    scan = pd.DataFrame(scan_rows).sort_values(["rho_box", "zeta_box"]).reset_index(drop=True)
    detail = canonical_df.copy()
    diag_floor_idx = int(detail["diag_anchor"].abs().idxmin())
    anchor_floor_idx = int(detail["anchor_exact"].abs().idxmin())
    diag_floor = float(detail["diag_anchor"].abs().min())
    mu_anchor = float(detail["mirror_anchor"].abs().div(detail["diag_anchor"].abs().clip(lower=1e-300)).max())
    induced_anchor_floor = float((1.0 - mu_anchor) * diag_floor)

    prev = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_self_core_anchor_source_summary.csv").iloc[0]
    self_pair_exact_floor = float(prev["self_pair_exact_floor_abs"])
    exact_total_floor = float(prev["exact_total_floor_abs"])

    summary = pd.DataFrame(
        [
            {
                "R_star": R_star,
                "lambda_star": lambda_star,
                "rho_anchor_canonical": float(args.rho_anchor),
                "zeta_anchor_canonical": float(args.zeta_anchor),
                "rho_box_opt": float(best[3]),
                "zeta_box_opt": float(best[4]),
                "diag_box_floor_abs_opt": float(best[1]),
                "mu_mirror_over_box_opt": float(-best[2]),
                "induced_anchor_lower_bound_opt": float(best[0]),
                "diag_anchor_floor_abs": diag_floor,
                "diag_anchor_floor_D": float(detail.loc[diag_floor_idx, "D"]),
                "anchor_exact_floor_abs": float(detail["anchor_exact"].abs().min()),
                "anchor_exact_floor_D": float(detail.loc[anchor_floor_idx, "D"]),
                "mu_anchor": mu_anchor,
                "induced_anchor_lower_bound_canonical": induced_anchor_floor,
                "all_diag_same_sign_as_anchor": int((detail["diag_same_sign_as_anchor"] == 1).all()),
                "max_mirror_over_diag_abs": mu_anchor,
                "self_pair_exact_floor_abs": self_pair_exact_floor,
                "exact_total_floor_abs": exact_total_floor,
                "induced_self_pair_floor_from_canonical": float(lambda_star * induced_anchor_floor),
            }
        ]
    )
    summary["anchor_exact_floor_over_induced"] = summary["anchor_exact_floor_abs"] / summary["induced_anchor_lower_bound_canonical"]
    summary["self_pair_exact_floor_over_induced"] = summary["self_pair_exact_floor_abs"] / summary["induced_self_pair_floor_from_canonical"]
    summary["exact_total_floor_over_induced"] = summary["exact_total_floor_abs"] / summary["induced_self_pair_floor_from_canonical"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_anchor_static_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_anchor_static_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_anchor_static_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    scan.to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
