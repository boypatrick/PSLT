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


def _load_lambda_star() -> tuple[float, float]:
    summary_path = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv"
    df = pd.read_csv(summary_path)
    row = df.iloc[0]
    return float(row["R_star"]), float(row["lambda_star"])


def _evaluate_box(
    fields_by_D: dict[float, dict[str, np.ndarray]],
    Ds: list[float],
    R_star: float,
    rho_box: float,
    zeta_box: float,
) -> tuple[bool, pd.DataFrame]:
    rows = []
    safe = True
    for D in Ds:
        f = fields_by_D[D]
        rr = f["rr"]
        zz = f["zz"]
        self_density = f["contrast_density"] * (f["delta_plus"] + f["delta_minus"])
        rp = np.sqrt(rr * rr + (zz - D / 2.0) ** 2)
        rm = np.sqrt(rr * rr + (zz + D / 2.0) ** 2)
        core = (rp <= R_star) | (rm <= R_star)
        exact_core = float(np.sum(self_density[core]))
        sign_core = 1.0 if exact_core >= 0.0 else -1.0

        anchor = (
            ((rr <= rho_box) & (np.abs(zz - D / 2.0) <= zeta_box))
            | ((rr <= rho_box) & (np.abs(zz + D / 2.0) <= zeta_box))
        ) & core
        shell = core & (~anchor)

        anchor_val = float(np.sum(self_density[anchor]))
        shell_val = float(np.sum(self_density[shell]))
        anchor_same = anchor_val == 0.0 or np.sign(anchor_val) == sign_core
        shell_same = shell_val == 0.0 or np.sign(shell_val) == sign_core
        safe = safe and anchor_same and shell_same
        rows.append(
            {
                "D": D,
                "R_star": R_star,
                "rho_box": rho_box,
                "zeta_box": zeta_box,
                "self_core_exact": exact_core,
                "self_anchor": anchor_val,
                "self_shell": shell_val,
                "anchor_same_sign_as_core": int(anchor_same),
                "shell_same_sign_as_core": int(shell_same),
                "anchor_abs_share_of_core": abs(anchor_val) / max(abs(exact_core), 1e-300),
                "shell_abs_share_of_core": abs(shell_val) / max(abs(exact_core), 1e-300),
            }
        )
    return safe, pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-min", type=float, default=0.55)
    ap.add_argument("--rho-max", type=float, default=0.85)
    ap.add_argument("--rho-step", type=float, default=0.01)
    ap.add_argument("--zeta-min", type=float, default=1.05)
    ap.add_argument("--zeta-max", type=float, default=1.30)
    ap.add_argument("--zeta-step", type=float, default=0.01)
    ap.add_argument("--rho-canonical", type=float, default=0.70)
    ap.add_argument("--zeta-canonical", type=float, default=1.21)
    ap.add_argument("--rho-max-grid", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    R_star, lambda_star = _load_lambda_star()
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
            safe, df = _evaluate_box(fields_by_D, Ds, R_star, float(rho_box), float(zeta_box))
            if not safe:
                scan_rows.append(
                    {
                        "rho_box": float(rho_box),
                        "zeta_box": float(zeta_box),
                        "safe_box": 0,
                        "anchor_floor_abs": np.nan,
                        "max_shell_abs_share_of_core": np.nan,
                    }
                )
                continue
            anchor_floor = float(df["self_anchor"].abs().min())
            max_shell_abs_share = float(df["shell_abs_share_of_core"].max())
            row = {
                "rho_box": float(rho_box),
                "zeta_box": float(zeta_box),
                "safe_box": 1,
                "anchor_floor_abs": anchor_floor,
                "max_shell_abs_share_of_core": max_shell_abs_share,
            }
            scan_rows.append(row)
            if best is None or anchor_floor > best["anchor_floor_abs"]:
                best = row
                best_df = df.copy()

    if best is None:
        raise RuntimeError("No same-sign anchor box found in the scanned range.")

    safe_canonical, canonical_df = _evaluate_box(
        fields_by_D,
        Ds,
        R_star,
        float(args.rho_canonical),
        float(args.zeta_canonical),
    )
    if not safe_canonical:
        raise RuntimeError("Canonical anchor box is not safe on the audited knot set.")

    scan = pd.DataFrame(scan_rows).sort_values(["rho_box", "zeta_box"]).reset_index(drop=True)
    detail = canonical_df.copy()

    core_floor_idx = int(detail["self_core_exact"].abs().idxmin())
    anchor_floor_idx = int(detail["self_anchor"].abs().idxmin())
    anchor_floor = float(detail["self_anchor"].abs().min())
    induced_self_pair_floor = float(lambda_star * anchor_floor)

    summary = pd.DataFrame(
        [
            {
                "R_star": R_star,
                "lambda_star": lambda_star,
                "rho_box_opt": float(best["rho_box"]),
                "zeta_box_opt": float(best["zeta_box"]),
                "anchor_floor_abs_opt": float(best["anchor_floor_abs"]),
                "rho_box_canonical": float(args.rho_canonical),
                "zeta_box_canonical": float(args.zeta_canonical),
                "anchor_floor_abs_canonical": anchor_floor,
                "self_core_exact_floor_abs": float(abs(detail.loc[core_floor_idx, "self_core_exact"])),
                "self_core_exact_floor_D": float(detail.loc[core_floor_idx, "D"]),
                "core_floor_over_anchor_floor": float(abs(detail.loc[core_floor_idx, "self_core_exact"]) / max(anchor_floor, 1e-300)),
                "all_anchor_same_sign_as_core": int((detail["anchor_same_sign_as_core"] == 1).all()),
                "all_shell_same_sign_as_core": int((detail["shell_same_sign_as_core"] == 1).all()),
                "max_shell_abs_share_of_core": float(detail["shell_abs_share_of_core"].max()),
                "induced_self_pair_floor_from_anchor": induced_self_pair_floor,
                "self_pair_exact_floor_abs": float(pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv").iloc[0]["self_exact_floor_abs"]),
                "exact_total_floor_abs": float(pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv").iloc[0]["exact_total_floor_abs"]),
            }
        ]
    )
    summary["self_pair_exact_floor_over_induced"] = summary["self_pair_exact_floor_abs"] / summary["induced_self_pair_floor_from_anchor"]
    summary["exact_total_floor_over_induced"] = summary["exact_total_floor_abs"] / summary["induced_self_pair_floor_from_anchor"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_self_core_anchor_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_self_core_anchor_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_self_core_anchor_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    scan.to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
