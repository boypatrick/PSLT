#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
DETAIL = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_detail.csv"
SUMMARY = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--Rmin", type=float, default=1.20)
    ap.add_argument("--Rmax", type=float, default=1.40)
    ap.add_argument("--Rstep", type=float, default=0.01)
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))

    fields_by_D = {}
    for D in Ds:
        fields_by_D[D] = solve_fields(
            D=D,
            p=p,
            level=level,
            rho_max=float(args.rho_max),
            z_margin=float(args.z_margin),
            sigma=float(args.sigma),
            tol=float(args.tol),
            maxiter=int(args.maxiter),
        )

    scan_rows = []
    best = None
    R_values = np.arange(float(args.Rmin), float(args.Rmax) + 0.5 * float(args.Rstep), float(args.Rstep))
    for R in R_values:
        per_D = []
        safe = True
        for D in Ds:
            f = fields_by_D[D]
            rr = f["rr"]
            zz = f["zz"]
            contrast = f["contrast_density"]
            self_pair = contrast * (f["delta_plus"] + f["delta_minus"])
            exact = float(np.sum(self_pair))
            sign_exact = 1.0 if exact >= 0.0 else -1.0

            rp = np.sqrt(rr * rr + (zz - D / 2.0) ** 2)
            rm = np.sqrt(rr * rr + (zz + D / 2.0) ** 2)
            core = (rp <= R) | (rm <= R)
            core_val = float(np.sum(self_pair[core]))
            tail_val = float(np.sum(self_pair[~core]))

            core_same_sign = core_val == 0.0 or np.sign(core_val) == sign_exact
            tail_opposite_sign = tail_val == 0.0 or np.sign(tail_val) == -sign_exact
            if not (core_same_sign and tail_opposite_sign):
                safe = False
                break

            core_abs = abs(core_val)
            exact_abs = abs(exact)
            tail_abs = abs(tail_val)
            lam = exact_abs / max(core_abs, 1e-300)
            eta = tail_abs / max(core_abs, 1e-300)
            per_D.append(
                {
                    "D": D,
                    "R": R,
                    "self_pair_exact": exact,
                    "self_core": core_val,
                    "self_tail": tail_val,
                    "lambda_exact_over_core": lam,
                    "eta_tail_over_core": eta,
                    "core_abs": core_abs,
                    "tail_abs": tail_abs,
                }
            )

        if not safe:
            scan_rows.append(
                {
                    "R": R,
                    "safe_window": 0,
                    "induced_self_floor": np.nan,
                    "lambda_min": np.nan,
                    "core_floor_abs": np.nan,
                    "tail_over_core_max": np.nan,
                }
            )
            continue

        scan_df = pd.DataFrame(per_D)
        lambda_min = float(scan_df["lambda_exact_over_core"].min())
        core_floor_abs = float(scan_df["core_abs"].min())
        tail_over_core_max = float(scan_df["eta_tail_over_core"].max())
        induced_self_floor = float(lambda_min * core_floor_abs)
        row = {
            "R": float(R),
            "safe_window": 1,
            "induced_self_floor": induced_self_floor,
            "lambda_min": lambda_min,
            "core_floor_abs": core_floor_abs,
            "tail_over_core_max": tail_over_core_max,
        }
        scan_rows.append(row)
        if best is None or induced_self_floor > best["induced_self_floor"]:
            best = row
            best_df = scan_df.copy()

    if best is None:
        raise RuntimeError("No safe core radius found in the scanned range.")

    best_R = float(best["R"])
    detail_rows = []
    for D in Ds:
        f = fields_by_D[D]
        contrast = f["contrast_density"]
        self_pair = contrast * (f["delta_plus"] + f["delta_minus"])
        cross = contrast * f["delta_cross"]
        exact_total = float(np.sum(self_pair + cross))
        self_exact = float(np.sum(self_pair))
        cross_exact = float(np.sum(cross))
        rp = np.sqrt(f["rr"] * f["rr"] + (f["zz"] - D / 2.0) ** 2)
        rm = np.sqrt(f["rr"] * f["rr"] + (f["zz"] + D / 2.0) ** 2)
        core = (rp <= best_R) | (rm <= best_R)
        self_core = float(np.sum(self_pair[core]))
        self_tail = float(np.sum(self_pair[~core]))
        detail_rows.append(
            {
                "D": D,
                "R_star": best_R,
                "exact_total": exact_total,
                "self_pair_exact": self_exact,
                "self_core": self_core,
                "self_tail": self_tail,
                "cross_exact": cross_exact,
                "self_core_same_sign_as_self": int(self_core == 0.0 or np.sign(self_core) == np.sign(self_exact)),
                "self_tail_opposite_sign_to_self": int(self_tail == 0.0 or np.sign(self_tail) == -np.sign(self_exact)),
                "cross_same_sign_as_total": int(cross_exact == 0.0 or np.sign(cross_exact) == np.sign(exact_total)),
                "lambda_exact_over_core": abs(self_exact) / max(abs(self_core), 1e-300),
                "eta_tail_over_core": abs(self_tail) / max(abs(self_core), 1e-300),
                "self_core_abs_share_of_self": abs(self_core) / max(abs(self_exact), 1e-300),
                "self_tail_abs_share_of_self": abs(self_tail) / max(abs(self_exact), 1e-300),
                "cross_abs_share_of_total": abs(cross_exact) / max(abs(exact_total), 1e-300),
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values("D").reset_index(drop=True)
    scan = pd.DataFrame(scan_rows).sort_values("R").reset_index(drop=True)

    self_floor_idx = int(detail["self_pair_exact"].abs().idxmin())
    total_floor_idx = int(detail["exact_total"].abs().idxmin())

    summary = pd.DataFrame(
        [
            {
                "R_star": best_R,
                "lambda_star": float(best["lambda_min"]),
                "eta_star": float(best["tail_over_core_max"]),
                "self_core_floor_abs_at_R_star": float(best["core_floor_abs"]),
                "induced_self_floor": float(best["induced_self_floor"]),
                "self_exact_floor_abs": float(abs(detail.loc[self_floor_idx, "self_pair_exact"])),
                "self_exact_floor_D": float(detail.loc[self_floor_idx, "D"]),
                "self_floor_over_induced": float(abs(detail.loc[self_floor_idx, "self_pair_exact"]) / max(float(best["induced_self_floor"]), 1e-300)),
                "exact_total_floor_abs": float(abs(detail.loc[total_floor_idx, "exact_total"])),
                "exact_total_floor_D": float(detail.loc[total_floor_idx, "D"]),
                "exact_total_floor_over_induced": float(abs(detail.loc[total_floor_idx, "exact_total"]) / max(float(best["induced_self_floor"]), 1e-300)),
                "max_cross_abs_share_of_total": float(detail["cross_abs_share_of_total"].max()),
                "all_cross_same_sign_as_total": int((detail["cross_same_sign_as_total"] == 1).all()),
                "all_core_same_sign_as_self": int((detail["self_core_same_sign_as_self"] == 1).all()),
                "all_tail_opposite_sign_to_self": int((detail["self_tail_opposite_sign_to_self"] == 1).all()),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_self_floor_source_scan.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    scan.to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
