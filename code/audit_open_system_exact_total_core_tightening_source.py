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

    fields_by_D: dict[float, dict[str, np.ndarray]] = {}
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
    best_detail: pd.DataFrame | None = None
    R_values = np.arange(float(args.Rmin), float(args.Rmax) + 0.5 * float(args.Rstep), float(args.Rstep))
    for R in R_values:
        per_D = []
        safe = True
        for D in Ds:
            f = fields_by_D[D]
            rr = f["rr"]
            zz = f["zz"]
            contrast = f["contrast_density"]
            exact_density = contrast * (f["delta_plus"] + f["delta_minus"] + f["delta_cross"])
            exact_total = float(np.sum(exact_density))
            sign_total = 1.0 if exact_total >= 0.0 else -1.0

            rp = np.sqrt(rr * rr + (zz - D / 2.0) ** 2)
            rm = np.sqrt(rr * rr + (zz + D / 2.0) ** 2)
            core = (rp <= R) | (rm <= R)

            exact_core = float(np.sum(exact_density[core]))
            exact_tail = float(np.sum(exact_density[~core]))

            core_same_sign = exact_core == 0.0 or np.sign(exact_core) == sign_total
            tail_opposite_sign = exact_tail == 0.0 or np.sign(exact_tail) == -sign_total
            if not (core_same_sign and tail_opposite_sign):
                safe = False
                break

            core_abs = abs(exact_core)
            tail_abs = abs(exact_tail)
            exact_abs = abs(exact_total)
            lam = exact_abs / max(core_abs, 1e-300)
            eta = tail_abs / max(core_abs, 1e-300)
            per_D.append(
                {
                    "D": D,
                    "R": float(R),
                    "exact_total": exact_total,
                    "exact_core": exact_core,
                    "exact_tail": exact_tail,
                    "core_same_sign_as_total": int(core_same_sign),
                    "tail_opposite_sign_to_total": int(tail_opposite_sign),
                    "lambda_total_over_core": lam,
                    "eta_total_tail_over_core": eta,
                    "core_abs": core_abs,
                    "tail_abs": tail_abs,
                }
            )

        if not safe:
            scan_rows.append(
                {
                    "R": float(R),
                    "safe_window": 0,
                    "induced_total_floor": np.nan,
                    "lambda_total_min": np.nan,
                    "core_floor_abs": np.nan,
                    "tail_over_core_max": np.nan,
                }
            )
            continue

        df = pd.DataFrame(per_D).sort_values("D").reset_index(drop=True)
        lambda_total_min = float(df["lambda_total_over_core"].min())
        core_floor_abs = float(df["core_abs"].min())
        tail_over_core_max = float(df["eta_total_tail_over_core"].max())
        induced_total_floor = float(lambda_total_min * core_floor_abs)

        row = {
            "R": float(R),
            "safe_window": 1,
            "induced_total_floor": induced_total_floor,
            "lambda_total_min": lambda_total_min,
            "core_floor_abs": core_floor_abs,
            "tail_over_core_max": tail_over_core_max,
        }
        scan_rows.append(row)
        if best is None or induced_total_floor > best["induced_total_floor"]:
            best = row
            best_detail = df.copy()

    if best is None or best_detail is None:
        raise RuntimeError("No safe exact-total core radius found in the scanned range.")

    detail = best_detail.copy()
    scan = pd.DataFrame(scan_rows).sort_values("R").reset_index(drop=True)

    floor_idx = int(detail["exact_total"].abs().idxmin())
    summary = pd.DataFrame(
        [
            {
                "R_star": float(best["R"]),
                "lambda_total_star": float(best["lambda_total_min"]),
                "eta_total_star": float(best["tail_over_core_max"]),
                "exact_core_floor_abs_at_R_star": float(best["core_floor_abs"]),
                "induced_exact_total_floor": float(best["induced_total_floor"]),
                "exact_total_floor_abs": float(abs(detail.loc[floor_idx, "exact_total"])),
                "exact_total_floor_D": float(detail.loc[floor_idx, "D"]),
                "exact_total_floor_over_induced": float(
                    abs(detail.loc[floor_idx, "exact_total"]) / max(float(best["induced_total_floor"]), 1e-300)
                ),
                "all_core_same_sign_as_total": int((detail["core_same_sign_as_total"] == 1).all()),
                "all_tail_opposite_sign_to_total": int((detail["tail_opposite_sign_to_total"] == 1).all()),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_exact_total_core_tightening_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_exact_total_core_tightening_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_exact_total_core_tightening_source_scan.csv"

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    scan.to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
