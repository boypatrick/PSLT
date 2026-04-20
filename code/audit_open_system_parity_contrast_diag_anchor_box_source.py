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


def _eval_box(
    fields_by_D: dict[float, dict[str, np.ndarray]],
    Ds: list[float],
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

        plus_anchor = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
        minus_anchor = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_anchor)
        plus_box = (rr <= rho_box) & (np.abs(zz - D / 2.0) <= zeta_box)
        minus_box = (rr <= rho_box) & (np.abs(zz + D / 2.0) <= zeta_box)
        plus_box &= plus_anchor
        minus_box &= minus_anchor

        diag_anchor = float(
            np.sum(C[plus_anchor] * f["delta_plus"][plus_anchor])
            + np.sum(C[minus_anchor] * f["delta_minus"][minus_anchor])
        )
        diag_box = float(
            np.sum(C[plus_box] * f["delta_plus"][plus_box])
            + np.sum(C[minus_box] * f["delta_minus"][minus_box])
        )
        diag_shell = diag_anchor - diag_box
        sign_diag = 1.0 if diag_anchor >= 0.0 else -1.0

        box_same = diag_box == 0.0 or np.sign(diag_box) == sign_diag
        shell_same = diag_shell == 0.0 or np.sign(diag_shell) == sign_diag
        safe = safe and box_same and shell_same

        rows.append(
            {
                "D": D,
                "rho_anchor": rho_anchor,
                "zeta_anchor": zeta_anchor,
                "rho_box": rho_box,
                "zeta_box": zeta_box,
                "diag_anchor": diag_anchor,
                "diag_box": diag_box,
                "diag_shell": diag_shell,
                "box_same_sign_as_diag": int(box_same),
                "shell_same_sign_as_diag": int(shell_same),
                "box_abs_share_of_diag": abs(diag_box) / max(abs(diag_anchor), 1e-300),
                "shell_abs_share_of_diag": abs(diag_shell) / max(abs(diag_anchor), 1e-300),
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
    detail_best = None
    rho_values = np.arange(float(args.rho_min), float(args.rho_max) + 0.5 * float(args.rho_step), float(args.rho_step))
    zeta_values = np.arange(float(args.zeta_min), float(args.zeta_max) + 0.5 * float(args.zeta_step), float(args.zeta_step))
    for rho_box in rho_values:
        for zeta_box in zeta_values:
            # exclude trivial full anchor; we want a strict subbox
            if abs(float(rho_box) - float(args.rho_anchor)) < 5e-12 and abs(float(zeta_box) - float(args.zeta_anchor)) < 5e-12:
                continue
            safe, df = _eval_box(
                fields_by_D,
                Ds,
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
                        "max_shell_abs_share_of_diag": np.nan,
                    }
                )
                continue
            box_floor = float(df["diag_box"].abs().min())
            max_shell_abs_share = float(df["shell_abs_share_of_diag"].max())
            row = {
                "rho_box": float(rho_box),
                "zeta_box": float(zeta_box),
                "safe_box": 1,
                "diag_box_floor_abs": box_floor,
                "max_shell_abs_share_of_diag": max_shell_abs_share,
            }
            scan_rows.append(row)
            cand = (box_floor, -max_shell_abs_share, float(rho_box), float(zeta_box))
            if best is None or cand > best:
                best = cand
                detail_best = df.copy()

    if best is None or detail_best is None:
        raise RuntimeError("No strict same-center subbox found inside canonical anchor.")

    scan = pd.DataFrame(scan_rows).sort_values(["rho_box", "zeta_box"]).reset_index(drop=True)
    detail = detail_best.copy()
    diag_floor_idx = int(detail["diag_box"].abs().idxmin())
    anchor_floor = float(detail["diag_anchor"].abs().min())
    box_floor = float(detail["diag_box"].abs().min())

    summary = pd.DataFrame(
        [
            {
                "rho_anchor": float(args.rho_anchor),
                "zeta_anchor": float(args.zeta_anchor),
                "rho_box_opt": float(best[2]),
                "zeta_box_opt": float(best[3]),
                "diag_box_floor_abs_opt": box_floor,
                "diag_box_floor_D": float(detail.loc[diag_floor_idx, "D"]),
                "diag_anchor_floor_abs": anchor_floor,
                "diag_anchor_floor_over_box_floor": anchor_floor / max(box_floor, 1e-300),
                "all_box_same_sign_as_diag": int((detail["box_same_sign_as_diag"] == 1).all()),
                "all_shell_same_sign_as_diag": int((detail["shell_same_sign_as_diag"] == 1).all()),
                "max_shell_abs_share_of_diag": float(detail["shell_abs_share_of_diag"].max()),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_box_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_box_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_box_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    scan.to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
