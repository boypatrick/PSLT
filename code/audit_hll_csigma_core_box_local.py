#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_csigma_source import build_rows, finite_box_amplitude  # noqa: E402
from extract_chi_localized_2d import PhysicalParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Invert the finite-box core-to-box matching condition pointwise in D "
            "to test whether c_sigma itself is locally stable."
        )
    )
    parser.add_argument(
        "--overlap-detail-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument("--h2", type=float, default=1.0e-3)
    parser.add_argument("--cmin", type=float, default=18.0)
    parser.add_argument("--cmax", type=float, default=28.0)
    parser.add_argument("--nscan", type=int, default=4001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    p = PhysicalParams()

    rows = build_rows(detail, p, args.h2)
    c_grid = np.linspace(args.cmin, args.cmax, int(args.nscan), dtype=float)
    c_ref = float(rows.loc[(rows["D"] - ref_d).abs().idxmin(), "c_sigma_exact"])

    out_rows: list[dict[str, float]] = []
    for _, row in rows.iterrows():
        D = float(row["D"])
        kappa_iso = float(row["kappa_iso"])
        amp_exact = float(row["amp_exact"])
        rho_max = float(row["rho_max"])
        z_margin = float(row["z_margin"])
        sigma_grid = c_grid * (kappa_iso ** -0.25)
        amp_box = np.array(
            [finite_box_amplitude(D, s, rho_max, z_margin) for s in sigma_grid],
            dtype=float,
        )
        log_resid = np.abs(np.log(np.clip(amp_box, 1.0e-300, None)) - np.log(max(amp_exact, 1.0e-300)))
        idx_best = int(np.argmin(log_resid))
        c_local = float(c_grid[idx_best])
        sigma_local = float(sigma_grid[idx_best])
        amp_local = float(amp_box[idx_best])
        out_rows.append(
            {
                "D": D,
                "kappa_iso": kappa_iso,
                "sigma_exact": float(row["sigma_exact"]),
                "c_sigma_exact": float(row["c_sigma_exact"]),
                "amp_exact_disc": amp_exact,
                "c_sigma_box_local": c_local,
                "sigma_box_local": sigma_local,
                "amp_box_local": amp_local,
                "log_amp_resid_local": float(log_resid[idx_best]),
                "rel_amp_resid_local": float(abs(amp_local / max(amp_exact, 1.0e-300) - 1.0)),
                "rel_drift_local_vs_ref": float(abs(c_local / c_ref - 1.0)),
                "rel_drift_local_vs_exact": float(abs(c_local / max(float(row["c_sigma_exact"]), 1.0e-300) - 1.0)),
            }
        )

    out_detail = pd.DataFrame(out_rows).sort_values("D").reset_index(drop=True)
    nearest = out_detail.loc[(out_detail["D"] - ref_d).abs().idxmin()]

    summary = pd.DataFrame(
        [
            {
                "section": "local_core_box_inversion",
                "c_sigma_ref": c_ref,
                "c_sigma_box_local_min": float(out_detail["c_sigma_box_local"].min()),
                "c_sigma_box_local_max": float(out_detail["c_sigma_box_local"].max()),
                "c_sigma_box_local_mean": float(out_detail["c_sigma_box_local"].mean()),
                "c_sigma_box_local_rel_span": float(
                    (out_detail["c_sigma_box_local"].max() - out_detail["c_sigma_box_local"].min())
                    / out_detail["c_sigma_box_local"].mean()
                ),
                "max_rel_drift_local_vs_ref": float(out_detail["rel_drift_local_vs_ref"].max()),
                "mean_rel_drift_local_vs_ref": float(out_detail["rel_drift_local_vs_ref"].mean()),
                "max_rel_drift_local_vs_exact": float(out_detail["rel_drift_local_vs_exact"].max()),
                "mean_rel_drift_local_vs_exact": float(out_detail["rel_drift_local_vs_exact"].mean()),
                "max_rel_amp_resid_local": float(out_detail["rel_amp_resid_local"].max()),
                "mean_rel_amp_resid_local": float(out_detail["rel_amp_resid_local"].mean()),
            },
            {
                "section": "reference_point_local_inversion",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "nearest_detail_D": float(nearest["D"]),
                "c_sigma_exact_ref": float(nearest["c_sigma_exact"]),
                "c_sigma_box_local_ref": float(nearest["c_sigma_box_local"]),
                "rel_drift_ref_local_vs_exact": float(nearest["rel_drift_local_vs_exact"]),
                "amp_exact_disc_ref": float(nearest["amp_exact_disc"]),
                "amp_box_local_ref": float(nearest["amp_box_local"]),
                "rel_amp_resid_local_ref": float(nearest["rel_amp_resid_local"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "local_core_to_box_inversion",
                "formula": "A_mu^box(D;c)=A_mu^disc(D), solved pointwise in D",
                "status": "tests whether the same c_sigma emerges at each audited D",
                "value_ref": float(nearest["c_sigma_box_local"]),
            },
            {
                "candidate": "reference_curvature_calibration",
                "formula": "c_sigma(D_*)=sigma_mu(D_*) * kappa_iso(D_*)^(1/4)",
                "status": "single-point curvature calibration",
                "value_ref": float(nearest["c_sigma_exact"]),
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_csigma_core_box_local_summary.csv"
    detail_out = outdir / "hll_csigma_core_box_local_detail.csv"
    constants_out = outdir / "hll_csigma_core_box_local_constants.csv"
    meta_out = outdir / "hll_csigma_core_box_local_run_meta.json"

    summary.to_csv(summary_out, index=False)
    out_detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "cmin": float(args.cmin),
                "cmax": float(args.cmax),
                "nscan": int(args.nscan),
                "h2": float(args.h2),
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()
