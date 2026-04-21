#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTDIR / name).sort_values("D").reset_index(drop=True)


def main() -> None:
    band_mag = load("chi_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source_detail.csv")[
        ["D", "N_mass_band", "kernel_avg_weighted"]
    ]
    band_neg = load("chi_open_system_parity_contrast_diag_anchor_band_negative_carrier_source_detail.csv")[
        ["D", "W_band_neg", "M_band_neg", "rho_bar_weighted"]
    ]
    band_overlap = load("chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_detail.csv")[
        ["D", "M_band_abs", "theta_neg"]
    ]
    mirror_avg = load("chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_detail.csv")[
        ["D", "A_band", "S_self", "self_bias"]
    ]
    mirror_ratio = load("chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_detail.csv")[
        ["D", "self_mean", "mirror_mean", "mir_self_ratio"]
    ]
    neg = load("chi_open_system_parity_contrast_diag_anchor_mass_negative_source_detail.csv")[
        ["D", "N_mass_inner", "P_mass_inner", "canonical_N_mass_tail", "canonical_tail_over_P_mass_inner"]
    ]
    shoulder = load("chi_open_system_parity_contrast_diag_anchor_mass_shoulder_source_detail.csv")[
        ["D", "canonical_N_mass_rim", "canonical_rim_over_P_mass_inner"]
    ]
    band = load("chi_open_system_parity_contrast_diag_anchor_mass_shoulder_band_source_detail.csv")[
        ["D", "canonical_N_mass_core", "canonical_core_over_P_mass_inner", "canonical_N_mass_band"]
    ]

    detail = band_mag.merge(band_neg, on="D")
    detail = detail.merge(band_overlap, on="D")
    detail = detail.merge(mirror_avg, on="D")
    detail = detail.merge(mirror_ratio, on="D")
    detail = detail.merge(neg, on="D")
    detail = detail.merge(shoulder, on="D")
    detail = detail.merge(band, on="D")

    detail["band_mainline_exact"] = (
        4.0
        * math.pi
        * detail["kernel_avg_weighted"]
        * detail["rho_bar_weighted"]
        * detail["theta_neg"]
        * detail["self_bias"]
        * detail["mir_self_ratio"]
        * (detail["S_self"] ** 2)
        / detail["A_band"]
    )
    detail["band_mainline_resid_abs"] = (detail["N_mass_band"] - detail["band_mainline_exact"]).abs()

    detail["band_over_P_mass_inner"] = detail["canonical_N_mass_band"] / detail["P_mass_inner"]
    detail["mainline_over_P_mass_inner"] = detail["band_mainline_exact"] / detail["P_mass_inner"]
    detail["nu_mass_inner_exact"] = detail["N_mass_inner"] / detail["P_mass_inner"]
    detail["nu_mass_inner_reconstructed"] = (
        detail["mainline_over_P_mass_inner"]
        + detail["canonical_core_over_P_mass_inner"]
        + detail["canonical_rim_over_P_mass_inner"]
        + detail["canonical_tail_over_P_mass_inner"]
    )
    detail["nu_mass_inner_resid_abs"] = (
        detail["nu_mass_inner_exact"] - detail["nu_mass_inner_reconstructed"]
    ).abs()

    band_floor_idx = int(detail["N_mass_band"].idxmin())
    nu_worst_idx = int(detail["nu_mass_inner_exact"].idxmax())
    mirror_floor_idx = int(detail["mirror_mean"].idxmin())
    self_floor_idx = int(detail["self_mean"].idxmin())
    ratio_floor_idx = int(detail["mir_self_ratio"].idxmin())

    summary = pd.DataFrame(
        [
            {
                "max_band_mainline_resid_abs": float(detail["band_mainline_resid_abs"].max()),
                "max_nu_reconstruction_resid_abs": float(detail["nu_mass_inner_resid_abs"].max()),
                "band_floor_abs": float(detail["N_mass_band"].min()),
                "band_floor_D": float(detail.loc[band_floor_idx, "D"]),
                "nu_mass_inner_max": float(detail["nu_mass_inner_exact"].max()),
                "nu_mass_inner_max_D": float(detail.loc[nu_worst_idx, "D"]),
                "mirror_mean_floor_abs": float(detail["mirror_mean"].min()),
                "mirror_mean_floor_D": float(detail.loc[mirror_floor_idx, "D"]),
                "self_mean_floor_abs": float(detail["self_mean"].min()),
                "self_mean_floor_D": float(detail.loc[self_floor_idx, "D"]),
                "mir_self_ratio_floor_abs": float(detail["mir_self_ratio"].min()),
                "mir_self_ratio_floor_D": float(detail.loc[ratio_floor_idx, "D"]),
                "closed_multiplier_knots_align": int(
                    float(detail.loc[mirror_floor_idx, "D"])
                    == float(detail.loc[self_floor_idx, "D"])
                    == float(detail.loc[ratio_floor_idx, "D"])
                ),
                "max_core_over_P": float(detail["canonical_core_over_P_mass_inner"].max()),
                "max_rim_over_P": float(detail["canonical_rim_over_P_mass_inner"].max()),
                "max_tail_over_P": float(detail["canonical_tail_over_P_mass_inner"].max()),
                "max_mainline_over_P": float(detail["mainline_over_P_mass_inner"].max()),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_parity_contrast_synthesis_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_synthesis_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
