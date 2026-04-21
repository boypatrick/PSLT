#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
DETAIL = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_detail.csv"
SUMMARY = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_summary.csv"
FIG = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_figure.png"


def main() -> None:
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY).iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(detail["D"], detail["mir_self_ratio"], marker="o", label=r"$S_{\rm mir}/S_{\rm self}$")
    ax.axhline(summary["mir_self_ratio_min"], color="tab:red", linestyle="--", alpha=0.7, label="ratio floor")
    ax.set_xlabel("D")
    ax.set_ylabel("ratio")
    ax.set_title("Mirror/Self Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(detail["D"], detail["mirror_mean"], marker="o", label=r"$\bar\Phi_{\rm band}^{(\rm mir)}$")
    ax.plot(detail["D"], detail["factorized_mean"], marker="s", linestyle="--", label=r"$(S_{\rm mir}/S_{\rm self})\bar\Phi_{\rm self}$")
    ax.axhline(summary["mirror_mean_floor_abs"], color="tab:red", linestyle=":", alpha=0.7, label="exact floor")
    ax.set_xlabel("D")
    ax.set_ylabel("mean amplitude")
    ax.set_title("Exact Factorization")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(detail["D"], detail["self_mean"], marker="o", label=r"$\bar\Phi_{\rm self}$")
    ax.plot(detail["D"], detail["mirror_mean"], marker="s", label=r"$\bar\Phi_{\rm mir}$")
    ax.axvline(summary["mirror_mean_floor_D"], color="tab:red", linestyle="--", alpha=0.7, label="floor knot")
    ax.set_xlabel("D")
    ax.set_ylabel("band mean")
    ax.set_title("Self vs Mirror Means")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(FIG)


if __name__ == "__main__":
    main()
