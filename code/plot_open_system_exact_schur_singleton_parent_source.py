#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    summary = pd.read_csv(
        OUT_DIR / "chi_open_system_exact_schur_singleton_parent_source_summary.csv"
    ).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    names = ["plus", "minus", "self-pair", "cross", "core", "req global", "req local"]
    vals = [
        float(summary["plus_core"]),
        float(summary["minus_core"]),
        float(summary["self_pair_core"]),
        float(summary["cross_core"]),
        float(summary["total_core"]),
        float(summary["required_core_floor_global"]),
        float(summary["required_core_floor_local"]),
    ]
    colors = ["#6BAED6", "#9ECAE1", "#2A9D8F", "#E9C46A", "#264653", "#457B9D", "#E63946"]
    ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.93)
    ax.set_xticks(np.arange(len(names)), names, rotation=20)
    ax.set_ylabel("magnitude")
    ax.set_title(r"Singleton $D=11$ Parent Decomposition")
    ax.text(
        0.02,
        0.97,
        rf"$\eta_{{11}}={float(summary['cross_abs_share_of_total_core']):.4f}$ cross share" "\n"
        rf"self-pair / req$_g$ = {float(summary['self_pair_over_required_global']):.4f}" "\n"
        rf"self-pair / req$_l$ = {float(summary['self_pair_over_required_local']):.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    ax = axes[1]
    reqs = [
        float(summary["required_core_floor_global"]),
        float(summary["required_core_floor_local"]),
    ]
    actuals = [
        float(summary["self_pair_core"]),
        float(summary["total_core"]),
    ]
    labels = ["req global", "req local", "self-pair", "exact core"]
    vals = reqs + actuals
    ax.bar(np.arange(len(labels)), vals, color=["#457B9D", "#E63946", "#2A9D8F", "#264653"], alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
    ax.set_ylabel("core floor scale")
    ax.set_title(r"What Still Needs a Proof at $D=11$")
    ax.axhline(float(summary["self_pair_core"]), color="#2A9D8F", linestyle="--", linewidth=1.6)
    ax.axhline(float(summary["total_core"]), color="#264653", linestyle=":", linewidth=1.8)

    out = OUT_DIR / "chi_open_system_exact_schur_singleton_parent_source_figure.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
