#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

TMP_MPL = Path("/tmp/pslt_mplconfig")
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPL))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"

SUMMARY = pd.read_csv(OUTDIR / "runtime_direct_detlin_generalization_fast_summary.csv")
DETAIL = pd.read_csv(OUTDIR / "runtime_direct_detlin_generalization_fast_detail.csv")

OUT_PNG = OUTDIR / "runtime_direct_detlin_generalization_fast.png"
OUT_META = OUTDIR / "runtime_direct_detlin_generalization_fast_run_meta.json"


def main() -> None:
    case_order = list(SUMMARY["case"])
    label_map = dict(zip(SUMMARY["case"], SUMMARY["case_label"]))
    target_order = sorted(float(v) for v in DETAIL["target_D"].unique())

    x = np.arange(len(case_order))
    labels = [label_map[c] for c in case_order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    axes[0].bar(x, SUMMARY["p95_abs_delta_mu_mumu"], color="#4e79a7")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Broader-grid p95 |Δmu_mumu|")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, SUMMARY["acceptance_mismatch"], color="#e15759")
    axes[1].set_title("Broader-grid acceptance mismatch")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].grid(alpha=0.25, axis="y")

    width = 0.36
    focus = (
        DETAIL.groupby(["case", "target_D"])
        .agg(
            p95_abs_delta_mu_mumu=(
                "abs_delta_mu_mumu",
                lambda s: float(np.percentile(s, 95.0)),
            ),
        )
        .reset_index()
    )
    colors = ["#59a14f", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1"]
    for idx, target_D in enumerate(target_order):
        sub = focus[np.isclose(focus["target_D"], target_D)].copy()
        sub["case"] = pd.Categorical(sub["case"], categories=case_order, ordered=True)
        sub = sub.sort_values("case")
        offset = (idx - (len(target_order) - 1) / 2.0) * width / 1.8
        axes[2].bar(
            x + offset,
            sub["p95_abs_delta_mu_mumu"],
            width=width / 1.8,
            label=f"D={target_D:.1f}",
            color=colors[idx % len(colors)],
        )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("Slice p95 |Δmu_mumu| by target D")
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend(fontsize=7, ncol=2)

    fig.suptitle("Strict all-direct broader-grid generalization check")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    OUT_META.write_text(
        json.dumps(
            {
                "summary_rows": SUMMARY.to_dict(orient="records"),
                "target_order": target_order,
            },
            indent=2,
        )
    )
    for path in [OUT_PNG, OUT_META]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
