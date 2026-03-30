#!/usr/bin/env python3
"""
Assess twolobe_d8compd60snapmaplate as a promotion-ready descendant candidate.

This is a decision layer. It consumes the existing runtime-validation summary
and evaluates explicit gates for:
  - preserving reviewer-target gains,
  - preserving D21 complement closure,
  - improving dense-grid D60 drift strips,
  - avoiding new parent-relative regressions away from the intended drift strip.

Outputs:
  - output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.csv
  - output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.json
  - output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.png
  - paper/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

SUMMARY_CSV = (
    OUTDIR / "observable_two_lobe_d8compd60snapmaplate_runtime_validation_summary.csv"
)


def _pick(summary: pd.DataFrame, case: str, subset: str) -> pd.Series:
    mask = (summary["case"] == case) & (summary["subset"] == subset)
    sub = summary.loc[mask]
    if sub.empty:
        raise KeyError(f"Missing row case={case!r}, subset={subset!r}")
    return sub.iloc[0]


def _gate(name: str, passed: bool, actual: float, threshold: float, relation: str, note: str) -> Dict[str, object]:
    return {
        "gate": name,
        "passed": int(bool(passed)),
        "actual": float(actual),
        "threshold": float(threshold),
        "relation": relation,
        "note": note,
    }


def _to_bool(gates: Iterable[Dict[str, object]]) -> int:
    return int(all(bool(g["passed"]) for g in gates))


def _plot(gates_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    order = gates_df.index.to_numpy()
    values = gates_df["margin"].to_numpy(dtype=float)
    colors = ["#59a14f" if p else "#e15759" for p in gates_df["passed"].astype(bool)]
    ax.barh(order, values, color=colors)
    ax.axvline(0.0, color="k", lw=0.9, alpha=0.5)
    ax.set_yticks(order)
    ax.set_yticklabels(gates_df["gate"])
    ax.set_xlabel("Gate margin (positive passes)")
    ax.set_title("twolobe_d8compd60snapmaplate Promotion Gate Checklist")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Promotion gate checklist for twolobe_d8compd60snapmaplate.")
    ap.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    ap.add_argument("--tol-preserve", type=float, default=1.0e-9)
    ap.add_argument("--tol-parent-regression", type=float, default=1.0e-9)
    ap.add_argument("--min-drift-improvement", type=float, default=1.0e-6)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary_csv)

    gates: list[Dict[str, object]] = []

    preserve_pairs = [
        ("D60E21_fullmap", "reviewer_targets", "Preserve D60 canonical reviewer"),
        ("D60E41_holdout", "reviewer_targets", "Preserve D60 holdout reviewer"),
        ("D21E41_fullmap", "reviewer_targets", "Preserve D21 canonical reviewer"),
        ("D21E21_holdout", "reviewer_targets", "Preserve D21 holdout reviewer"),
        ("D21E41_fullmap", "nonreviewer_complement", "Preserve D21 canonical complement"),
        ("D21E21_holdout", "nonreviewer_complement", "Preserve D21 holdout complement"),
        ("D21E41_fullmap", "all_points", "Preserve D21 canonical all-points"),
        ("D21E21_holdout", "all_points", "Preserve D21 holdout all-points"),
    ]
    for case, subset, label in preserve_pairs:
        row = _pick(summary, case, subset)
        actual = float(row["p95_abs_delta_cf"] - row["p95_abs_delta_parent"])
        threshold = float(args.tol_preserve)
        gates.append(
            _gate(
                label,
                actual <= threshold,
                actual=actual,
                threshold=threshold,
                relation="<=",
                note=f"{case}/{subset} candidate p95 should not exceed parent p95",
            )
        )

    drift_pairs = [
        ("D60E21_fullmap", "drift_strip", "Improve D60 canonical drift strip"),
        ("D60E41_holdout", "drift_strip", "Improve D60 holdout drift strip"),
        ("D80E41_monitor", "drift_strip", "Improve D80 holdout drift strip"),
        ("D100E41_monitor", "drift_strip", "Improve D100 holdout drift strip"),
    ]
    for case, subset, label in drift_pairs:
        row = _pick(summary, case, subset)
        actual = float(row["p95_abs_delta_parent"] - row["p95_abs_delta_cf"])
        threshold = float(args.min_drift_improvement)
        gates.append(
            _gate(
                label,
                actual >= threshold,
                actual=actual,
                threshold=threshold,
                relation=">=",
                note=f"{case}/{subset} candidate should improve parent drift p95",
            )
        )

    regression_pairs = [
        ("D60E21_fullmap", "nonreviewer_complement", "No new D60 canonical complement regression"),
        ("D60E41_holdout", "nonreviewer_complement", "No new D60 holdout complement regression"),
        ("D80E41_monitor", "nonreviewer_complement", "No new D80 holdout complement regression"),
        ("D100E41_monitor", "nonreviewer_complement", "No new D100 holdout complement regression"),
    ]
    for case, subset, label in regression_pairs:
        row = _pick(summary, case, subset)
        actual = float(row["worst_point_improvement_vs_parent"])
        threshold = -float(args.tol_parent_regression)
        gates.append(
            _gate(
                label,
                actual >= threshold,
                actual=actual,
                threshold=threshold,
                relation=">=",
                note=f"{case}/{subset} worst point should stay within parent-regression tolerance",
            )
        )

    flip_pairs = [
        ("D21E41_fullmap", "all_points", "No new D21 canonical acceptance flips"),
        ("D21E21_holdout", "all_points", "No new D21 holdout acceptance flips"),
        ("D60E21_fullmap", "reviewer_targets", "No new D60 canonical reviewer flips"),
        ("D60E41_holdout", "reviewer_targets", "No new D60 holdout reviewer flips"),
    ]
    for case, subset, label in flip_pairs:
        row = _pick(summary, case, subset)
        actual = float(row["acceptance_flip_fraction_vs_base"])
        threshold = 0.0
        gates.append(
            _gate(
                label,
                np.isclose(actual, threshold, atol=1.0e-12),
                actual=actual,
                threshold=threshold,
                relation="==",
                note=f"{case}/{subset} acceptance flips vs base must stay zero",
            )
        )

    gates_df = pd.DataFrame(gates)
    margin = []
    for _, row in gates_df.iterrows():
        if row["relation"] == "<=":
            margin.append(float(row["threshold"] - row["actual"]))
        elif row["relation"] == ">=":
            margin.append(float(row["actual"] - row["threshold"]))
        else:
            margin.append(float(-abs(row["actual"] - row["threshold"])))
    gates_df["margin"] = margin

    go_flag = _to_bool(gates)
    decision = "GO_promotion_gate_closed" if go_flag else "NO_GO_keep_monitoring"

    summary_out: Dict[str, object] = {
        "candidate_mode": "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compd60snapmaplate",
        "parent_mode": "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate",
        "decision": decision,
        "go_flag": go_flag,
        "n_gates": int(len(gates_df)),
        "n_passed": int(gates_df["passed"].sum()),
        "n_failed": int((1 - gates_df["passed"]).sum()),
        "tol_preserve": float(args.tol_preserve),
        "tol_parent_regression": float(args.tol_parent_regression),
        "min_drift_improvement": float(args.min_drift_improvement),
        "d60e21_drift_parent": float(_pick(summary, "D60E21_fullmap", "drift_strip")["p95_abs_delta_parent"]),
        "d60e21_drift_candidate": float(_pick(summary, "D60E21_fullmap", "drift_strip")["p95_abs_delta_cf"]),
        "d100e41_drift_parent": float(_pick(summary, "D100E41_monitor", "drift_strip")["p95_abs_delta_parent"]),
        "d100e41_drift_candidate": float(_pick(summary, "D100E41_monitor", "drift_strip")["p95_abs_delta_cf"]),
        "d21e41_complement_parent": float(_pick(summary, "D21E41_fullmap", "nonreviewer_complement")["p95_abs_delta_parent"]),
        "d21e41_complement_candidate": float(_pick(summary, "D21E41_fullmap", "nonreviewer_complement")["p95_abs_delta_cf"]),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    stem = "observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist"
    out_csv = OUTDIR / f"{stem}.csv"
    out_json = OUTDIR / f"{stem}.json"
    out_gates_csv = OUTDIR / f"{stem}_gates.csv"
    out_png = OUTDIR / f"{stem}.png"
    paper_csv = PAPER_DIR / f"{stem}.csv"
    paper_gates_csv = PAPER_DIR / f"{stem}_gates.csv"

    pd.DataFrame([summary_out]).to_csv(out_csv, index=False)
    gates_df.to_csv(out_gates_csv, index=False)
    out_json.write_text(json.dumps({"summary": summary_out, "gates": gates}, indent=2))
    paper_csv.write_text(out_csv.read_text())
    paper_gates_csv.write_text(out_gates_csv.read_text())
    _plot(gates_df, out_png)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_gates_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_png}")
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
