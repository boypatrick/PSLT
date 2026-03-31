#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "observable_two_lobe_d8compd60snapmaplate_publish_switch_checklist"
TOL_EQUAL = 1e-9
TOL_PUBLISH_NO_WORSEN = 1e-5

VALIDATION_PATH = OUTDIR / "observable_two_lobe_d8compd60snapmaplate_runtime_validation_summary.csv"
PROMOTION_PATH = OUTDIR / "observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.json"


def _row(df: pd.DataFrame, case: str, subset: str) -> pd.Series:
    mask = (df["case"] == case) & (df["subset"] == subset)
    matches = df.loc[mask]
    if matches.empty:
        raise KeyError(f"missing {case}/{subset}")
    return matches.iloc[0]


def _gate(name: str, passed: bool, actual: float | int, threshold: float | int, relation: str, note: str) -> dict[str, object]:
    return {
        "gate": name,
        "passed": int(bool(passed)),
        "actual": float(actual) if isinstance(actual, (int, float)) else actual,
        "threshold": float(threshold) if isinstance(threshold, (int, float)) else threshold,
        "relation": relation,
        "note": note,
    }


def _plot(gates_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = ["#59a14f" if int(x) else "#e15759" for x in gates_df["passed"]]
    ax.barh(gates_df["gate"], gates_df["passed"], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("pass")
    ax.set_title("D8compd60 Publish Switch Checklist")
    for idx, row in enumerate(gates_df.itertuples(index=False)):
        ax.text(1.01, idx, f"{row.relation} {row.threshold}", va="center", fontsize=8)
    png_path = OUTDIR / f"{STEM}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    (PAPER_DIR / png_path.name).write_bytes(png_path.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(VALIDATION_PATH)
    with PROMOTION_PATH.open("r", encoding="utf-8") as fh:
        promotion = json.load(fh)

    gates: list[dict[str, object]] = []

    promo_go = int(promotion["summary"]["go_flag"])
    gates.append(
        _gate(
            "Upstream promotion gate already closed",
            promo_go == 1,
            promo_go,
            1,
            "==",
            "Previous explicit promotion gate should already be GO before publish-switch assessment.",
        )
    )

    for case, subset, label in (
        ("D60E21_fullmap", "all_points", "Improve D60 canonical all-points vs canonical"),
        ("D60E21_fullmap", "reviewer_targets", "Improve D60 canonical reviewer vs canonical"),
        ("D60E21_fullmap", "nonreviewer_complement", "Preserve D60 canonical complement vs canonical"),
        ("D21E41_fullmap", "all_points", "Improve D21 canonical all-points vs canonical"),
        ("D21E41_fullmap", "reviewer_targets", "Improve D21 canonical reviewer vs canonical"),
        ("D21E41_fullmap", "nonreviewer_complement", "Preserve D21 canonical complement vs canonical"),
        ("D60E41_holdout", "reviewer_targets", "Improve D60 holdout reviewer vs canonical"),
        ("D60E41_holdout", "nonreviewer_complement", "Preserve D60 holdout complement vs canonical"),
        ("D21E21_holdout", "all_points", "Improve D21 holdout all-points vs canonical"),
        ("D21E21_holdout", "reviewer_targets", "Improve D21 holdout reviewer vs canonical"),
        ("D21E21_holdout", "nonreviewer_complement", "Preserve D21 holdout complement vs canonical"),
    ):
        row = _row(validation, case, subset)
        if "Preserve" in label:
            passed = float(row["p95_abs_delta_cf"]) <= float(row["p95_abs_delta_base"]) + TOL_PUBLISH_NO_WORSEN
            gates.append(
                _gate(
                    label,
                    passed,
                    float(row["p95_abs_delta_cf"]) - float(row["p95_abs_delta_base"]),
                    TOL_PUBLISH_NO_WORSEN,
                    "<=",
                    f"{case}/{subset}: candidate p95 should stay within publish no-worsen tolerance.",
                )
            )
        else:
            passed = float(row["p95_abs_delta_cf"]) <= float(row["p95_abs_delta_base"]) - 1e-6
            gates.append(
                _gate(
                    label,
                    passed,
                    float(row["p95_abs_delta_base"]) - float(row["p95_abs_delta_cf"]),
                    1e-6,
                    ">=",
                    f"{case}/{subset}: candidate should beat canonical p95 by a non-zero margin.",
                )
            )

    for case, subset, label in (
        ("D21E41_fullmap", "all_points", "No new D21 canonical acceptance flips"),
        ("D21E21_holdout", "all_points", "No new D21 holdout acceptance flips"),
        ("D60E21_fullmap", "reviewer_targets", "No new D60 canonical reviewer flips"),
        ("D60E41_holdout", "reviewer_targets", "No new D60 holdout reviewer flips"),
    ):
        row = _row(validation, case, subset)
        gates.append(
            _gate(
                label,
                float(row["acceptance_flip_fraction_vs_base"]) == 0.0,
                float(row["acceptance_flip_fraction_vs_base"]),
                0.0,
                "==",
                f"{case}/{subset}: acceptance flips vs canonical base must stay zero.",
            )
        )

    for case, label in (
        ("D60E21_fullmap", "Improve D60 canonical drift strip vs parent"),
        ("D60E41_holdout", "Improve D60 holdout drift strip vs parent"),
        ("D80E41_monitor", "Improve D80 dense-grid drift strip vs parent"),
        ("D100E41_monitor", "Improve D100 dense-grid drift strip vs parent"),
    ):
        row = _row(validation, case, "drift_strip")
        improvement = float(row["p95_abs_delta_parent"]) - float(row["p95_abs_delta_cf"])
        gates.append(
            _gate(
                label,
                improvement >= 1e-6,
                improvement,
                1e-6,
                ">=",
                f"{case}/drift_strip: candidate should improve parent drift on the monitored strip.",
            )
        )

    for case, label in (
        ("D80E41_monitor", "No new D80 complement regression vs parent"),
        ("D100E41_monitor", "No new D100 complement regression vs parent"),
    ):
        row = _row(validation, case, "nonreviewer_complement")
        gates.append(
            _gate(
                label,
                float(row["worst_point_improvement_vs_parent"]) >= -1e-9,
                float(row["worst_point_improvement_vs_parent"]),
                -1e-9,
                ">=",
                f"{case}/nonreviewer_complement: worst-point parent-relative regression must remain negligible.",
            )
        )

    gates_df = pd.DataFrame(gates)
    passed = int(gates_df["passed"].sum())
    total = int(len(gates_df))
    decision = "GO_branch_switch_recommended" if passed == total else "HOLD_branch_switch"

    summary = pd.DataFrame(
        [
            {
                "candidate_mode": promotion["summary"]["candidate_mode"],
                "decision": decision,
                "go_flag": int(passed == total),
                "n_gates": total,
                "n_passed": passed,
                "n_failed": total - passed,
                "promotion_gate_go_flag": promo_go,
                "d60e21_reviewer_base": float(_row(validation, "D60E21_fullmap", "reviewer_targets")["p95_abs_delta_base"]),
                "d60e21_reviewer_candidate": float(_row(validation, "D60E21_fullmap", "reviewer_targets")["p95_abs_delta_cf"]),
                "d21e41_reviewer_base": float(_row(validation, "D21E41_fullmap", "reviewer_targets")["p95_abs_delta_base"]),
                "d21e41_reviewer_candidate": float(_row(validation, "D21E41_fullmap", "reviewer_targets")["p95_abs_delta_cf"]),
                "d21e41_complement_base": float(_row(validation, "D21E41_fullmap", "nonreviewer_complement")["p95_abs_delta_base"]),
                "d21e41_complement_candidate": float(_row(validation, "D21E41_fullmap", "nonreviewer_complement")["p95_abs_delta_cf"]),
                "d100e41_drift_parent": float(_row(validation, "D100E41_monitor", "drift_strip")["p95_abs_delta_parent"]),
                "d100e41_drift_candidate": float(_row(validation, "D100E41_monitor", "drift_strip")["p95_abs_delta_cf"]),
            }
        ]
    )

    summary_path = OUTDIR / f"{STEM}.csv"
    gates_path = OUTDIR / f"{STEM}_gates.csv"
    json_path = OUTDIR / f"{STEM}.json"
    summary.to_csv(summary_path, index=False)
    gates_df.to_csv(gates_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary.iloc[0].to_dict(), "gates": gates}, indent=2),
        encoding="utf-8",
    )
    for path in (summary_path, gates_path):
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    _plot(gates_df)

    print(summary.to_string(index=False))
    print(gates_df.to_string(index=False))


if __name__ == "__main__":
    main()
