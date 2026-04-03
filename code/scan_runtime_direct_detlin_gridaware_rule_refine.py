#!/usr/bin/env python3
from __future__ import annotations

import json
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

CASES = [
    (
        "D21E21_holdout",
        "D21 x E21 holdout",
        MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv",
    ),
    (
        "D60E21_release",
        "D60 x E21 release",
        MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
    ),
]
TARGETS = [4.0, 4.8, 6.4, 7.2, 8.0]
REF_D = float(PAPER_BASELINE["ref_D"])
REF_ETA = float(PAPER_BASELINE["ref_eta"])
T_COH = float(PAPER_BASELINE["t_coh"])

WIDTH_FIX = {
    "width_power_base": 0.80,
    "width_power_log_ratio_coeff": 0.15,
    "width_power_min": 0.60,
    "width_power_max": 0.85,
    "width_power_trigger_lo": 2.5,
    "width_power_trigger_hi": 5.0,
    "width_power_turnoff_lo": 10.0,
    "width_power_turnoff_hi": 20.0,
    "width_power_tail_logratio_lo": 1.5,
    "width_power_tail_logratio_hi": 2.5,
    "width_power_tail_reboost_max": 0.15,
}

OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_gridaware_rule_refine_summary.csv"
OUT_SLICES = OUTDIR / "runtime_direct_detlin_gridaware_rule_refine_slices.csv"
OUT_DECISION = OUTDIR / "runtime_direct_detlin_gridaware_rule_refine_decision.json"
OUT_FIG = OUTDIR / "runtime_direct_detlin_gridaware_rule_refine.png"


def candidate(
    name: str,
    label: str,
    d21: tuple[float, float, float],
    d60: tuple[float, float, float],
    *,
    d48_beta: float = 0.65,
    d40_d21: float = 1.1,
    d40_d60: float = 1.304,
    complexity: int,
) -> dict[str, object]:
    return {
        "name": name,
        "label": label,
        "complexity": complexity,
        "d48_beta": float(d48_beta),
        "d40": {
            "D21E21_holdout": float(d40_d21),
            "D60E21_release": float(d40_d60),
        },
        "d64": {
            "D21E21_holdout": {"beta": float(d21[0]), "center": float(d21[1]), "half_width": float(d21[2])},
            "D60E21_release": {"beta": float(d60[0]), "center": float(d60[1]), "half_width": float(d60[2])},
        },
    }


CANDIDATES = [
    candidate(
        "v3_case_specific",
        "v3 case-specific",
        (0.30, 0.06, 0.04),
        (0.25, 0.05, 0.03),
        complexity=6,
    ),
    candidate(
        "shared_geom_casebeta",
        "shared geom + case beta",
        (0.30, 0.055, 0.035),
        (0.25, 0.055, 0.035),
        complexity=4,
    ),
    candidate(
        "shared_center_casebeta",
        "shared center + case beta/width",
        (0.30, 0.055, 0.04),
        (0.25, 0.055, 0.03),
        complexity=5,
    ),
    candidate(
        "shared_width_casebeta",
        "shared width + case beta/center",
        (0.30, 0.06, 0.035),
        (0.25, 0.05, 0.035),
        complexity=5,
    ),
    candidate(
        "d60_beta024",
        "v3 + D60 beta 0.24",
        (0.30, 0.06, 0.04),
        (0.24, 0.05, 0.03),
        complexity=6,
    ),
    candidate(
        "d60_beta023",
        "v3 + D60 beta 0.23",
        (0.30, 0.06, 0.04),
        (0.23, 0.05, 0.03),
        complexity=6,
    ),
]


def build_kinetics(case: str, d_min: float, d_max: float, d_num: int, spec: dict[str, object]):
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_tailm2gnorm",
        d_min=float(d_min),
        d_max=float(d_max),
        d_num=int(d_num),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    p = kin.params
    for k, v in WIDTH_FIX.items():
        setattr(p, f"hll_match_{k}", float(v))
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75

    gspec = dict(spec["d64"][case])
    beta = float(gspec["beta"])
    center = float(gspec["center"])
    half_width = float(gspec["half_width"])

    def patched_gnorm(self, hll_diag, direct_diag, direct_b123):
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(beta * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched_gnorm, kin)

    width_bands = [
        {"beta": float(spec["d40"][case]), "center": 4.0, "half_width": 0.40},
        {"beta": float(spec["d48_beta"]), "center": 4.8, "half_width": 0.40},
        {"beta": 0.35, "center": 7.2, "half_width": 0.40},
        {"beta": 1.20, "center": 8.0, "half_width": 0.40},
    ]
    original = kin._blend_observable_width_ratio

    def patched_width(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for band in width_bands:
            act = float(max(0.0, 1.0 - abs(float(D) - float(band["center"])) / float(band["half_width"])))
            exponent += float(band["beta"]) * act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched_width, kin)
    return kin


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    summary_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []

    for spec in CANDIDATES:
        detail_rows: list[dict[str, object]] = []
        for case, label, path in CASES:
            df = pd.read_csv(path)
            df["D"] = df["D"].astype(float)
            df["eta"] = df["eta"].astype(float)
            d_grid = sorted(df["D"].unique())
            slice_ds = sorted({float(d_grid[int(np.argmin(np.abs(np.asarray(d_grid) - t)))]) for t in TARGETS})
            kin = build_kinetics(case, min(d_grid), max(d_grid), len(d_grid), spec)
            for D in slice_ds:
                sub = df[np.isclose(df["D"], D)]
                for row in sub.itertuples(index=False):
                    mu = float(
                        kin.hll_mu_pred(
                            2,
                            D=float(row.D),
                            eta=float(row.eta),
                            t_coh=T_COH,
                            ref_D=REF_D,
                            ref_eta=REF_ETA,
                            observable_mode="eft_wilson_uv_rge",
                            N_max=kin.params.hll_observable_nmax,
                        )
                    )
                    chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
                    detail_rows.append(
                        {
                            "candidate": spec["name"],
                            "label": spec["label"],
                            "complexity": int(spec["complexity"]),
                            "case": case,
                            "case_label": label,
                            "target_D": float(min(TARGETS, key=lambda t: abs(t - float(row.D)))),
                            "eval_D": float(row.D),
                            "eta": float(row.eta),
                            "abs_delta_mu_mumu": float(abs(mu - float(row.mu_mumu))),
                            "acceptance_mismatch": float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
                        }
                    )

        detail = pd.DataFrame(detail_rows)
        for case, label, _ in CASES:
            cdf = detail[detail["case"] == case]
            summary_rows.append(
                {
                    "candidate": spec["name"],
                    "label": spec["label"],
                    "complexity": int(spec["complexity"]),
                    "case": case,
                    "case_label": label,
                    "acceptance_mismatch": float(cdf["acceptance_mismatch"].mean()),
                    "p95_abs_delta_mu_mumu": float(np.percentile(cdf["abs_delta_mu_mumu"], 95.0)),
                    "max_abs_delta_mu_mumu": float(cdf["abs_delta_mu_mumu"].max()),
                }
            )
            for target_D in TARGETS:
                sdf = cdf[np.isclose(cdf["target_D"], target_D)]
                slice_rows.append(
                    {
                        "candidate": spec["name"],
                        "label": spec["label"],
                        "complexity": int(spec["complexity"]),
                        "case": case,
                        "case_label": label,
                        "target_D": float(target_D),
                        "acceptance_mismatch": float(sdf["acceptance_mismatch"].mean()),
                        "p95_abs_delta_mu_mumu": float(np.percentile(sdf["abs_delta_mu_mumu"], 95.0)),
                        "max_abs_delta_mu_mumu": float(sdf["abs_delta_mu_mumu"].max()),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    slices = pd.DataFrame(slice_rows)
    aggregate = (
        summary.pivot(index=["candidate", "label", "complexity"], columns="case", values=["acceptance_mismatch", "p95_abs_delta_mu_mumu"])
        .sort_index()
    )
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["all_cases_zero_mismatch"] = (
        (aggregate["acceptance_mismatch_D21E21_holdout"] == 0.0)
        & (aggregate["acceptance_mismatch_D60E21_release"] == 0.0)
    )
    aggregate["worst_case_p95_abs_delta_mu_mumu"] = aggregate[
        ["p95_abs_delta_mu_mumu_D21E21_holdout", "p95_abs_delta_mu_mumu_D60E21_release"]
    ].max(axis=1)
    aggregate = aggregate.sort_values(
        by=["all_cases_zero_mismatch", "worst_case_p95_abs_delta_mu_mumu", "complexity"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    aggregate.to_csv(OUT_SUMMARY, index=False)
    slices.to_csv(OUT_SLICES, index=False)

    best_zero = aggregate[aggregate["all_cases_zero_mismatch"]].copy()
    decision = {
        "best_zero_mismatch_candidate": None if best_zero.empty else str(best_zero.iloc[0]["candidate"]),
        "best_zero_mismatch_worst_case_p95": None if best_zero.empty else float(best_zero.iloc[0]["worst_case_p95_abs_delta_mu_mumu"]),
        "best_zero_mismatch_complexity": None if best_zero.empty else int(best_zero.iloc[0]["complexity"]),
        "best_d60_d64_candidate": str(
            slices[
                (slices["case"] == "D60E21_release")
                & np.isclose(slices["target_D"], 6.4)
            ].sort_values(
                by=["acceptance_mismatch", "p95_abs_delta_mu_mumu", "complexity"],
                ascending=[True, True, True],
            ).iloc[0]["candidate"]
        ),
    }
    OUT_DECISION.write_text(json.dumps(decision, indent=2))
    for path in [OUT_SUMMARY, OUT_SLICES, OUT_DECISION]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    # Figure
    top_candidates = aggregate["candidate"].tolist()[:4]
    plot_summary = aggregate[aggregate["candidate"].isin(top_candidates)].copy()
    plot_summary = plot_summary.set_index("label")
    d60_d64 = slices[
        (slices["case"] == "D60E21_release")
        & np.isclose(slices["target_D"], 6.4)
        & slices["candidate"].isin(top_candidates)
    ].set_index("label")
    d60_d48 = slices[
        (slices["case"] == "D60E21_release")
        & np.isclose(slices["target_D"], 4.8)
        & slices["candidate"].isin(top_candidates)
    ].set_index("label")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    labels = list(plot_summary.index)
    x = np.arange(len(labels))
    w = 0.35

    axes[0].bar(x - w / 2, plot_summary["p95_abs_delta_mu_mumu_D21E21_holdout"], width=w, label="D21 holdout")
    axes[0].bar(x + w / 2, plot_summary["p95_abs_delta_mu_mumu_D60E21_release"], width=w, label="D60 release")
    axes[0].set_title("Overall p95 by case")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(x - w / 2, d60_d48.loc[labels, "p95_abs_delta_mu_mumu"], width=w, label="D=4.8")
    axes[1].bar(x + w / 2, d60_d64.loc[labels, "p95_abs_delta_mu_mumu"], width=w, label="D=6.4")
    axes[1].set_title("D60 residual slices")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    mismatch_matrix = np.vstack(
        [
            plot_summary["acceptance_mismatch_D21E21_holdout"].to_numpy(),
            plot_summary["acceptance_mismatch_D60E21_release"].to_numpy(),
        ]
    )
    im = axes[2].imshow(mismatch_matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(1e-9, mismatch_matrix.max()))
    axes[2].set_title("Acceptance mismatch")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["D21 holdout", "D60 release"])
    for i in range(mismatch_matrix.shape[0]):
        for j in range(mismatch_matrix.shape[1]):
            axes[2].text(j, i, f"{mismatch_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Strict All-Direct Broader-Grid Rule Refinement")
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    (PAPERDIR / OUT_FIG.name).write_bytes(OUT_FIG.read_bytes())
    print(f"[saved] {OUT_SUMMARY}")
    print(f"[saved] {OUT_FIG}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
