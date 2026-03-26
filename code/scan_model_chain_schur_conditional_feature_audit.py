#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

D60_SUMMARY = (
    ROOT
    / "output"
    / "kinetic_action_chain"
    / "model_chain_fullwidthrefamp_pointamp2_widthboost_d60_reviewerref_hotspot_audit_summary.csv"
)
D21_SUMMARY = (
    ROOT
    / "output"
    / "kinetic_action_chain"
    / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit_summary.csv"
)
SCHUR_MAP = (
    ROOT
    / "output"
    / "hll_uv_matching"
    / "hll_uv_action_loop_eymh_logdet_schur_audit_map_D21E21_fix.csv"
)
BLOCKDET_MAP = (
    ROOT
    / "output"
    / "hll_uv_matching"
    / "hll_uv_action_loop_eymh_parent_blockdet_audit_map_D21E21_fix.csv"
)

POSITIVE_D60 = (6.169492525423729, 6.4406779661016955)
GUARD_D60 = (4.813559322033898, 5.627118644067797, 6.711864406779661)
GUARD_D21 = (4.0, 6.4, 8.0)

FEATURE_LABELS = {
    "projected_logdet_part_kernel": "logdet_part_kernel",
    "projected_logdet_part_action": "logdet_part_action",
    "projected_schur_gain": "schur_gain",
    "projected_schur_action": "schur_action",
    "projected_logdet_schur_response_action": "logdet_schur_action",
    "projected_logdet_schur_response_weight": "logdet_schur_weight",
    "projected_logdet_part_share": "logdet_part_share",
    "projected_schur_share": "schur_share",
    "eymh_parented_prefactor_diag": "eymh_prefactor_diag",
    "parent_blockdet_det_ratio": "blockdet_det_ratio",
    "parent_blockdet_schur_norm": "blockdet_schur_norm",
    "parent_blockdet_action": "blockdet_action",
}

BASE_FAMILIES = {
    "pointamp_leverage": "pointamp leverage",
    "family_score": "family score",
}


def _interp_scalar(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(d, x, y)


def _load_reviewer_profiles() -> pd.DataFrame:
    d60 = pd.read_csv(D60_SUMMARY).copy()
    d60["grid"] = "D60"
    d21 = pd.read_csv(D21_SUMMARY).copy()
    d21["grid"] = "D21"
    detail = pd.concat([d60, d21], ignore_index=True, sort=False)
    detail["D"] = detail["D"].astype(float)
    detail = detail.sort_values(["grid", "D"]).reset_index(drop=True)

    roles = []
    for row in detail.itertuples():
        if row.grid == "D60" and any(np.isclose(row.D, x) for x in POSITIVE_D60):
            roles.append("positive")
        elif row.grid == "D60" and any(np.isclose(row.D, x) for x in GUARD_D60):
            roles.append("guard")
        elif row.grid == "D21" and any(np.isclose(row.D, x) for x in GUARD_D21):
            roles.append("guard")
        else:
            roles.append("other")
    detail["eval_role"] = roles
    detail["pointamp_leverage"] = np.clip(
        detail["p95_abs_delta_mu"].to_numpy(dtype=float)
        - detail["p95_abs_delta_mu_pointamp_anchor"].to_numpy(dtype=float),
        0.0,
        None,
    )
    detail["partial_excess"] = np.clip(
        detail["p95_abs_delta_log_partial"].to_numpy(dtype=float)
        - detail["p95_abs_delta_log_width"].to_numpy(dtype=float),
        0.0,
        None,
    )
    # This intentionally rewards D-slices that are both point-amp sensitive and
    # partial-dominated, matching the current D60 reviewer blocker structure.
    detail["family_score"] = detail["pointamp_leverage"] * np.maximum(
        detail["partial_excess"], 1e-9
    )
    return detail


def _load_feature_profiles() -> pd.DataFrame:
    schur = pd.read_csv(SCHUR_MAP)
    blockdet = pd.read_csv(BLOCKDET_MAP)
    merged = schur.merge(blockdet, on=["D", "eta"], how="inner", suffixes=("", "_block"))

    rows = []
    for d, grp in merged.groupby("D", sort=True):
        row = {"D": float(d)}
        for feature in FEATURE_LABELS:
            row[feature] = float(np.mean(grp[feature].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _attach_features(detail: pd.DataFrame, feature_profiles: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    x = feature_profiles["D"].to_numpy(dtype=float)
    eval_mask = out["eval_role"].isin(["positive", "guard"]).to_numpy()
    eval_d = out.loc[eval_mask, "D"].to_numpy(dtype=float)

    for feature in FEATURE_LABELS:
        vals = _interp_scalar(out["D"].to_numpy(dtype=float), x, feature_profiles[feature].to_numpy(dtype=float))
        out[feature] = vals

        eval_vals = _interp_scalar(eval_d, x, feature_profiles[feature].to_numpy(dtype=float))
        vmin = float(np.min(eval_vals))
        vmax = float(np.max(eval_vals))
        if abs(vmax - vmin) < 1e-30:
            norm_vals = np.full_like(vals, 0.5, dtype=float)
        else:
            norm_vals = (vals - vmin) / (vmax - vmin)
        out[f"{feature}__norm"] = norm_vals
    return out


def _build_family_rows(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = detail.loc[detail["eval_role"].isin(["positive", "guard"])].copy().reset_index(drop=True)
    detail_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    base_sep_cache: dict[str, float] = {}

    for base_key, base_label in BASE_FAMILIES.items():
        family_specs = [
            {
                "family": base_key,
                "base_family": base_key,
                "base_label": base_label,
                "feature": "none",
                "feature_label": "none",
                "polarity": "base",
                "score": eval_df[base_key].to_numpy(dtype=float),
            }
        ]
        for feature, feature_label in FEATURE_LABELS.items():
            norm = eval_df[f"{feature}__norm"].to_numpy(dtype=float)
            base = eval_df[base_key].to_numpy(dtype=float)
            family_specs.append(
                {
                    "family": f"{base_key}__hi__{feature}",
                    "base_family": base_key,
                    "base_label": base_label,
                    "feature": feature,
                    "feature_label": feature_label,
                    "polarity": "high_trigger",
                    "score": base * norm,
                }
            )
            family_specs.append(
                {
                    "family": f"{base_key}__lo__{feature}",
                    "base_family": base_key,
                    "base_label": base_label,
                    "feature": feature,
                    "feature_label": feature_label,
                    "polarity": "low_trigger",
                    "score": base * (1.0 - norm),
                }
            )

        for spec in family_specs:
            scores = np.asarray(spec["score"], dtype=float)
            pos_mask = eval_df["eval_role"].eq("positive").to_numpy()
            guard_mask = eval_df["eval_role"].eq("guard").to_numpy()
            pos_scores = scores[pos_mask]
            guard_scores = scores[guard_mask]
            pos_min = float(np.min(pos_scores))
            pos_mean = float(np.mean(pos_scores))
            guard_max = float(np.max(guard_scores))
            guard_mean = float(np.mean(guard_scores))
            separation = float(pos_min / max(guard_max, 1e-30))
            margin = float(pos_min - guard_max)

            if spec["polarity"] == "base":
                base_sep_cache[spec["base_family"]] = separation
                improvement = 1.0
            else:
                improvement = float(separation / max(base_sep_cache[spec["base_family"]], 1e-30))

            row = {
                "family": spec["family"],
                "base_family": spec["base_family"],
                "base_label": spec["base_label"],
                "feature": spec["feature"],
                "feature_label": spec["feature_label"],
                "polarity": spec["polarity"],
                "positive_min_score": pos_min,
                "positive_mean_score": pos_mean,
                "guard_max_score": guard_max,
                "guard_mean_score": guard_mean,
                "separation": separation,
                "margin": margin,
                "improvement_vs_base": improvement,
                "separable": float(separation > 1.0),
            }

            key_cols = {
                "D60_D6p169_score": ("D60", 6.169492525423729),
                "D60_D6p441_score": ("D60", 6.4406779661016955),
                "D60_guard_D4p814_score": ("D60", 4.813559322033898),
                "D60_guard_D5p627_score": ("D60", 5.627118644067797),
                "D60_guard_D6p712_score": ("D60", 6.711864406779661),
                "D21_guard_D4p0_score": ("D21", 4.0),
                "D21_guard_D6p4_score": ("D21", 6.4),
                "D21_guard_D8p0_score": ("D21", 8.0),
            }
            for col, (grid, d_val) in key_cols.items():
                mask = eval_df["grid"].eq(grid) & np.isclose(eval_df["D"], d_val)
                row[col] = float(scores[mask.to_numpy()][0])

            summary_rows.append(row)

            for idx, drow in eval_df.iterrows():
                detail_rows.append(
                    {
                        "family": spec["family"],
                        "base_family": spec["base_family"],
                        "feature": spec["feature"],
                        "polarity": spec["polarity"],
                        "grid": drow["grid"],
                        "D": float(drow["D"]),
                        "eval_role": drow["eval_role"],
                        "score": float(scores[idx]),
                        "pointamp_leverage": float(drow["pointamp_leverage"]),
                        "partial_excess": float(drow["partial_excess"]),
                        "family_score": float(drow["family_score"]),
                        "p95_abs_delta_mu": float(drow["p95_abs_delta_mu"]),
                        "p95_abs_delta_log_partial": float(drow["p95_abs_delta_log_partial"]),
                        "p95_abs_delta_log_width": float(drow["p95_abs_delta_log_width"]),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["separation", "margin", "positive_min_score"], ascending=[False, False, False]
    ).reset_index(drop=True)
    detail_long = pd.DataFrame(detail_rows)
    return summary, detail_long


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    top = summary.head(min(12, len(summary))).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)

    colors = {"base": "#355C7D", "high_trigger": "#2A9D8F", "low_trigger": "#E76F51"}
    for polarity, grp in summary.groupby("polarity", sort=False):
        axes[0].scatter(
            grp["guard_max_score"],
            grp["positive_min_score"],
            s=36,
            alpha=0.8,
            label=polarity,
            color=colors.get(polarity, "#666666"),
        )
    axes[0].plot([0, max(summary["guard_max_score"].max(), 1e-9)], [0, max(summary["guard_max_score"].max(), 1e-9)], ls="--", lw=1, color="0.5")
    axes[0].set_xlabel("guard max score")
    axes[0].set_ylabel("positive min score")
    axes[0].set_title("Conditional family separation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(top))
    axes[1].bar(x - 0.18, top["positive_min_score"], width=0.36, label="positive min")
    axes[1].bar(x + 0.18, top["guard_max_score"], width=0.36, label="guard max")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [
            ("base\n" + r.base_family)
            if r.feature == "none"
            else f"{r.base_family}\n{r.polarity[:2]}:{r.feature_label}"
            for r in top.itertuples()
        ],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    axes[1].set_ylabel("score")
    axes[1].set_title("Top conditional families")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    reviewer = _load_reviewer_profiles()
    feature_profiles = _load_feature_profiles()
    reviewer = _attach_features(reviewer, feature_profiles)

    summary, detail_long = _build_family_rows(reviewer)
    slices = summary.head(24).copy()

    stem = "model_chain_schur_conditional_feature_audit"
    detail_path = OUTDIR / f"{stem}_detail.csv"
    summary_path = OUTDIR / f"{stem}_summary.csv"
    slices_path = OUTDIR / f"{stem}_slices.csv"
    png_path = OUTDIR / f"{stem}.png"
    meta_path = OUTDIR / f"{stem}_run_meta.json"

    detail_long.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)

    meta = {
        "positive_d60": list(POSITIVE_D60),
        "guard_d60": list(GUARD_D60),
        "guard_d21": list(GUARD_D21),
        "features": list(FEATURE_LABELS.keys()),
        "base_families": list(BASE_FAMILIES.keys()),
        "source_files": {
            "d60_summary": str(D60_SUMMARY),
            "d21_summary": str(D21_SUMMARY),
            "schur_map": str(SCHUR_MAP),
            "blockdet_map": str(BLOCKDET_MAP),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
