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

SCHUR_MAP = ROOT / "output" / "hll_uv_matching" / "hll_uv_action_loop_eymh_logdet_schur_audit_map_D21E21_fix.csv"
BLOCKDET_MAP = ROOT / "output" / "hll_uv_matching" / "hll_uv_action_loop_eymh_parent_blockdet_audit_map_D21E21_fix.csv"

D60_REF_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
D60_CUR_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv"
D21_REF_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv"
D21_CUR_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E41.csv"

D60_POINTAMP_BAND = (6.169492525423729, 6.4406779661016955)
D60_WIDTH_BAND = (4.813559322033898, 5.627118644067797, 6.711864406779661)
D21_GUARD = 6.4

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


def _load_residual_profile(ref_map: Path, cur_map: Path, prefix: str) -> pd.DataFrame:
    ref = pd.read_csv(ref_map)
    cur = pd.read_csv(cur_map)
    merged = (
        ref.merge(cur, on=["D", "eta"], suffixes=("_ref", "_cur"))
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    merged["abs_delta_mu"] = (merged["mu_mumu_cur"] - merged["mu_mumu_ref"]).abs()

    rows = []
    for d, g in merged.groupby("D", sort=True):
        vals = g["abs_delta_mu"].to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        rel_std = float(np.std(vals, ddof=0) / max(mean_val, 1e-30))
        rows.append(
            {
                "D": float(d),
                f"{prefix}_p95_abs_delta_mu": float(np.percentile(vals, 95.0)),
                f"{prefix}_max_abs_delta_mu": float(np.max(vals)),
                f"{prefix}_mean_abs_delta_mu": mean_val,
                f"{prefix}_rel_std_abs_delta_mu": rel_std,
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _load_feature_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    schur = pd.read_csv(SCHUR_MAP)
    blockdet = pd.read_csv(BLOCKDET_MAP)
    merged = schur.merge(blockdet, on=["D", "eta"], how="inner", suffixes=("", "_block"))

    profile_rows = []
    eta_rows = []
    for d, g in merged.groupby("D", sort=True):
        row = {"D": float(d)}
        for feature in FEATURE_LABELS:
            vals = g[feature].to_numpy(dtype=float)
            row[feature] = float(np.mean(vals))
            eta_rows.append(
                {
                    "D": float(d),
                    "feature": feature,
                    "mean_value": float(np.mean(vals)),
                    "rel_std_eta": float(np.std(vals, ddof=0) / max(abs(float(np.mean(vals))), 1e-30)),
                }
            )
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows).sort_values("D").reset_index(drop=True)
    eta_stats = pd.DataFrame(eta_rows)
    return profile, eta_stats


def _interp_scalar(D: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(D, x, y)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_feature(
    feature: str,
    feature_profile: pd.DataFrame,
    eta_stats: pd.DataFrame,
    d60_profile: pd.DataFrame,
    d21_profile: pd.DataFrame,
) -> dict[str, float | str]:
    x = feature_profile["D"].to_numpy(dtype=float)
    y = feature_profile[feature].to_numpy(dtype=float)

    d60_x = d60_profile["D"].to_numpy(dtype=float)
    d60_y = d60_profile["d60_p95_abs_delta_mu"].to_numpy(dtype=float)
    d21_x = d21_profile["D"].to_numpy(dtype=float)
    d21_y = d21_profile["d21_p95_abs_delta_mu"].to_numpy(dtype=float)

    d60_feat = _interp_scalar(d60_x, x, y)
    d21_feat = _interp_scalar(d21_x, x, y)

    def iv(d: float) -> float:
        return float(_interp_scalar(np.asarray([d], dtype=float), x, y)[0])

    v_4p814 = iv(4.813559322033898)
    v_5p627 = iv(5.627118644067797)
    v_6p169 = iv(6.169492525423729)
    v_6p4 = iv(6.4)
    v_6p441 = iv(6.4406779661016955)
    v_6p712 = iv(6.711864406779661)

    hotspot_vals = np.asarray([v_6p169, v_6p441], dtype=float)
    width_guard_vals = np.asarray([v_4p814, v_5p627, v_6p712], dtype=float)
    d21_guard_vals = np.asarray([v_6p4], dtype=float)
    all_guard_vals = np.concatenate([width_guard_vals, d21_guard_vals])

    high_trigger_separation = float(np.min(hotspot_vals) / max(float(np.max(all_guard_vals)), 1e-30))
    low_trigger_separation = float(np.min(all_guard_vals) / max(float(np.max(hotspot_vals)), 1e-30))

    eta_sub = eta_stats.loc[eta_stats["feature"] == feature, "rel_std_eta"].to_numpy(dtype=float)
    corr_d60 = _corr(d60_feat, d60_y)
    corr_d21 = _corr(d21_feat, d21_y)

    return {
        "feature": feature,
        "label": FEATURE_LABELS[feature],
        "corr_d60_p95_abs_delta_mu": corr_d60,
        "corr_d21_p95_abs_delta_mu": corr_d21,
        "abs_corr_sum": float(np.nan_to_num(abs(corr_d60)) + np.nan_to_num(abs(corr_d21))),
        "eta_flatness_p90_rel_std": float(np.percentile(eta_sub, 90.0)),
        "eta_flatness_max_rel_std": float(np.max(eta_sub)),
        "value_D4p814": v_4p814,
        "value_D5p627": v_5p627,
        "value_D6p169": v_6p169,
        "value_D6p4": v_6p4,
        "value_D6p441": v_6p441,
        "value_D6p712": v_6p712,
        "d60_pointamp_band_mean": float(np.mean(hotspot_vals)),
        "d60_width_band_mean": float(np.mean(width_guard_vals)),
        "d21_guard_value": v_6p4,
        "high_trigger_separation": high_trigger_separation,
        "low_trigger_separation": low_trigger_separation,
        "best_monotone_separation": float(max(high_trigger_separation, low_trigger_separation)),
        "separable_monotone": float(max(high_trigger_separation, low_trigger_separation) > 1.0),
    }


def _build_detail(
    feature_profile: pd.DataFrame,
    d60_profile: pd.DataFrame,
    d21_profile: pd.DataFrame,
) -> pd.DataFrame:
    d_union = np.unique(
        np.concatenate(
            [
                feature_profile["D"].to_numpy(dtype=float),
                d60_profile["D"].to_numpy(dtype=float),
                d21_profile["D"].to_numpy(dtype=float),
            ]
        )
    )
    d_union.sort()

    detail = pd.DataFrame({"D": d_union})
    detail["d60_p95_abs_delta_mu"] = _interp_scalar(
        detail["D"].to_numpy(dtype=float),
        d60_profile["D"].to_numpy(dtype=float),
        d60_profile["d60_p95_abs_delta_mu"].to_numpy(dtype=float),
    )
    detail["d21_p95_abs_delta_mu"] = _interp_scalar(
        detail["D"].to_numpy(dtype=float),
        d21_profile["D"].to_numpy(dtype=float),
        d21_profile["d21_p95_abs_delta_mu"].to_numpy(dtype=float),
    )
    detail["is_d60_pointamp_hotspot"] = detail["D"].isin(D60_POINTAMP_BAND).astype(int)
    detail["is_d60_width_hotspot"] = detail["D"].isin(D60_WIDTH_BAND).astype(int)
    detail["is_d21_guard"] = np.isclose(detail["D"], D21_GUARD).astype(int)

    x = feature_profile["D"].to_numpy(dtype=float)
    for feature in FEATURE_LABELS:
        detail[feature] = _interp_scalar(detail["D"].to_numpy(dtype=float), x, feature_profile[feature].to_numpy(dtype=float))
    return detail


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)

    axes[0].scatter(
        summary["high_trigger_separation"],
        summary["low_trigger_separation"],
        c=summary["eta_flatness_p90_rel_std"],
        cmap="viridis",
        s=48,
    )
    axes[0].axvline(1.0, color="0.6", ls="--", lw=1)
    axes[0].axhline(1.0, color="0.6", ls="--", lw=1)
    axes[0].set_title("Monotone separation test")
    axes[0].set_xlabel("high-trigger separation")
    axes[0].set_ylabel("low-trigger separation")
    axes[0].grid(True, alpha=0.25)

    top = summary.head(min(6, len(summary)))
    for _, row in top.iterrows():
        axes[0].annotate(
            row["label"],
            (row["high_trigger_separation"], row["low_trigger_separation"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    sc = axes[1].scatter(
        summary["corr_d60_p95_abs_delta_mu"],
        summary["corr_d21_p95_abs_delta_mu"],
        c=summary["best_monotone_separation"],
        cmap="plasma",
        s=48,
    )
    axes[1].axvline(0.0, color="0.6", ls="--", lw=1)
    axes[1].axhline(0.0, color="0.6", ls="--", lw=1)
    axes[1].set_title("Correlation vs reviewer residuals")
    axes[1].set_xlabel("corr with D60 p95 |Δμ|")
    axes[1].set_ylabel("corr with D21 p95 |Δμ|")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[1], label="best monotone separation")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d60_profile = _load_residual_profile(D60_REF_MAP, D60_CUR_MAP, "d60")
    d21_profile = _load_residual_profile(D21_REF_MAP, D21_CUR_MAP, "d21")
    feature_profile, eta_stats = _load_feature_profiles()

    summary_rows = [
        _summarize_feature(feature, feature_profile, eta_stats, d60_profile, d21_profile)
        for feature in FEATURE_LABELS
    ]
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["best_monotone_separation", "abs_corr_sum", "eta_flatness_p90_rel_std"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    detail = _build_detail(feature_profile, d60_profile, d21_profile)
    slices = summary[
        [
            "feature",
            "label",
            "value_D4p814",
            "value_D5p627",
            "value_D6p169",
            "value_D6p4",
            "value_D6p441",
            "value_D6p712",
            "high_trigger_separation",
            "low_trigger_separation",
            "best_monotone_separation",
        ]
    ].copy()

    stem = "model_chain_schur_guard_correlation_audit"
    detail_path = OUTDIR / f"{stem}_detail.csv"
    summary_path = OUTDIR / f"{stem}_summary.csv"
    slices_path = OUTDIR / f"{stem}_slices.csv"
    png_path = OUTDIR / f"{stem}.png"
    meta_path = OUTDIR / f"{stem}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "schur_map": str(SCHUR_MAP),
                "parent_blockdet_map": str(BLOCKDET_MAP),
                "d60_ref_map": str(D60_REF_MAP),
                "d60_cur_map": str(D60_CUR_MAP),
                "d21_ref_map": str(D21_REF_MAP),
                "d21_cur_map": str(D21_CUR_MAP),
                "d60_pointamp_band": [float(x) for x in D60_POINTAMP_BAND],
                "d60_width_band": [float(x) for x in D60_WIDTH_BAND],
                "d21_guard": float(D21_GUARD),
                "features": FEATURE_LABELS,
            },
            indent=2,
        )
    )
    _mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
