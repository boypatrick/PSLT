#!/usr/bin/env python3
"""
Component-level audit for runtime-direct visibility drift.

This audit is tuned for the strict all-direct visibility bottleneck. It uses:

  - full/release map comparisons on the reviewer grids
  - D-dependent direct visibility components (`B_N`, `y_raw`, `m2`)
  - targeted worst/anchor probes for the eta-dependent width-ratio layer

The point is to determine whether the residual drift is coming from raw direct
operator inputs, mediator scales, or from the runtime-direct visibility
normalization layer itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from publish_full_direct_map import _extract_b_metrics  # noqa: E402
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SIGDIR = ROOT / "output" / "hll_signal_strength"
INSTABILITY_MAP = OUTDIR / "runtime_direct_visibility_instability_map.csv"

SURFACES = {
    "D21E41": {
        "d_num": 21,
        "eta_num": 41,
        "full_map": SIGDIR / "hll_signal_strength_map_full_direct_map_full_release_D21E41.csv",
        "runtime_map": SIGDIR / "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_D21E41.csv",
        "release_map": SIGDIR / "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_tuned_profile_anchored_D21E41.csv",
    },
    "D60E21": {
        "d_num": 60,
        "eta_num": 21,
        "full_map": SIGDIR / "hll_signal_strength_map_full_direct_map_full_release_D60E21.csv",
        "runtime_map": SIGDIR / "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_D60E21.csv",
        "release_map": SIGDIR / "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_tuned_profile_anchored_D60E21.csv",
    },
}


def _safe_log_ratio(num: float, den: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(num), floor) / max(float(den), floor)))


def _snap_ref_d_to_grid(ref_d: float, d_values: np.ndarray) -> float:
    vals = np.asarray(d_values, dtype=float)
    idx = int(np.argmin(np.abs(vals - float(ref_d))))
    return float(vals[idx])


def _baseline_common() -> Dict[str, float | int | bool | str]:
    return {
        "observable_mode": str(PAPER_BASELINE["hll_observable_mode"]),
        "d_min": float(PAPER_BASELINE["D_min"]),
        "d_max": float(PAPER_BASELINE["D_max"]),
        "uv_blend": float(PAPER_BASELINE["hll_uv_blend"]),
        "uv_m2_power": float(PAPER_BASELINE["hll_uv_m2_power"]),
        "uv_match_kappa_diag": float(PAPER_BASELINE["hll_uv_match_kappa_diag"]),
        "uv_match_kappa_offdiag": float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"]),
        "uv_match_mode": str(PAPER_BASELINE["hll_uv_match_mode"]),
        "uv_match_input_diag_scale": float(PAPER_BASELINE["hll_uv_match_input_diag_scale"]),
        "uv_match_input_offdiag_scale": float(PAPER_BASELINE["hll_uv_match_input_offdiag_scale"]),
        "uv_rge_mu_low": float(PAPER_BASELINE["hll_uv_rge_mu_low"]),
        "uv_rge_gamma_diag": float(PAPER_BASELINE["hll_uv_rge_gamma_diag"]),
        "uv_rge_gamma_offdiag": float(PAPER_BASELINE["hll_uv_rge_gamma_offdiag"]),
        "uv_rge_log_clip": float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
        "runtime_direct_force": False,
        "runtime_direct_no_cache": False,
        "runtime_direct_chi_rho_max": float(PAPER_BASELINE.get("runtime_direct_chi_rho_max", 3.0)),
        "runtime_direct_chi_z_margin": float(PAPER_BASELINE.get("runtime_direct_chi_z_margin", 6.0)),
        "runtime_direct_chi_n_mu": int(PAPER_BASELINE.get("runtime_direct_chi_n_mu", 120)),
        "runtime_direct_chi_tol": float(PAPER_BASELINE.get("runtime_direct_chi_tol", 1e-8)),
        "runtime_direct_chi_maxiter": int(PAPER_BASELINE.get("runtime_direct_chi_maxiter", 30000)),
        "runtime_direct_chi_sigma": float(PAPER_BASELINE.get("runtime_direct_chi_sigma", 2.5)),
        "runtime_direct_superrad_zmax": float(PAPER_BASELINE.get("runtime_direct_superrad_zmax", 80.0)),
        "runtime_direct_superrad_ref_d": float(PAPER_BASELINE.get("runtime_direct_superrad_ref_d", 12.0)),
        "runtime_direct_superrad_n_ref": int(PAPER_BASELINE.get("runtime_direct_superrad_n_ref", 2)),
    }


def _build_kinetics(surface_cfg: Dict[str, object]) -> Dict[str, object]:
    common = _baseline_common()
    common["d_num"] = int(surface_cfg["d_num"])
    return {
        "full_direct": make_baseline_kinetics(chain_mode="full_direct", **common),
        "cell_direct_runtime_release": make_baseline_kinetics(chain_mode="cell_direct_runtime_release", **common),
        "cell_direct_runtime_release_tuned": make_baseline_kinetics(
            chain_mode="cell_direct_runtime_release_tuned",
            **common,
        ),
    }


def _read_map(path: Path, prefix: str) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={
            "mu_ee": f"mu_ee_{prefix}",
            "mu_mumu": f"mu_mumu_{prefix}",
            "mu_tautau": f"mu_tautau_{prefix}",
            "chi2_mumu": f"chi2_mumu_{prefix}",
        }
    )


def _collect_d_component_cache(
    surface: str,
    kin_full,
    kin_cmp,
    scenario: str,
    d_values: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for d_val in np.asarray(d_values, dtype=float):
        b_full = np.array([kin_full.B_N(1, d_val), kin_full.B_N(2, d_val), kin_full.B_N(3, d_val)], dtype=float)
        b_cmp = np.array([kin_cmp.B_N(1, d_val), kin_cmp.B_N(2, d_val), kin_cmp.B_N(3, d_val)], dtype=float)
        y_full = np.asarray(kin_full._hll_yraw_vector(float(d_val)), dtype=float)
        y_cmp = np.asarray(kin_cmp._hll_yraw_vector(float(d_val)), dtype=float)
        m2_full = np.asarray(kin_full._hll_m2_vector(float(d_val)), dtype=float)
        m2_cmp = np.asarray(kin_cmp._hll_m2_vector(float(d_val)), dtype=float)
        alpha = 0.0
        if scenario == "cell_direct_runtime_release_tuned":
            alpha = float(kin_cmp._runtime_direct_b_profile_blend_at(float(d_val)))

        log_b = np.array([_safe_log_ratio(b_cmp[i], b_full[i]) for i in range(3)], dtype=float)
        log_y = np.array([_safe_log_ratio(y_cmp[i], y_full[i]) for i in range(3)], dtype=float)
        log_m2 = np.array([_safe_log_ratio(m2_cmp[i], m2_full[i]) for i in range(3)], dtype=float)
        rows.append(
            {
                "surface": surface,
                "scenario": scenario,
                "D": float(d_val),
                "alpha_blend": alpha,
                "B1_full": float(b_full[0]),
                "B2_full": float(b_full[1]),
                "B3_full": float(b_full[2]),
                "B1_cmp": float(b_cmp[0]),
                "B2_cmp": float(b_cmp[1]),
                "B3_cmp": float(b_cmp[2]),
                "yraw1_full": float(y_full[0]),
                "yraw2_full": float(y_full[1]),
                "yraw3_full": float(y_full[2]),
                "yraw1_cmp": float(y_cmp[0]),
                "yraw2_cmp": float(y_cmp[1]),
                "yraw3_cmp": float(y_cmp[2]),
                "m2_1_full": float(m2_full[0]),
                "m2_2_full": float(m2_full[1]),
                "m2_3_full": float(m2_full[2]),
                "m2_1_cmp": float(m2_cmp[0]),
                "m2_2_cmp": float(m2_cmp[1]),
                "m2_3_cmp": float(m2_cmp[2]),
                "log_B1_ratio": float(log_b[0]),
                "log_B2_ratio": float(log_b[1]),
                "log_B3_ratio": float(log_b[2]),
                "log_yraw1_ratio": float(log_y[0]),
                "log_yraw2_ratio": float(log_y[1]),
                "log_yraw3_ratio": float(log_y[2]),
                "log_m2_1_ratio": float(log_m2[0]),
                "log_m2_2_ratio": float(log_m2[1]),
                "log_m2_3_ratio": float(log_m2[2]),
                "b_visibility_norm_score": float(np.max(np.abs(log_b[:2]))),
                "yraw_operator_score": float(np.max(np.abs(log_y))),
                "m2_spectrum_score": float(np.max(np.abs(log_m2))),
            }
        )
    return pd.DataFrame(rows)


def _collect_probe_rows(
    surface: str,
    scenario: str,
    full_map: pd.DataFrame,
    cmp_map: pd.DataFrame,
    kin_full,
    kin_cmp,
    ref_d: float,
    ref_eta: float,
) -> List[Dict[str, object]]:
    merged = full_map.merge(cmp_map, on=["D", "eta"], how="inner", validate="one_to_one").sort_values(["D", "eta"])
    idx_worst = int(np.argmax(np.abs(merged["mu_mumu_cmp"].to_numpy(dtype=float) - merged["mu_mumu_full"].to_numpy(dtype=float))))
    dist = (merged["D"].astype(float) - ref_d) ** 2 + (merged["eta"].astype(float) - ref_eta) ** 2
    idx_anchor = int(dist.idxmin())
    out: List[Dict[str, object]] = []
    for point_kind, idx in (("worst_abs_delta", idx_worst), ("reference_anchor", idx_anchor)):
        row = merged.iloc[int(idx)]
        d_val = float(row["D"])
        eta_val = float(row["eta"])
        full_metrics = _extract_b_metrics(kin_full, d_val, eta_val, ref_d=ref_d, ref_eta=ref_eta)
        cmp_metrics = _extract_b_metrics(kin_cmp, d_val, eta_val, ref_d=ref_d, ref_eta=ref_eta)
        out.append(
            {
                "surface": surface,
                "scenario": scenario,
                "point_kind": point_kind,
                "D": d_val,
                "eta": eta_val,
                "abs_delta_mu_mumu_map": float(abs(row["mu_mumu_cmp"] - row["mu_mumu_full"])),
                "log_width_ratio": _safe_log_ratio(cmp_metrics["width_ratio"], full_metrics["width_ratio"]),
                "width_ratio_full": float(full_metrics["width_ratio"]),
                "width_ratio_cmp": float(cmp_metrics["width_ratio"]),
                "mu_mumu_full_model": float(full_metrics["mu_mumu_model"]),
                "mu_mumu_cmp_model": float(cmp_metrics["mu_mumu_model"]),
                "model_map_delta_residual": float((cmp_metrics["mu_mumu_model"] - full_metrics["mu_mumu_model"]) - (float(row["mu_mumu_cmp"]) - float(row["mu_mumu_full"]))),
            }
        )
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _subset_rows(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df
    if subset == "release_hotspot":
        return df[df["release_hotspot_candidate"]].copy()
    if subset == "release_nonhotspot":
        return df[~df["release_hotspot_candidate"]].copy()
    raise ValueError(f"Unknown subset: {subset}")


def build_summary(map_df: pd.DataFrame, probe_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    probe_wide = probe_df.pivot(index=["surface", "scenario"], columns="point_kind")
    probe_wide.columns = ["__".join([str(a), str(b)]) for a, b in probe_wide.columns]
    probe_wide = probe_wide.reset_index()

    for (surface, scenario), grp0 in map_df.groupby(["surface", "scenario"], sort=True):
        probe = probe_wide[(probe_wide["surface"] == surface) & (probe_wide["scenario"] == scenario)]
        probe_row = probe.iloc[0].to_dict() if not probe.empty else {}
        for subset in ("all", "release_hotspot", "release_nonhotspot"):
            grp = _subset_rows(grp0, subset)
            if grp.empty:
                continue
            component_scores = {
                "b_visibility_norm": grp["b_visibility_norm_score"].to_numpy(dtype=float),
                "yraw_operator": grp["yraw_operator_score"].to_numpy(dtype=float),
                "m2_spectrum": grp["m2_spectrum_score"].to_numpy(dtype=float),
            }
            dominant = np.array([max(component_scores, key=lambda k: component_scores[k][i]) for i in range(len(grp))], dtype=object)
            rows.append(
                {
                    "surface": surface,
                    "scenario": scenario,
                    "subset": subset,
                    "n_points": int(len(grp)),
                    "mean_alpha_blend": float(grp["alpha_blend"].mean()),
                    "p95_abs_delta_mu_mumu_map": float(grp["abs_delta_mu_mumu_map"].quantile(0.95)),
                    "max_abs_delta_mu_mumu_map": float(grp["abs_delta_mu_mumu_map"].max()),
                    "p95_b_visibility_norm_score": float(grp["b_visibility_norm_score"].quantile(0.95)),
                    "p95_yraw_operator_score": float(grp["yraw_operator_score"].quantile(0.95)),
                    "p95_m2_spectrum_score": float(grp["m2_spectrum_score"].quantile(0.95)),
                    "corr_abs_delta_vs_b_visibility_norm": _safe_corr(grp["abs_delta_mu_mumu_map"].to_numpy(dtype=float), grp["b_visibility_norm_score"].to_numpy(dtype=float)),
                    "corr_abs_delta_vs_yraw_operator": _safe_corr(grp["abs_delta_mu_mumu_map"].to_numpy(dtype=float), grp["yraw_operator_score"].to_numpy(dtype=float)),
                    "corr_abs_delta_vs_m2_spectrum": _safe_corr(grp["abs_delta_mu_mumu_map"].to_numpy(dtype=float), grp["m2_spectrum_score"].to_numpy(dtype=float)),
                    "dominant_component_share_b_visibility_norm": float(np.mean(dominant == "b_visibility_norm")),
                    "dominant_component_share_yraw_operator": float(np.mean(dominant == "yraw_operator")),
                    "dominant_component_share_m2_spectrum": float(np.mean(dominant == "m2_spectrum")),
                    "probe_worst_D": probe_row.get("D__worst_abs_delta", np.nan),
                    "probe_worst_eta": probe_row.get("eta__worst_abs_delta", np.nan),
                    "probe_worst_log_width_ratio": probe_row.get("log_width_ratio__worst_abs_delta", np.nan),
                    "probe_worst_model_map_delta_residual": probe_row.get("model_map_delta_residual__worst_abs_delta", np.nan),
                    "probe_anchor_log_width_ratio": probe_row.get("log_width_ratio__reference_anchor", np.nan),
                    "probe_anchor_model_map_delta_residual": probe_row.get("model_map_delta_residual__reference_anchor", np.nan),
                }
            )

    piv = map_df.pivot_table(
        index=["surface", "D", "eta", "release_hotspot_candidate"],
        columns="scenario",
        values=["abs_delta_mu_mumu_map", "b_visibility_norm_score", "alpha_blend"],
    )
    piv.columns = ["__".join([str(a), str(b)]) for a, b in piv.columns]
    piv = piv.reset_index()
    for surface, grp0 in piv.groupby("surface", sort=True):
        for subset_name, mask in (
            ("all", np.ones(len(grp0), dtype=bool)),
            ("release_hotspot", grp0["release_hotspot_candidate"].to_numpy(dtype=bool)),
            ("release_nonhotspot", ~grp0["release_hotspot_candidate"].to_numpy(dtype=bool)),
        ):
            grp = grp0.loc[mask].copy()
            if grp.empty:
                continue
            delta_runtime = grp["abs_delta_mu_mumu_map__cell_direct_runtime_release"].to_numpy(dtype=float)
            delta_tuned = grp["abs_delta_mu_mumu_map__cell_direct_runtime_release_tuned"].to_numpy(dtype=float)
            b_runtime = grp["b_visibility_norm_score__cell_direct_runtime_release"].to_numpy(dtype=float)
            b_tuned = grp["b_visibility_norm_score__cell_direct_runtime_release_tuned"].to_numpy(dtype=float)
            rows.append(
                {
                    "surface": surface,
                    "scenario": "runtime_vs_release_tuned",
                    "subset": subset_name,
                    "n_points": int(len(grp)),
                    "mean_alpha_blend": float(np.mean(grp["alpha_blend__cell_direct_runtime_release_tuned"].to_numpy(dtype=float))),
                    "mean_abs_delta_runtime": float(np.mean(delta_runtime)),
                    "mean_abs_delta_release_tuned": float(np.mean(delta_tuned)),
                    "mean_abs_delta_reduction": float(np.mean(delta_runtime - delta_tuned)),
                    "p95_abs_delta_runtime": float(np.percentile(delta_runtime, 95.0)),
                    "p95_abs_delta_release_tuned": float(np.percentile(delta_tuned, 95.0)),
                    "p95_abs_delta_reduction": float(np.percentile(delta_runtime - delta_tuned, 95.0)),
                    "mean_b_visibility_runtime": float(np.mean(b_runtime)),
                    "mean_b_visibility_release_tuned": float(np.mean(b_tuned)),
                    "mean_b_visibility_reduction": float(np.mean(b_runtime - b_tuned)),
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(map_df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), constrained_layout=True)
    for row, surface in enumerate(["D21E41", "D60E21"]):
        sdf = map_df[map_df["surface"] == surface].copy()
        for scenario, color in (
            ("cell_direct_runtime_release", "#d95f02"),
            ("cell_direct_runtime_release_tuned", "#1b9e77"),
        ):
            grp = sdf[sdf["scenario"] == scenario].groupby("D", as_index=False).agg(
                abs_delta_mu_mumu_map=("abs_delta_mu_mumu_map", "mean"),
                b_visibility_norm_score=("b_visibility_norm_score", "mean"),
                yraw_operator_score=("yraw_operator_score", "mean"),
                m2_spectrum_score=("m2_spectrum_score", "mean"),
            ).sort_values("D")
            axes[row, 0].plot(grp["D"], grp["abs_delta_mu_mumu_map"], label=scenario, color=color)
        grp = sdf[sdf["scenario"] == "cell_direct_runtime_release_tuned"].groupby("D", as_index=False).agg(
            b_visibility_norm_score=("b_visibility_norm_score", "mean"),
            yraw_operator_score=("yraw_operator_score", "mean"),
            m2_spectrum_score=("m2_spectrum_score", "mean"),
            release_hotspot_candidate=("release_hotspot_candidate", "max"),
        ).sort_values("D")
        axes[row, 1].plot(grp["D"], grp["b_visibility_norm_score"], label="B visibility", color="#1b9e77")
        axes[row, 1].plot(grp["D"], grp["yraw_operator_score"], label="y_raw", color="#7570b3")
        axes[row, 1].plot(grp["D"], grp["m2_spectrum_score"], label="m2", color="#e7298a")
        hot_d = grp.loc[grp["release_hotspot_candidate"].astype(bool), "D"].to_numpy(dtype=float)
        for ax in axes[row]:
            for d_val in hot_d:
                ax.axvline(d_val, color="k", alpha=0.08, linewidth=1.0)
        axes[row, 0].set_title(f"{surface}: mean |Δμ_μμ|")
        axes[row, 1].set_title(f"{surface}: mean direct-component drift")
        axes[row, 0].set_xlabel("D")
        axes[row, 1].set_xlabel("D")
        axes[row, 0].set_ylabel("mean")
        axes[row, 1].set_ylabel("mean")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("Runtime-direct visibility component audit")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    instability = pd.read_csv(INSTABILITY_MAP)
    map_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []
    meta: Dict[str, object] = {"surfaces": {}}

    for surface, cfg in SURFACES.items():
        print(f"[audit] surface={surface}: init")
        full_map = _read_map(Path(cfg["full_map"]), "full")
        runtime_map = _read_map(Path(cfg["runtime_map"]), "cmp")
        release_map = _read_map(Path(cfg["release_map"]), "cmp")
        d_vals = np.sort(full_map["D"].astype(float).unique())
        ref_d = _snap_ref_d_to_grid(float(PAPER_BASELINE["ref_D"]), d_vals)
        ref_eta = float(PAPER_BASELINE["ref_eta"])
        kin = _build_kinetics(cfg)
        instability_surface = instability[instability["surface"] == surface][["D", "eta", "risk_score", "high_anchor_candidate"]].copy()

        caches = {
            "cell_direct_runtime_release": _collect_d_component_cache(surface, kin["full_direct"], kin["cell_direct_runtime_release"], "cell_direct_runtime_release", d_vals),
            "cell_direct_runtime_release_tuned": _collect_d_component_cache(surface, kin["full_direct"], kin["cell_direct_runtime_release_tuned"], "cell_direct_runtime_release_tuned", d_vals),
        }

        comparisons = [
            ("cell_direct_runtime_release", runtime_map, kin["cell_direct_runtime_release"]),
            ("cell_direct_runtime_release_tuned", release_map, kin["cell_direct_runtime_release_tuned"]),
        ]
        for scenario, cmp_map, kin_cmp in comparisons:
            print(f"[audit] surface={surface}: scenario={scenario}")
            merged = (
                full_map.merge(cmp_map, on=["D", "eta"], how="inner", validate="one_to_one")
                .merge(instability_surface, on=["D", "eta"], how="left", validate="one_to_one")
                .merge(caches[scenario].drop(columns=["surface", "scenario"]), on=["D"], how="left", validate="many_to_one")
                .sort_values(["D", "eta"])
                .reset_index(drop=True)
            )
            merged["scenario"] = scenario
            merged["surface"] = surface
            merged["risk_score_release_tuned"] = merged["risk_score"].astype(float)
            merged["release_hotspot_candidate"] = merged["high_anchor_candidate"].fillna(False).astype(bool)
            merged["abs_delta_mu_mumu_map"] = (merged["mu_mumu_cmp"] - merged["mu_mumu_full"]).abs()
            merged["rel_delta_mu_mumu_map"] = merged["abs_delta_mu_mumu_map"] / merged["mu_mumu_full"].abs().clip(lower=1e-30)
            map_rows.extend(merged.to_dict(orient="records"))
            probe_rows.extend(_collect_probe_rows(surface, scenario, full_map, cmp_map, kin["full_direct"], kin_cmp, ref_d, ref_eta))
        print(f"[audit] surface={surface}: done")

        meta["surfaces"][surface] = {"ref_D": ref_d, "ref_eta": ref_eta}

    map_df = pd.DataFrame(map_rows)
    probe_df = pd.DataFrame(probe_rows)
    summary_df = build_summary(map_df, probe_df)
    print("[audit] summary built")

    out_map = OUTDIR / "runtime_direct_visibility_component_audit_map.csv"
    out_probe = OUTDIR / "runtime_direct_visibility_component_audit_probes.csv"
    out_summary = OUTDIR / "runtime_direct_visibility_component_audit_summary.csv"
    out_png = OUTDIR / "runtime_direct_visibility_component_audit.png"
    out_meta = OUTDIR / "runtime_direct_visibility_component_audit_run_meta.json"

    map_df.sort_values(["surface", "scenario", "D", "eta"]).to_csv(out_map, index=False)
    probe_df.sort_values(["surface", "scenario", "point_kind"]).to_csv(out_probe, index=False)
    summary_df.to_csv(out_summary, index=False)
    plot_profiles(map_df, out_png)
    out_meta.write_text(json.dumps(meta, indent=2))

    for path in (out_map, out_probe, out_summary, out_png, out_meta):
        paper_path = PAPER_DIR / path.name
        if path.suffix.lower() == ".png":
            paper_path.write_bytes(path.read_bytes())
        else:
            paper_path.write_text(path.read_text())

    print(f"[ok] wrote {out_map}")
    print(f"[ok] wrote {out_probe}")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_meta}")


if __name__ == "__main__":
    main()
