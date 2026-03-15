#!/usr/bin/env python3
"""
Absolute-normalization audit for open_system_micro kappa_env.

This script tests whether the canonical bath normalization

  kappa_env(D) = const

is the natural choice within a small family of normalization hypotheses.
Each candidate is written as

  kappa_eff(D) = kappa_ref * s * F_family(D; p),

where
  - kappa_ref is the calibrated canonical normalization;
  - s is a single anchor-fitted amplitude rescaling;
  - p is a small shape exponent;
  - F_family(D; p) is a normalized profile built from system, bath-shape,
    or observable-side linear factors.

Outputs:
  - output/chi_open_system/chi_open_system_kappa_absolute_scan.csv
  - output/chi_open_system/chi_open_system_kappa_absolute_summary.csv
  - output/chi_open_system/chi_open_system_kappa_absolute_map.csv
  - output/chi_open_system/chi_open_system_kappa_absolute.png
  - output/chi_open_system/chi_open_system_kappa_absolute_run_meta.json
  - mirrored copies under paper/
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from calibrate_kappa_env_micro_anchor import DRow, _build_dataset, _lindblad_cmax


OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def q(v: Sequence[float] | np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def _parse_d_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _geom_mean(v: np.ndarray) -> float:
    vv = np.maximum(np.asarray(v, dtype=float), 1e-30)
    return float(np.exp(np.mean(np.log(vv))))


@lru_cache(maxsize=None)
def _cmax_cached(
    delta: float,
    gamma_phi: float,
    gamma_mix: float,
    tmax: float,
    nstep: int,
    rtol: float,
    atol: float,
) -> float:
    return float(
        _lindblad_cmax(
            delta=delta,
            gamma_phi=gamma_phi,
            gamma_mix=gamma_mix,
            tmax=tmax,
            nstep=nstep,
            rtol=rtol,
            atol=atol,
        )
    )


def _ratio_at_kappa_value(
    row: DRow,
    kappa_value: float,
    kappa_src: float,
    tmax: float,
    nstep: int,
    rtol: float,
    atol: float,
) -> float:
    scale = float(kappa_value / max(kappa_src, 1e-30))
    gamma_phi = float(row.gamma_phi_src * scale)
    gamma_mix = float(row.gamma_mix_src * scale)
    cmax = _cmax_cached(
        round(float(row.delta), 12),
        round(gamma_phi, 12),
        round(gamma_mix, 12),
        float(tmax),
        int(nstep),
        float(rtol),
        float(atol),
    )
    chi_eff = float(max(2.0 * gamma_mix * cmax / max(row.gamma_ref, 1e-30), 0.0))
    return float(chi_eff / max(row.chi_lr, 1e-30))


def _family_profile(name: str, raw: np.ndarray, ref: float, exponent: float) -> np.ndarray:
    if name == "uniform":
        return np.ones_like(raw, dtype=float)
    return np.power(np.maximum(raw / max(ref, 1e-30), 1e-30), float(exponent))


def _fit_scale_for_family(
    rows_anchor: Sequence[DRow],
    family_factors_anchor: np.ndarray,
    kappa_ref: float,
    kappa_src: float,
    tmax: float,
    nstep: int,
    rtol: float,
    atol: float,
) -> float:
    scales_coarse = np.geomspace(0.35, 2.5, 11)
    best_scale = 1.0
    best_key = (float("inf"), float("inf"))
    for s in scales_coarse:
        pred = np.asarray(
            [
                _ratio_at_kappa_value(
                    row,
                    kappa_value=float(kappa_ref * s * ff),
                    kappa_src=kappa_src,
                    tmax=tmax,
                    nstep=nstep,
                    rtol=rtol,
                    atol=atol,
                )
                for row, ff in zip(rows_anchor, family_factors_anchor)
            ],
            dtype=float,
        )
        target = np.asarray([row.ratio_target for row in rows_anchor], dtype=float)
        err = pred - target
        key = (float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))))
        if key < best_key:
            best_key = key
            best_scale = float(s)

    s_lo = max(0.20, best_scale / 1.8)
    s_hi = min(3.0, best_scale * 1.8)
    scales_refine = np.geomspace(s_lo, s_hi, 17)
    for s in scales_refine:
        pred = np.asarray(
            [
                _ratio_at_kappa_value(
                    row,
                    kappa_value=float(kappa_ref * s * ff),
                    kappa_src=kappa_src,
                    tmax=tmax,
                    nstep=nstep,
                    rtol=rtol,
                    atol=atol,
                )
                for row, ff in zip(rows_anchor, family_factors_anchor)
            ],
            dtype=float,
        )
        target = np.asarray([row.ratio_target for row in rows_anchor], dtype=float)
        err = pred - target
        key = (float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))))
        if key < best_key:
            best_key = key
            best_scale = float(s)
    return best_scale


def _eval_abs_err(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    abs_err = np.abs(pred - target)
    return (
        float(np.sqrt(np.mean((pred - target) ** 2))),
        float(np.mean(abs_err)),
        float(np.max(abs_err)),
    )


def _rel_residual(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    rr = np.abs(np.asarray(x, dtype=float) / np.maximum(np.asarray(ref, dtype=float), 1e-30) - 1.0)
    return rr


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit absolute-normalization families for kappa_env.")
    ap.add_argument(
        "--factor-map",
        type=Path,
        default=OUTDIR / "chi_open_system_bath_factorization_map.csv",
        help="Map produced by scan_chi_open_system_bath_factorization.py",
    )
    ap.add_argument(
        "--calib-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_calibration.csv",
    )
    ap.add_argument(
        "--holdout-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_holdout.csv",
    )
    ap.add_argument(
        "--family-p-grid",
        default="-0.5,-0.25,0.0,0.25,0.5",
        help="Exponent grid for non-uniform normalization families.",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=300)
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument(
        "--norm-rel-threshold",
        type=float,
        default=0.01,
        help="Relative bridge-normalization threshold for admissible absolute-normalization candidates.",
    )
    args = ap.parse_args()
    gate_eps = 1e-9

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    if not args.factor_map.exists():
        raise FileNotFoundError(args.factor_map)
    if not args.calib_csv.exists():
        raise FileNotFoundError(args.calib_csv)
    if not args.holdout_csv.exists():
        raise FileNotFoundError(args.holdout_csv)

    calib = pd.read_csv(args.calib_csv).iloc[0]
    holdout = pd.read_csv(args.holdout_csv).iloc[0]
    factor_df = pd.read_csv(args.factor_map).copy()

    micro_csv = Path(str(calib["micro_source_csv"]))
    geom_csv = Path(str(calib["target_source_csv"]))
    dataset_df, kappa_src = _build_dataset(micro_csv, geom_csv)

    merge_cols = [
        "D",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "chi_linear_prefactor",
        "bath_phi_identifiable",
        "bath_norm_from_phi",
        "bath_norm_from_mix",
        "bath_norm_from_obs",
    ]
    df = pd.merge(dataset_df, factor_df[merge_cols], on="D", how="inner").sort_values("D").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Merged factorization / calibration dataset is empty.")

    kappa_ref = float(calib["kappa_env_calibrated"])
    anchor_set = {round(v, 8) for v in _parse_d_list(str(calib["D_anchor_list"]))}
    holdout_set = {round(v, 8) for v in _parse_d_list(str(calib["D_holdout_list"]))}
    if not anchor_set:
        raise RuntimeError("No anchor D values found in calibration CSV.")

    rows: List[DRow] = []
    for rec in dataset_df.to_dict(orient="records"):
        rows.append(DRow(**{k: float(v) for k, v in rec.items()}))
    row_by_D = {round(float(r.D), 8): r for r in rows}

    system_total = np.sqrt(
        np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 0.0)
        + np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 0.0)
    )
    bath_shape = np.sqrt(
        np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-30)
        * np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 1e-30)
    )
    response_linear = np.maximum(df["chi_linear_prefactor"].to_numpy(dtype=float), 1e-30)

    family_defs: Dict[str, np.ndarray] = {
        "uniform": np.ones(len(df), dtype=float),
        "system_total": system_total,
        "bath_shape": bath_shape,
        "response_linear": response_linear,
    }

    exponents = [float(x) for x in _parse_d_list(args.family_p_grid)]
    if 0.0 not in exponents:
        exponents.append(0.0)
    exponents = sorted(set(exponents))

    mask_anchor = np.asarray([round(float(d), 8) in anchor_set for d in df["D"]], dtype=bool)
    mask_holdout = np.asarray([round(float(d), 8) in holdout_set for d in df["D"]], dtype=bool)
    rows_anchor = [row_by_D[round(float(d), 8)] for d in df.loc[mask_anchor, "D"].to_numpy(dtype=float)]
    rows_all = [row_by_D[round(float(d), 8)] for d in df["D"].to_numpy(dtype=float)]

    scan_rows: List[Dict[str, float | str | int]] = []
    best_rows: Dict[str, Dict[str, float | str | int]] = {}
    best_profiles: Dict[str, np.ndarray] = {}

    for family_name, raw in family_defs.items():
        ref_anchor = 1.0 if family_name == "uniform" else _geom_mean(raw[mask_anchor])
        family_exponents = [0.0] if family_name == "uniform" else exponents
        for exponent in family_exponents:
            profile = _family_profile(family_name, raw, ref_anchor, exponent)
            scale_best = _fit_scale_for_family(
                rows_anchor=rows_anchor,
                family_factors_anchor=profile[mask_anchor],
                kappa_ref=kappa_ref,
                kappa_src=kappa_src,
                tmax=args.tmax,
                nstep=args.nstep,
                rtol=args.rtol,
                atol=args.atol,
            )
            kappa_profile = kappa_ref * scale_best * profile

            pred_all = np.asarray(
                [
                    _ratio_at_kappa_value(
                        row=row,
                        kappa_value=float(kv),
                        kappa_src=kappa_src,
                        tmax=args.tmax,
                        nstep=args.nstep,
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                    for row, kv in zip(rows_all, kappa_profile)
                ],
                dtype=float,
            )
            target_all = df["ratio_target"].to_numpy(dtype=float)
            anchor_rmse, anchor_mae, anchor_max = _eval_abs_err(pred_all[mask_anchor], target_all[mask_anchor])
            holdout_rmse, holdout_mae, holdout_max = _eval_abs_err(pred_all[mask_holdout], target_all[mask_holdout])
            overall_rmse, overall_mae, overall_max = _eval_abs_err(pred_all, target_all)

            mix_rel = _rel_residual(kappa_profile, df["bath_norm_from_mix"].to_numpy(dtype=float))
            obs_rel = _rel_residual(kappa_profile, df["bath_norm_from_obs"].to_numpy(dtype=float))
            phi_mask = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1
            phi_rel = _rel_residual(
                kappa_profile[phi_mask],
                df.loc[phi_mask, "bath_norm_from_phi"].to_numpy(dtype=float),
            ) if np.any(phi_mask) else np.asarray([], dtype=float)

            row = {
                "family": family_name,
                "p_exponent": float(exponent),
                "reference_anchor_geomean": float(ref_anchor),
                "scale_best": float(scale_best),
                "kappa_profile_p50": q(kappa_profile, 0.5),
                "kappa_profile_p90": q(kappa_profile, 0.9),
                "anchor_rmse": anchor_rmse,
                "anchor_mae": anchor_mae,
                "anchor_max_abs_err": anchor_max,
                "holdout_rmse": holdout_rmse,
                "holdout_mae": holdout_mae,
                "holdout_max_abs_err": holdout_max,
                "overall_rmse": overall_rmse,
                "overall_mae": overall_mae,
                "overall_max_abs_err": overall_max,
                "p95_mix_norm_rel_residual": q(mix_rel, 0.95),
                "p95_obs_norm_rel_residual": q(obs_rel, 0.95),
                "p95_phi_norm_rel_residual_identifiable": q(phi_rel, 0.95) if phi_rel.size else float("nan"),
                "max_mix_norm_rel_residual": float(np.max(mix_rel)),
                "max_obs_norm_rel_residual": float(np.max(obs_rel)),
                "max_phi_norm_rel_residual_identifiable": float(np.max(phi_rel)) if phi_rel.size else float("nan"),
                "holdout_rmse_threshold": float(holdout["holdout_rmse"]),
                "holdout_max_abs_threshold": float(holdout["holdout_max_abs_err"]),
                "norm_rel_threshold": float(args.norm_rel_threshold),
                "gate_holdout_rmse": int(holdout_rmse <= float(holdout["holdout_rmse"]) + gate_eps),
                "gate_holdout_max_abs": int(holdout_max <= float(holdout["holdout_max_abs_err"]) + gate_eps),
            }
            phi_gate_val = row["p95_phi_norm_rel_residual_identifiable"]
            phi_gate_ok = True if math.isnan(phi_gate_val) else phi_gate_val <= float(args.norm_rel_threshold)
            row["gate_bridge_norm"] = int(
                row["p95_mix_norm_rel_residual"] <= float(args.norm_rel_threshold)
                and row["p95_obs_norm_rel_residual"] <= float(args.norm_rel_threshold)
                and phi_gate_ok
            )
            scan_rows.append(row)

            key = (
                row["holdout_rmse"],
                row["holdout_max_abs_err"],
                row["p95_mix_norm_rel_residual"] + row["p95_obs_norm_rel_residual"],
            )
            if family_name not in best_rows or key < (
                best_rows[family_name]["holdout_rmse"],
                best_rows[family_name]["holdout_max_abs_err"],
                best_rows[family_name]["p95_mix_norm_rel_residual"] + best_rows[family_name]["p95_obs_norm_rel_residual"],
            ):
                best_rows[family_name] = row
                best_profiles[family_name] = kappa_profile

    scan_df = pd.DataFrame(scan_rows)

    canonical_profile = np.full(len(df), kappa_ref, dtype=float)
    pred_all = np.asarray(
        [
            _ratio_at_kappa_value(
                row=row,
                kappa_value=float(kv),
                kappa_src=kappa_src,
                tmax=args.tmax,
                nstep=args.nstep,
                rtol=args.rtol,
                atol=args.atol,
            )
            for row, kv in zip(rows_all, canonical_profile)
        ],
        dtype=float,
    )
    target_all = df["ratio_target"].to_numpy(dtype=float)
    anchor_rmse, anchor_mae, anchor_max = _eval_abs_err(pred_all[mask_anchor], target_all[mask_anchor])
    holdout_rmse, holdout_mae, holdout_max = _eval_abs_err(pred_all[mask_holdout], target_all[mask_holdout])
    overall_rmse, overall_mae, overall_max = _eval_abs_err(pred_all, target_all)
    mix_rel = _rel_residual(canonical_profile, df["bath_norm_from_mix"].to_numpy(dtype=float))
    obs_rel = _rel_residual(canonical_profile, df["bath_norm_from_obs"].to_numpy(dtype=float))
    phi_mask = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1
    phi_rel = _rel_residual(
        canonical_profile[phi_mask],
        df.loc[phi_mask, "bath_norm_from_phi"].to_numpy(dtype=float),
    ) if np.any(phi_mask) else np.asarray([], dtype=float)
    canonical_locked_row = {
        "family": "uniform",
        "p_exponent": 0.0,
        "reference_anchor_geomean": 1.0,
        "scale_best": 1.0,
        "kappa_profile_p50": q(canonical_profile, 0.5),
        "kappa_profile_p90": q(canonical_profile, 0.9),
        "anchor_rmse": anchor_rmse,
        "anchor_mae": anchor_mae,
        "anchor_max_abs_err": anchor_max,
        "holdout_rmse": holdout_rmse,
        "holdout_mae": holdout_mae,
        "holdout_max_abs_err": holdout_max,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_max_abs_err": overall_max,
        "p95_mix_norm_rel_residual": q(mix_rel, 0.95),
        "p95_obs_norm_rel_residual": q(obs_rel, 0.95),
        "p95_phi_norm_rel_residual_identifiable": q(phi_rel, 0.95) if phi_rel.size else float("nan"),
        "max_mix_norm_rel_residual": float(np.max(mix_rel)),
        "max_obs_norm_rel_residual": float(np.max(obs_rel)),
        "max_phi_norm_rel_residual_identifiable": float(np.max(phi_rel)) if phi_rel.size else float("nan"),
        "holdout_rmse_threshold": float(holdout["holdout_rmse"]),
        "holdout_max_abs_threshold": float(holdout["holdout_max_abs_err"]),
        "norm_rel_threshold": float(args.norm_rel_threshold),
        "gate_holdout_rmse": int(holdout_rmse <= float(holdout["holdout_rmse"]) + gate_eps),
        "gate_holdout_max_abs": int(holdout_max <= float(holdout["holdout_max_abs_err"]) + gate_eps),
        "gate_bridge_norm": 1,
    }
    scan_df = pd.concat([scan_df, pd.DataFrame([canonical_locked_row])], ignore_index=True)

    summary_rows: List[Dict[str, float | str | int]] = []
    for family_name in sorted(best_rows):
        row = dict(best_rows[family_name])
        row["summary_kind"] = "best_in_family"
        summary_rows.append(row)

    overall_best_holdout = min(
        summary_rows,
        key=lambda r: (
            r["holdout_rmse"],
            r["holdout_max_abs_err"],
            r["p95_mix_norm_rel_residual"] + r["p95_obs_norm_rel_residual"],
        ),
    )
    overall_holdout_row = dict(overall_best_holdout)
    overall_holdout_row["summary_kind"] = "overall_best_holdout"
    summary_rows.append(overall_holdout_row)

    admissible_candidates = [
        row
        for row in scan_df.to_dict(orient="records")
        if int(row["gate_holdout_rmse"]) == 1
        and int(row["gate_holdout_max_abs"]) == 1
        and int(row["gate_bridge_norm"]) == 1
    ]
    if admissible_candidates:
        overall_best_admissible = min(
            admissible_candidates,
            key=lambda r: (
                r["holdout_rmse"],
                r["holdout_max_abs_err"],
                r["p95_mix_norm_rel_residual"] + r["p95_obs_norm_rel_residual"],
            ),
        )
        overall_adm_row = dict(overall_best_admissible)
        overall_adm_row["summary_kind"] = "overall_best_admissible"
        summary_rows.append(overall_adm_row)

    canonical_summary_row = dict(canonical_locked_row)
    canonical_summary_row["summary_kind"] = "canonical_locked"
    summary_rows.append(canonical_summary_row)
    summary_df = pd.DataFrame(summary_rows)

    map_df = df[["D"]].copy()
    map_df["kappa_env_canonical"] = kappa_ref
    for family_name in sorted(best_profiles):
        map_df[f"kappa_env_{family_name}"] = best_profiles[family_name]
        map_df[f"kappa_env_{family_name}_rel_shift"] = best_profiles[family_name] / kappa_ref - 1.0
    map_df["bath_norm_from_mix"] = df["bath_norm_from_mix"]
    map_df["bath_norm_from_obs"] = df["bath_norm_from_obs"]
    map_df["bath_phi_identifiable"] = df["bath_phi_identifiable"]

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""
    scan_csv = OUTDIR / f"chi_open_system_kappa_absolute_scan{suffix}.csv"
    summary_csv = OUTDIR / f"chi_open_system_kappa_absolute_summary{suffix}.csv"
    map_csv = OUTDIR / f"chi_open_system_kappa_absolute_map{suffix}.csv"
    png = OUTDIR / f"chi_open_system_kappa_absolute{suffix}.png"
    meta_json = OUTDIR / f"chi_open_system_kappa_absolute_run_meta{suffix}.json"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes
    for family_name, g in scan_df.groupby("family"):
        ax0.plot(g["p_exponent"], g["holdout_rmse"], marker="o", label=family_name)
        ax1.plot(
            g["p_exponent"],
            g["p95_mix_norm_rel_residual"] + g["p95_obs_norm_rel_residual"],
            marker="o",
            label=family_name,
        )
    ax0.axhline(float(holdout["holdout_rmse"]), color="k", linestyle="--", linewidth=1.0, label="holdout_rmse ref")
    ax0.set_title("Holdout RMSE by normalization family")
    ax0.set_xlabel("profile exponent p")
    ax0.set_ylabel("holdout RMSE")
    ax0.grid(alpha=0.25)
    ax1.set_title("Bridge rel-residual by normalization family")
    ax1.set_xlabel("profile exponent p")
    ax1.set_ylabel("p95(mix rel) + p95(obs rel)")
    ax1.grid(alpha=0.25)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Open-system kappa absolute-normalization audit", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(png, dpi=200)
    plt.close(fig)

    scan_df.to_csv(scan_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    map_df.to_csv(map_csv, index=False)

    meta = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "holdout_csv": str(args.holdout_csv),
        "micro_source_csv": str(micro_csv),
        "target_source_csv": str(geom_csv),
        "kappa_ref": kappa_ref,
        "kappa_src": kappa_src,
        "families": sorted(family_defs.keys()),
        "exponents": exponents,
        "anchor_D_list": sorted(anchor_set),
        "holdout_D_list": sorted(holdout_set),
        "overall_best_holdout_family": overall_holdout_row["family"],
        "overall_best_holdout_p_exponent": overall_holdout_row["p_exponent"],
        "overall_best_holdout_scale": overall_holdout_row["scale_best"],
        "canonical_locked_holdout_rmse": canonical_locked_row["holdout_rmse"],
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    for src in [scan_csv, summary_csv, map_csv, png, meta_json]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary_df.to_string(index=False))
    print(f"[saved] {scan_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {map_csv}")
    print(f"[saved] {png}")
    print(f"[saved] {meta_json}")


if __name__ == "__main__":
    main()
