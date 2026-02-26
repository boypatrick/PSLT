#!/usr/bin/env python3
"""
Multi-anchor calibration for open_system_micro kappa_env with holdout validation.

Design goals:
  - avoid full-map fitting;
  - calibrate kappa_env from multiple anchor D points;
  - report explicit holdout validation metrics on non-anchor D points.

Inputs:
  - output/chi_open_system/chi_open_system_micro_D*.csv
  - output/chi_open_system/chi_open_system_geometry_D*.csv

Outputs:
  - output/chi_open_system/kappa_env_anchor_calibration.csv
  - output/chi_open_system/kappa_env_anchor_calibration.json
  - output/chi_open_system/kappa_env_anchor_holdout.csv
  - output/chi_open_system/kappa_env_anchor_residuals.csv
  - mirrored CSV copies under paper/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class DRow:
    D: float
    delta: float
    gamma_phi_src: float
    gamma_mix_src: float
    gamma_ref: float
    chi_lr: float
    ratio_target: float


def _pick_largest_csv(pattern: str) -> Path:
    cands = sorted(OUTDIR.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No files matched: {OUTDIR / pattern}")
    best: Optional[Path] = None
    best_n = -1
    for p in cands:
        try:
            n = len(pd.read_csv(p))
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best = p
    if best is None:
        raise RuntimeError(f"Could not parse any CSV for pattern: {pattern}")
    return best


def _infer_kappa_src(df: pd.DataFrame) -> float:
    if "kappa_env" in df.columns:
        arr = df["kappa_env"].to_numpy(dtype=float)
        if np.isfinite(arr).any():
            return float(np.nanmedian(arr))
    num = df["gamma_mix_micro"].to_numpy(dtype=float)
    den = np.maximum(
        df["g_x_micro"].to_numpy(dtype=float) ** 2 * df["Sxx_delta"].to_numpy(dtype=float),
        1e-30,
    )
    return float(np.nanmedian(num / den))


def _pick_first_present(row: pd.Series, keys: Sequence[str]) -> float:
    for k in keys:
        if k in row.index:
            v = float(row[k])
            if np.isfinite(v):
                return v
    raise KeyError(f"None of keys found/finite: {keys}")


def _build_dataset(micro_csv: Path, geom_csv: Path) -> Tuple[pd.DataFrame, float]:
    df_micro = pd.read_csv(micro_csv).copy()
    df_geom = pd.read_csv(geom_csv).copy()

    if "D" not in df_micro.columns or "D" not in df_geom.columns:
        raise KeyError("Both CSVs must contain column 'D'.")
    if "ratio_proxy_to_baseline" not in df_geom.columns:
        raise KeyError("Geometry CSV must contain 'ratio_proxy_to_baseline'.")

    kappa_src = _infer_kappa_src(df_micro)
    micro_rows: List[Dict[str, float]] = []
    for _, r in df_micro.iterrows():
        dval = float(r["D"])
        try:
            micro_rows.append(
                {
                    "D": dval,
                    "delta": _pick_first_present(r, ("delta_micro", "delta")),
                    "gamma_phi_src": _pick_first_present(r, ("gamma_phi_micro", "gamma_phi")),
                    "gamma_mix_src": _pick_first_present(r, ("gamma_mix_micro", "gamma_mix")),
                    "gamma_ref": _pick_first_present(r, ("gamma_ref_micro", "gamma_ref", "Gamma_ref")),
                    "chi_lr": _pick_first_present(r, ("chi_LR_baseline",)),
                }
            )
        except KeyError:
            continue

    dfm = pd.DataFrame(micro_rows)
    dfg = df_geom[["D", "ratio_proxy_to_baseline"]].rename(columns={"ratio_proxy_to_baseline": "ratio_target"})
    df = pd.merge(dfm, dfg, on="D", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        raise RuntimeError("No overlapping valid D rows after merging micro and geometry CSVs.")

    df["gamma_phi_src"] = np.maximum(df["gamma_phi_src"].to_numpy(dtype=float), 1e-30)
    df["gamma_mix_src"] = np.maximum(df["gamma_mix_src"].to_numpy(dtype=float), 1e-30)
    df["gamma_ref"] = np.maximum(df["gamma_ref"].to_numpy(dtype=float), 1e-30)
    df["chi_lr"] = np.maximum(df["chi_lr"].to_numpy(dtype=float), 1e-30)
    df = df.sort_values("D").reset_index(drop=True)
    return df, float(kappa_src)


def _parse_d_list(text: str) -> List[float]:
    if not text.strip():
        return []
    out: List[float] = []
    for tok in text.split(","):
        s = tok.strip()
        if s:
            out.append(float(s))
    return out


def _lindblad_cmax(delta: float, gamma_phi: float, gamma_mix: float, tmax: float, nstep: int, rtol: float, atol: float) -> float:
    h = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    l_ops = [np.sqrt(max(gamma_phi, 0.0)) * sigma_z, np.sqrt(max(gamma_mix, 0.0)) * sigma_x]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rho = y.reshape(2, 2)
        drho = -1j * (h @ rho - rho @ h)
        for l_op in l_ops:
            drho += l_op @ rho @ l_op.conj().T - 0.5 * (l_op.conj().T @ l_op @ rho + rho @ l_op.conj().T @ l_op)
        return drho.reshape(-1)

    y0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
    t_eval = np.linspace(0.0, tmax, nstep)
    sol = solve_ivp(rhs, (0.0, tmax), y0, t_eval=t_eval, rtol=rtol, atol=atol)
    cmax = 0.0
    for i in range(sol.y.shape[1]):
        rho = sol.y[:, i].reshape(2, 2)
        cmax = max(cmax, float(abs(rho[0, 1])))
    return cmax


def _ratio_at_kappa(kappa: float, row: DRow, kappa_src: float, tmax: float, nstep: int, rtol: float, atol: float) -> float:
    scale = float(kappa / max(kappa_src, 1e-30))
    gamma_phi = float(row.gamma_phi_src * scale)
    gamma_mix = float(row.gamma_mix_src * scale)
    cmax = _lindblad_cmax(
        delta=row.delta,
        gamma_phi=gamma_phi,
        gamma_mix=gamma_mix,
        tmax=tmax,
        nstep=nstep,
        rtol=rtol,
        atol=atol,
    )
    chi_eff = float(max(2.0 * gamma_mix * cmax / max(row.gamma_ref, 1e-30), 0.0))
    return float(chi_eff / max(row.chi_lr, 1e-30))


def _fit_metrics(kappa: float, rows: Sequence[DRow], kappa_src: float, tmax: float, nstep: int, rtol: float, atol: float) -> Tuple[float, float]:
    pred = np.asarray([_ratio_at_kappa(kappa, r, kappa_src, tmax, nstep, rtol, atol) for r in rows], dtype=float)
    tgt = np.asarray([r.ratio_target for r in rows], dtype=float)
    abs_err = np.abs(pred - tgt)
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    mae = float(np.mean(abs_err))
    return rmse, mae


def _find_best_kappa(
    rows_anchor: Sequence[DRow],
    kappa_src: float,
    kappa_min: float,
    kappa_max: float,
    n_coarse: int,
    tmax: float,
    nstep: int,
    rtol: float,
    atol: float,
) -> float:
    if not (kappa_min > 0 and kappa_max > kappa_min):
        raise ValueError("Require 0 < kappa_min < kappa_max.")
    k_coarse = np.geomspace(kappa_min, kappa_max, max(int(n_coarse), 7))
    coarse_eval: List[Tuple[float, float, float]] = []
    for k in k_coarse:
        rmse, mae = _fit_metrics(float(k), rows_anchor, kappa_src, tmax, nstep, rtol, atol)
        coarse_eval.append((float(k), rmse, mae))
    k_best, _, _ = min(coarse_eval, key=lambda x: (x[1], x[2]))

    logk = np.log10(k_best)
    k_lo = max(kappa_min, 10 ** (logk - 0.30))
    k_hi = min(kappa_max, 10 ** (logk + 0.30))
    k_ref = np.geomspace(k_lo, k_hi, 25)
    ref_eval: List[Tuple[float, float, float]] = []
    for k in k_ref:
        rmse, mae = _fit_metrics(float(k), rows_anchor, kappa_src, tmax, nstep, rtol, atol)
        ref_eval.append((float(k), rmse, mae))
    k_best, _, _ = min(ref_eval, key=lambda x: (x[1], x[2]))
    return float(k_best)


def _format_d_list(vals: Sequence[float]) -> str:
    out: List[str] = []
    for v in vals:
        if float(v).is_integer():
            out.append(str(int(round(float(v)))))
        else:
            out.append(f"{float(v):.6g}")
    return ",".join(out)


def _stats_from_abs(abs_err: np.ndarray) -> Dict[str, float]:
    if abs_err.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs_err": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "mae": float(np.mean(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate kappa_env via multi-anchor fit with holdout validation.")
    ap.add_argument("--micro-csv", default="", help="Path to chi_open_system_micro_D*.csv (default: auto-pick largest).")
    ap.add_argument("--geom-csv", default="", help="Path to chi_open_system_geometry_D*.csv (default: auto-pick largest).")
    ap.add_argument("--D-ref", type=float, default=12.0, help="Fallback anchor if --anchor-Ds is empty.")
    ap.add_argument("--anchor-Ds", default="6,9,12,15,18", help="Comma-separated anchor D list.")
    ap.add_argument(
        "--holdout-Ds",
        default="",
        help="Comma-separated holdout D list. Default: all available D not in anchors.",
    )
    ap.add_argument("--kappa-min", type=float, default=1e4)
    ap.add_argument("--kappa-max", type=float, default=2e6)
    ap.add_argument("--n-coarse", type=int, default=13)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=300)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    micro_csv = Path(args.micro_csv) if args.micro_csv else _pick_largest_csv("chi_open_system_micro_D*.csv")
    geom_csv = Path(args.geom_csv) if args.geom_csv else _pick_largest_csv("chi_open_system_geometry_D*.csv")
    df, kappa_src = _build_dataset(micro_csv=micro_csv, geom_csv=geom_csv)

    d_all = df["D"].to_numpy(dtype=float)
    d_set = {float(d) for d in d_all}
    anchor_ds = _parse_d_list(args.anchor_Ds)
    if not anchor_ds:
        anchor_ds = [float(args.D_ref)]
    anchor_ds = [float(d) for d in anchor_ds]
    if len(anchor_ds) < 2:
        raise ValueError("multi-anchor calibration requires at least 2 anchors.")
    for d in anchor_ds:
        if d not in d_set:
            raise ValueError(f"anchor D={d} is not available in merged dataset.")

    holdout_ds = _parse_d_list(args.holdout_Ds)
    if holdout_ds:
        for d in holdout_ds:
            if d not in d_set:
                raise ValueError(f"holdout D={d} is not available in merged dataset.")
    else:
        holdout_ds = [float(d) for d in d_all if float(d) not in set(anchor_ds)]
    if len(holdout_ds) == 0:
        raise ValueError("holdout set is empty. Provide --holdout-Ds or use fewer anchors.")

    anchor_mask = df["D"].isin(anchor_ds).to_numpy(dtype=bool)
    holdout_mask = df["D"].isin(holdout_ds).to_numpy(dtype=bool)
    rows_anchor = [
        DRow(
            D=float(r["D"]),
            delta=float(r["delta"]),
            gamma_phi_src=float(r["gamma_phi_src"]),
            gamma_mix_src=float(r["gamma_mix_src"]),
            gamma_ref=float(r["gamma_ref"]),
            chi_lr=float(r["chi_lr"]),
            ratio_target=float(r["ratio_target"]),
        )
        for _, r in df[anchor_mask].iterrows()
    ]

    k_best = _find_best_kappa(
        rows_anchor=rows_anchor,
        kappa_src=float(kappa_src),
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        n_coarse=int(args.n_coarse),
        tmax=float(args.tmax),
        nstep=int(args.nstep),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    pred_all = np.asarray(
        [
            _ratio_at_kappa(
                float(k_best),
                DRow(
                    D=float(r["D"]),
                    delta=float(r["delta"]),
                    gamma_phi_src=float(r["gamma_phi_src"]),
                    gamma_mix_src=float(r["gamma_mix_src"]),
                    gamma_ref=float(r["gamma_ref"]),
                    chi_lr=float(r["chi_lr"]),
                    ratio_target=float(r["ratio_target"]),
                ),
                float(kappa_src),
                float(args.tmax),
                int(args.nstep),
                float(args.rtol),
                float(args.atol),
            )
            for _, r in df.iterrows()
        ],
        dtype=float,
    )
    tgt_all = df["ratio_target"].to_numpy(dtype=float)
    abs_err_all = np.abs(pred_all - tgt_all)
    signed_all = pred_all - tgt_all

    anchor_abs = abs_err_all[anchor_mask]
    holdout_abs = abs_err_all[holdout_mask]

    anchor_stats = _stats_from_abs(anchor_abs)
    holdout_stats = _stats_from_abs(holdout_abs)
    overall_stats = _stats_from_abs(abs_err_all)

    anchor_tgt_mean = float(np.mean(tgt_all[anchor_mask]))
    anchor_pred_mean = float(np.mean(pred_all[anchor_mask]))
    anchor_abs_mean_gap = float(abs(anchor_pred_mean - anchor_tgt_mean))

    d_ref = float(np.median(np.asarray(anchor_ds, dtype=float)))
    summary: Dict[str, float | int | str] = {
        "kappa_env_calibrated": float(k_best),
        "ratio_anchor_target": anchor_tgt_mean,
        "ratio_anchor_pred": anchor_pred_mean,
        "ratio_anchor_abs_err": anchor_abs_mean_gap,
        "D_ref": d_ref,
        "anchor_mode": "multi_anchor_holdout",
        "n_anchor": int(np.sum(anchor_mask)),
        "n_holdout": int(np.sum(holdout_mask)),
        "D_anchor_list": _format_d_list(anchor_ds),
        "D_holdout_list": _format_d_list(holdout_ds),
        "anchor_rmse": float(anchor_stats["rmse"]),
        "anchor_mae": float(anchor_stats["mae"]),
        "anchor_max_abs_err": float(anchor_stats["max_abs_err"]),
        "holdout_rmse": float(holdout_stats["rmse"]),
        "holdout_mae": float(holdout_stats["mae"]),
        "holdout_max_abs_err": float(holdout_stats["max_abs_err"]),
        "overall_rmse": float(overall_stats["rmse"]),
        "overall_mae": float(overall_stats["mae"]),
        "overall_max_abs_err": float(overall_stats["max_abs_err"]),
        "kappa_src_micro_csv": float(kappa_src),
        "micro_source_csv": str(micro_csv),
        "target_source_csv": str(geom_csv),
        "target_definition": "multi-anchor least-squares fit on ratio_proxy_to_baseline with explicit holdout validation",
    }

    residual_rows: List[Dict[str, float | str | int]] = []
    for i, (_, r) in enumerate(df.iterrows()):
        split = "anchor" if bool(anchor_mask[i]) else ("holdout" if bool(holdout_mask[i]) else "unused")
        residual_rows.append(
            {
                "D": float(r["D"]),
                "split": split,
                "ratio_target": float(tgt_all[i]),
                "ratio_pred": float(pred_all[i]),
                "residual_signed": float(signed_all[i]),
                "residual_abs": float(abs_err_all[i]),
            }
        )

    holdout_table = {
        "n_holdout": int(np.sum(holdout_mask)),
        "holdout_rmse": float(holdout_stats["rmse"]),
        "holdout_mae": float(holdout_stats["mae"]),
        "holdout_max_abs_err": float(holdout_stats["max_abs_err"]),
        "anchor_rmse": float(anchor_stats["rmse"]),
        "anchor_mae": float(anchor_stats["mae"]),
        "anchor_max_abs_err": float(anchor_stats["max_abs_err"]),
        "overall_rmse": float(overall_stats["rmse"]),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUTDIR / "kappa_env_anchor_calibration.csv"
    out_json = OUTDIR / "kappa_env_anchor_calibration.json"
    out_holdout = OUTDIR / "kappa_env_anchor_holdout.csv"
    out_resid = OUTDIR / "kappa_env_anchor_residuals.csv"

    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))
    pd.DataFrame([holdout_table]).to_csv(out_holdout, index=False)
    pd.DataFrame(residual_rows).to_csv(out_resid, index=False)

    (PAPER_DIR / out_csv.name).write_text(out_csv.read_text())
    (PAPER_DIR / out_holdout.name).write_text(out_holdout.read_text())
    (PAPER_DIR / out_resid.name).write_text(out_resid.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_holdout}")
    print(f"[saved] {out_resid}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

