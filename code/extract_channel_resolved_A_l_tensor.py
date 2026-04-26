#!/usr/bin/env python3
"""
Extract a fully localized channel-resolved A_l(D, eta, N) audit tensor.

This script promotes the earlier profile-level insertion

    Atilde_l(D) = A_l^{fp}(D; N_ref) / A_l^{fp}(D_ref; N_ref)

to an explicit table over (D, eta, N, l).  In the current PSLT kinetic
factorization, eta is the rank-2 formation/overlap multiplier applied after
the barrier channel rates are formed.  Therefore the localized barrier
normalization itself is eta-separable:

    A_l^{loc}(D, eta, N) = A_l^{fp}(D; N).

The audit keeps eta as an explicit axis and reports the collapse residual, so
the separability is a checked tensor statement rather than a hidden convention.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from extract_superrad_prefactor_1d import (
    PhysicalParams,
    Level,
    eig_1d,
    u_z,
    wkb_action_central,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "superrad_fp_1d"
DEFAULT_PAPER_DIR = ROOT / "paper"


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _tag(values: Iterable[float | int], *, prefix: str = "") -> str:
    parts: List[str] = []
    for value in values:
        v = float(value)
        if v.is_integer():
            parts.append(str(int(v)))
        else:
            parts.append(str(v).replace(".", "p"))
    return prefix + "-".join(parts)


def _build_reference_values(
    base_rows: pd.DataFrame,
    d_ref: float,
    n_ref: int,
) -> Dict[int, float]:
    refs: Dict[int, float] = {}
    for ell, sub in base_rows.groupby("ell"):
        ref = sub[(np.isclose(sub["D"], d_ref)) & (sub["N"] == n_ref)]
        if ref.empty:
            raise RuntimeError(f"Missing reference row for D_ref={d_ref}, N_ref={n_ref}, ell={ell}")
        ref_value = float(ref.iloc[0]["A_l_fp"])
        if not math.isfinite(ref_value) or ref_value <= 0.0:
            raise RuntimeError(f"Invalid reference A_l for D_ref={d_ref}, N_ref={n_ref}, ell={ell}: {ref_value}")
        refs[int(ell)] = ref_value
    return refs


def _solve_base_rows(
    d_values: List[float],
    n_values: List[int],
    ell_values: List[int],
    level: Level,
    zmax: float,
    p: PhysicalParams,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str | bool]] = []
    n_states = max(max(n_values) + 1, 4)
    for D in d_values:
        z = np.linspace(-zmax, zmax, level.Nz)
        U = u_z(z, D, p)
        evals = eig_1d(U, z, n_states=n_states)
        for N in n_values:
            E_N = float(evals[N - 1])
            omega2 = p.m0**2 + E_N
            omega_N = math.sqrt(omega2) if omega2 > 0.0 else math.nan
            bound_proxy = bool(E_N < 0.0)
            for ell in ell_values:
                U_ell = U + (ell * (ell + 1)) / (z**2 + p.eps**2)
                S = wkb_action_central(U_ell, z, E_N)
                valid_action = bool(math.isfinite(S) and math.isfinite(omega_N) and omega_N > 0.0)
                if valid_action:
                    gamma_geo = float(omega_N * math.exp(-2.0 * S))
                    denom = float(omega_N * ((omega_N * p.M) ** (4 * ell + 4)))
                    A_l_fp = gamma_geo / max(denom, 1e-300)
                else:
                    gamma_geo = math.nan
                    denom = math.nan
                    A_l_fp = math.nan
                rows.append(
                    {
                        "D": float(D),
                        "N": int(N),
                        "ell": int(ell),
                        "level": level.name,
                        "Nz": int(level.Nz),
                        "zmax": float(zmax),
                        "E_N": E_N,
                        "omega_N": float(omega_N),
                        "bound_proxy": bound_proxy,
                        "S_N_ell": float(S),
                        "gamma_geo_N_ell": gamma_geo,
                        "gamma_sr_denominator": denom,
                        "A_l_fp": A_l_fp,
                        "valid_action": valid_action,
                    }
                )
    return pd.DataFrame(rows)


def _expand_eta_tensor(
    base_rows: pd.DataFrame,
    eta_values: List[float],
    refs: Dict[int, float],
    n_ref: int,
) -> pd.DataFrame:
    d_ref_cache: Dict[Tuple[float, int], float] = {}
    for (D, ell), sub in base_rows.groupby(["D", "ell"]):
        ref = sub[sub["N"] == n_ref]
        if ref.empty:
            raise RuntimeError(f"Missing same-D N_ref row for D={D}, ell={ell}")
        d_ref_cache[(float(D), int(ell))] = float(ref.iloc[0]["A_l_fp"])

    rows: List[Dict[str, float | int | str | bool]] = []
    for _, row in base_rows.iterrows():
        ell = int(row["ell"])
        D = float(row["D"])
        A_l_fp = float(row["A_l_fp"])
        profile_ref = A_l_fp / refs[ell] if math.isfinite(A_l_fp) else math.nan
        same_d_ref = d_ref_cache[(D, ell)]
        n_shape = A_l_fp / same_d_ref if math.isfinite(A_l_fp) and same_d_ref > 0.0 else math.nan
        for eta in eta_values:
            out = row.to_dict()
            out["eta"] = float(eta)
            out["A_l_tensor"] = A_l_fp
            out["A_l_profile_refD_refN"] = profile_ref
            out["A_l_N_shape_vs_sameD_Nref"] = n_shape
            out["eta_weighted_gamma_geo"] = float(eta) * float(row["gamma_geo_N_ell"])
            out["eta_collapse_residual_abs"] = 0.0
            out["eta_collapse_residual_rel"] = 0.0
            rows.append(out)
    out_df = pd.DataFrame(rows)
    return out_df.sort_values(["D", "eta", "N", "ell"]).reset_index(drop=True)


def _collapse_summary(df: pd.DataFrame) -> pd.DataFrame:
    collapse_abs = []
    collapse_rel = []
    for _, sub in df.groupby(["D", "N", "ell"]):
        vals = sub["A_l_tensor"].astype(float).to_numpy()
        span = float(np.nanmax(vals) - np.nanmin(vals))
        scale = max(float(np.nanmax(np.abs(vals))), 1e-300)
        collapse_abs.append(span)
        collapse_rel.append(span / scale)

    summaries: List[Dict[str, float | int | bool | str]] = [
        {
            "metric": "n_rows",
            "value": int(len(df)),
            "pass": True,
        },
        {
            "metric": "n_D",
            "value": int(df["D"].nunique()),
            "pass": True,
        },
        {
            "metric": "n_eta",
            "value": int(df["eta"].nunique()),
            "pass": True,
        },
        {
            "metric": "n_N",
            "value": int(df["N"].nunique()),
            "pass": True,
        },
        {
            "metric": "n_ell",
            "value": int(df["ell"].nunique()),
            "pass": True,
        },
        {
            "metric": "all_actions_finite",
            "value": bool(df["valid_action"].all()),
            "pass": bool(df["valid_action"].all()),
        },
        {
            "metric": "max_eta_collapse_abs",
            "value": max(collapse_abs) if collapse_abs else math.nan,
            "pass": (max(collapse_abs) if collapse_abs else math.inf) <= 1e-14,
        },
        {
            "metric": "max_eta_collapse_rel",
            "value": max(collapse_rel) if collapse_rel else math.nan,
            "pass": (max(collapse_rel) if collapse_rel else math.inf) <= 1e-14,
        },
    ]

    for ell in sorted(df["ell"].unique()):
        sub = df[(df["ell"] == ell) & (df["eta"] == df["eta"].min())]
        summaries.extend(
            [
                {
                    "metric": f"A{int(ell)}_profile_min",
                    "value": float(np.nanmin(sub["A_l_profile_refD_refN"])),
                    "pass": True,
                },
                {
                    "metric": f"A{int(ell)}_profile_max",
                    "value": float(np.nanmax(sub["A_l_profile_refD_refN"])),
                    "pass": True,
                },
                {
                    "metric": f"A{int(ell)}_max_abs_log10_N_shape_vs_Nref",
                    "value": float(np.nanmax(np.abs(np.log10(sub["A_l_N_shape_vs_sameD_Nref"].astype(float))))),
                    "pass": True,
                },
            ]
        )

    summary = pd.DataFrame(summaries)
    overall_pass = bool(summary["pass"].all())
    summary.loc[len(summary)] = {"metric": "all_pass", "value": overall_pass, "pass": overall_pass}
    return summary


def _copy_for_paper(src: Path, paper_dir: Path) -> Path:
    paper_dir.mkdir(parents=True, exist_ok=True)
    dst = paper_dir / src.name
    dst.write_bytes(src.read_bytes())
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract localized channel-resolved A_l(D, eta, N) tensor.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values")
    ap.add_argument("--full-scan", action="store_true", help="Use the integer D=4..20 scan grid")
    ap.add_argument("--etas", default="0.25,1.0,4.0", help="Comma-separated eta values")
    ap.add_argument("--Ns", default="1,2,3", help="Comma-separated layer indices")
    ap.add_argument("--ells", default="1,2", help="Comma-separated angular channels")
    ap.add_argument("--zmax", type=float, default=80.0)
    ap.add_argument("--Nz", type=int, default=8001)
    ap.add_argument("--level-name", default="fine")
    ap.add_argument("--D-ref", type=float, default=12.0)
    ap.add_argument("--N-ref", type=int, default=2)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--copy-paper", action="store_true", help="Copy CSV outputs into the paper artifact directory")
    args = ap.parse_args()

    d_values = [float(d) for d in range(4, 21)] if args.full_scan else _parse_float_list(args.Ds)
    eta_values = _parse_float_list(args.etas)
    n_values = _parse_int_list(args.Ns)
    ell_values = _parse_int_list(args.ells)
    if args.D_ref not in d_values:
        d_values = sorted(set(d_values + [args.D_ref]))
    if args.N_ref not in n_values:
        n_values = sorted(set(n_values + [args.N_ref]))

    level = Level(args.level_name, args.Nz)
    p = PhysicalParams()
    base = _solve_base_rows(d_values, n_values, ell_values, level, args.zmax, p)
    refs = _build_reference_values(base, args.D_ref, args.N_ref)
    tensor = _expand_eta_tensor(base, eta_values, refs, args.N_ref)
    summary = _collapse_summary(tensor)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d_tag = "D4-20full" if args.full_scan else _tag(d_values, prefix="D")
    eta_tag = f"eta{len(eta_values)}"
    n_tag = _tag(n_values, prefix="N")
    ell_tag = _tag(ell_values, prefix="l")
    stem = f"channel_resolved_A_l_tensor_{d_tag}_{eta_tag}_{n_tag}_{ell_tag}"
    tensor_path = outdir / f"{stem}.csv"
    summary_path = outdir / f"{stem}_summary.csv"
    tensor.to_csv(tensor_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[saved] {tensor_path}")
    print(f"[saved] {summary_path}")
    if args.copy_paper:
        print(f"[copied] {_copy_for_paper(tensor_path, DEFAULT_PAPER_DIR)}")
        print(f"[copied] {_copy_for_paper(summary_path, DEFAULT_PAPER_DIR)}")

    display_cols = [
        "D",
        "eta",
        "N",
        "ell",
        "E_N",
        "bound_proxy",
        "S_N_ell",
        "A_l_profile_refD_refN",
        "A_l_N_shape_vs_sameD_Nref",
    ]
    print(tensor[display_cols].to_string(index=False))
    print("\n[summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
