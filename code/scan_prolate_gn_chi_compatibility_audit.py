#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_auxiliary_extraction_audit import (
    OUTDIR as KIN_OUTDIR,
    PAPER_DIR,
    P,
    Z_MAX,
    N_TOTAL,
    PROLATE_MESH_PROFILES,
    build_parametrized_prolate_axis_mesh,
    build_uniform_axis_mesh,
    solve_nonuniform_bound_states,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
STEM = "prolate_gn_chi_compatibility_audit"

CHI_PATH = ROOT / "output" / "chi_fp_2d" / "localized_chi_Dgrid60_fine.csv"
GN_PATH = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_Dgrid60.csv"

HOTSPOT_BAND = (5.6, 6.5)
KEY_D = (5.627118644067797, 6.169492525423729, 6.4406779661016955, 6.711864406779661)

CALIBRATED_PROFILE = PROLATE_MESH_PROFILES["calibrated"]


def _anchor_ratio(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    base = float(arr[0])
    return arr / max(abs(base), 1e-30)


def _affine_profile(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 1e-30:
        return np.zeros_like(arr)
    return (arr - lo) / span


def _shape_distance(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    delta = lhs - rhs
    return {
        "l1": float(np.sum(np.abs(delta))),
        "linf": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def _corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _load_witness() -> pd.DataFrame:
    chi = pd.read_csv(CHI_PATH)
    if "level" in chi.columns:
        chi = chi.loc[chi["level"].astype(str).str.lower() == "fine"].copy()
    chi = chi[
        [
            "D",
            "E1",
            "E2",
            "DeltaE",
            "chi_LR",
            "omega1",
            "Gamma_ref",
        ]
    ].rename(
        columns={
            "E1": "chi_E1",
            "E2": "chi_E2",
            "DeltaE": "chi_DeltaE",
            "omega1": "chi_omega1",
        }
    )

    gn = pd.read_csv(GN_PATH)
    if "level" in gn.columns:
        gn = gn.loc[gn["level"].astype(str).str.lower() == "fine"].copy()
    gn = gn[
        [
            "D",
            "lambda1",
            "lambda2",
            "lambda3",
            "g1_raw",
            "g2_raw",
            "g3_raw",
            "g1_hat",
            "g2_hat",
            "g3_hat",
        ]
    ]
    return chi.merge(gn, on="D", how="inner").sort_values("D").reset_index(drop=True)


def _solve_profiles(d_values: np.ndarray) -> pd.DataFrame:
    rows = []
    n_inner = int(CALIBRATED_PROFILE["n_inner"])
    eta_power = float(CALIBRATED_PROFILE["eta_power"])
    xi_power = float(CALIBRATED_PROFILE["xi_power"])
    for D in d_values:
        z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
        e_u, om_u = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)

        z_pro = build_parametrized_prolate_axis_mesh(
            float(D),
            Z_MAX,
            N_TOTAL,
            n_inner,
            eta_power,
            xi_power,
        )
        e_p, om_p = solve_nonuniform_bound_states(z_pro, float(D), P, n_states=3)

        rows.append(
            {
                "D": float(D),
                "uniform_E1": float(e_u[0]),
                "uniform_E2": float(e_u[1]),
                "uniform_E3": float(e_u[2]),
                "uniform_omega1": float(om_u[0]),
                "uniform_omega2": float(om_u[1]),
                "uniform_omega3": float(om_u[2]),
                "uniform_delta_omega12": float(om_u[1] - om_u[0]),
                "uniform_inv_delta_omega12": float(1.0 / max(abs(om_u[1] - om_u[0]), 1e-30)),
                "uniform_gap_fraction": float((om_u[1] - om_u[0]) / max(abs(om_u[1]), 1e-30)),
                "uniform_mean_omega12": float(0.5 * (om_u[0] + om_u[1])),
                "prolate_E1": float(e_p[0]),
                "prolate_E2": float(e_p[1]),
                "prolate_E3": float(e_p[2]),
                "prolate_omega1": float(om_p[0]),
                "prolate_omega2": float(om_p[1]),
                "prolate_omega3": float(om_p[2]),
                "prolate_delta_omega12": float(om_p[1] - om_p[0]),
                "prolate_inv_delta_omega12": float(1.0 / max(abs(om_p[1] - om_p[0]), 1e-30)),
                "prolate_gap_fraction": float((om_p[1] - om_p[0]) / max(abs(om_p[1]), 1e-30)),
                "prolate_mean_omega12": float(0.5 * (om_p[0] + om_p[1])),
            }
        )
    return pd.DataFrame(rows)


PROXY_LIBRARY = [
    "omega1",
    "omega2",
    "omega3",
    "delta_omega12",
    "inv_delta_omega12",
    "gap_fraction",
    "mean_omega12",
]

WITNESS_LIBRARY = [
    "chi_LR",
    "Gamma_ref",
    "chi_omega1",
    "chi_DeltaE",
    "lambda1",
    "lambda2",
    "lambda3",
    "g2_raw",
    "g3_raw",
    "g2_hat",
    "g3_hat",
]


def _compatibility_rows(detail: pd.DataFrame) -> pd.DataFrame:
    hotspot = detail["D"].between(*HOTSPOT_BAND).to_numpy(dtype=bool)
    rows = []
    for witness in WITNESS_LIBRARY:
        witness_values = detail[witness].to_numpy(dtype=float)
        witness_anchor = _anchor_ratio(witness_values)
        witness_affine = _affine_profile(witness_values)
        for method in ["uniform", "prolate"]:
            for proxy in PROXY_LIBRARY:
                col = f"{method}_{proxy}"
                proxy_values = detail[col].to_numpy(dtype=float)
                proxy_anchor = _anchor_ratio(proxy_values)
                proxy_affine = _affine_profile(proxy_values)
                anchor_stats = _shape_distance(proxy_anchor, witness_anchor)
                affine_stats = _shape_distance(proxy_affine, witness_affine)
                hotspot_anchor = _shape_distance(proxy_anchor[hotspot], witness_anchor[hotspot])
                hotspot_affine = _shape_distance(proxy_affine[hotspot], witness_affine[hotspot])
                rows.append(
                    {
                        "witness": witness,
                        "method": f"{method}_aux",
                        "proxy": proxy,
                        "anchor_l1": anchor_stats["l1"],
                        "anchor_linf": anchor_stats["linf"],
                        "anchor_rmse": anchor_stats["rmse"],
                        "affine_l1": affine_stats["l1"],
                        "affine_linf": affine_stats["linf"],
                        "affine_rmse": affine_stats["rmse"],
                        "hotspot_anchor_l1": hotspot_anchor["l1"],
                        "hotspot_affine_l1": hotspot_affine["l1"],
                        "corr_raw": _corr(proxy_values, witness_values),
                        "corr_anchor": _corr(proxy_anchor, witness_anchor),
                        "corr_affine": _corr(proxy_affine, witness_affine),
                    }
                )
    return pd.DataFrame(rows)


def _best_summary(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for witness in WITNESS_LIBRARY:
        sub = comp.loc[comp["witness"] == witness].copy()
        best_uniform = sub.loc[sub["method"] == "uniform_aux"].sort_values(
            ["anchor_l1", "affine_l1", "hotspot_anchor_l1"]
        ).iloc[0]
        best_prolate = sub.loc[sub["method"] == "prolate_aux"].sort_values(
            ["anchor_l1", "affine_l1", "hotspot_anchor_l1"]
        ).iloc[0]
        rows.append(
            {
                "summary_type": "best_proxy_by_witness",
                "witness": witness,
                "uniform_best_proxy": str(best_uniform["proxy"]),
                "prolate_best_proxy": str(best_prolate["proxy"]),
                "uniform_anchor_l1": float(best_uniform["anchor_l1"]),
                "prolate_anchor_l1": float(best_prolate["anchor_l1"]),
                "uniform_affine_l1": float(best_uniform["affine_l1"]),
                "prolate_affine_l1": float(best_prolate["affine_l1"]),
                "uniform_hotspot_anchor_l1": float(best_uniform["hotspot_anchor_l1"]),
                "prolate_hotspot_anchor_l1": float(best_prolate["hotspot_anchor_l1"]),
                "prolate_over_uniform_anchor_ratio": float(
                    best_prolate["anchor_l1"] / max(best_uniform["anchor_l1"], 1e-30)
                ),
                "prolate_over_uniform_affine_ratio": float(
                    best_prolate["affine_l1"] / max(best_uniform["affine_l1"], 1e-30)
                ),
                "better_method_anchor": "prolate_aux"
                if float(best_prolate["anchor_l1"]) < float(best_uniform["anchor_l1"])
                else "uniform_aux",
                "better_method_affine": "prolate_aux"
                if float(best_prolate["affine_l1"]) < float(best_uniform["affine_l1"])
                else "uniform_aux",
            }
        )

    for method in ["uniform_aux", "prolate_aux"]:
        sub = comp.loc[comp["method"] == method].copy()
        grouped = (
            sub.sort_values(["anchor_l1", "affine_l1", "hotspot_anchor_l1"])
            .groupby("proxy", as_index=False)
            .first()
        )
        rows.append(
            {
                "summary_type": "proxy_global_rank",
                "method": method,
                "best_proxy_by_mean_anchor": str(
                    grouped.sort_values(["anchor_l1", "affine_l1"]).iloc[0]["proxy"]
                ),
                "mean_anchor_l1": float(grouped["anchor_l1"].mean()),
                "mean_affine_l1": float(grouped["affine_l1"].mean()),
                "mean_hotspot_anchor_l1": float(grouped["hotspot_anchor_l1"].mean()),
                "max_anchor_l1": float(grouped["anchor_l1"].max()),
                "max_affine_l1": float(grouped["affine_l1"].max()),
            }
        )
    return pd.DataFrame(rows)


def _build_slices(detail: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    best = (
        comp.sort_values(["witness", "method", "anchor_l1", "affine_l1", "hotspot_anchor_l1"])
        .groupby(["witness", "method"], as_index=False)
        .first()
    )
    proxy_map: dict[tuple[str, str], str] = {
        (str(row["witness"]), str(row["method"])): str(row["proxy"]) for _, row in best.iterrows()
    }
    rows = []
    for d in KEY_D:
        sub = detail.loc[np.isclose(detail["D"], d)].iloc[0]
        row = {"D": float(d)}
        for witness in ["chi_LR", "Gamma_ref", "g2_raw", "g3_raw"]:
            row[witness] = float(sub[witness])
            up = proxy_map[(witness, "uniform_aux")]
            pp = proxy_map[(witness, "prolate_aux")]
            row[f"uniform_best_proxy_for_{witness}"] = up
            row[f"prolate_best_proxy_for_{witness}"] = pp
            row[f"uniform_{witness}_proxy_value"] = float(sub[f"uniform_{up}"])
            row[f"prolate_{witness}_proxy_value"] = float(sub[f"prolate_{pp}"])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(detail: pd.DataFrame, comp: pd.DataFrame, out_png: Path) -> None:
    best = (
        comp.sort_values(["witness", "method", "anchor_l1", "affine_l1", "hotspot_anchor_l1"])
        .groupby(["witness", "method"], as_index=False)
        .first()
    )
    proxy_map = {(str(r["witness"]), str(r["method"])): str(r["proxy"]) for _, r in best.iterrows()}
    x = detail["D"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)

    for ax, witness in zip(axes.flat, ["chi_LR", "Gamma_ref", "g2_raw", "g3_raw"]):
        y = _anchor_ratio(detail[witness].to_numpy(dtype=float))
        up = proxy_map[(witness, "uniform_aux")]
        pp = proxy_map[(witness, "prolate_aux")]
        yu = _anchor_ratio(detail[f"uniform_{up}"].to_numpy(dtype=float))
        yp = _anchor_ratio(detail[f"prolate_{pp}"].to_numpy(dtype=float))
        ax.plot(x, y, lw=2, label=f"{witness} witness")
        ax.plot(x, yu, lw=2, label=f"uniform:{up}")
        ax.plot(x, yp, lw=2, label=f"prolate:{pp}")
        for d in KEY_D:
            ax.axvline(d, color="0.8", ls="--", lw=1)
        ax.axvspan(HOTSPOT_BAND[0], HOTSPOT_BAND[1], color="#f3e7c7", alpha=0.35)
        ax.set_title(f"{witness} anchor-ratio compatibility")
        ax.set_xlabel("D")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    witness = _load_witness()
    d_values = witness["D"].to_numpy(dtype=float)
    profiles = _solve_profiles(d_values)
    detail = witness.merge(profiles, on="D", how="inner").sort_values("D").reset_index(drop=True)
    comp = _compatibility_rows(detail)
    summary = _best_summary(comp)
    slices = _build_slices(detail, comp)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(detail, comp, png_path)
    meta = {
        "calibrated_profile": CALIBRATED_PROFILE,
        "witness_path": str(CHI_PATH),
        "gn_path": str(GN_PATH),
        "n_points": int(len(detail)),
        "hotspot_band": list(HOTSPOT_BAND),
        "notes": (
            "Compatibility audit compares calibrated prolate auxiliary extraction against uniform auxiliary "
            "extraction using chi/g witnesses as trend references; witnesses are not the same eigenproblem."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        target = PAPER_DIR / path.name
        target.write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
