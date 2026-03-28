#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_gn_chi_compatibility_audit import (
    HOTSPOT_BAND,
    KEY_D,
    PROXY_LIBRARY,
    WITNESS_LIBRARY,
    _affine_profile,
    _anchor_ratio,
    _corr,
    _load_witness,
    _shape_distance,
)
from scan_prolate_auxiliary_extraction_audit import (
    OUTDIR,
    PAPER_DIR,
    P,
    Z_MAX,
    N_TOTAL,
    build_uniform_axis_mesh,
    solve_nonuniform_bound_states,
)
from scan_uniform_lorentz_reparam_extraction_audit import _effective_D, _gamma


ROOT = Path(__file__).resolve().parent.parent
STEM = "uniform_lorentz_reparam_gn_chi_compatibility_audit"

METHOD_SPECS = {
    "uniform_aux": None,
    "lorentz_boosted_aux": {"mode": "boosted_frame", "beta": 0.6},
    "lorentz_contract_aux": {"mode": "length_contraction", "beta": 0.4},
}


def _solve_profiles(d_values: np.ndarray) -> pd.DataFrame:
    rows = []
    z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
    for D in d_values:
        e_u, om_u = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)
        row = {
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
        }

        for method, spec in METHOD_SPECS.items():
            if spec is None:
                continue
            d_eff = _effective_D(float(D), mode=str(spec["mode"]), beta=float(spec["beta"]))
            evals, omega = solve_nonuniform_bound_states(z_uniform, d_eff, P, n_states=3)
            prefix = method.replace("_aux", "")
            row.update(
                {
                    f"{prefix}_mode": str(spec["mode"]),
                    f"{prefix}_beta": float(spec["beta"]),
                    f"{prefix}_gamma": _gamma(float(spec["beta"])),
                    f"{prefix}_D_eff": float(d_eff),
                    f"{prefix}_E1": float(evals[0]),
                    f"{prefix}_E2": float(evals[1]),
                    f"{prefix}_E3": float(evals[2]),
                    f"{prefix}_omega1": float(omega[0]),
                    f"{prefix}_omega2": float(omega[1]),
                    f"{prefix}_omega3": float(omega[2]),
                    f"{prefix}_delta_omega12": float(omega[1] - omega[0]),
                    f"{prefix}_inv_delta_omega12": float(1.0 / max(abs(omega[1] - omega[0]), 1e-30)),
                    f"{prefix}_gap_fraction": float((omega[1] - omega[0]) / max(abs(omega[1]), 1e-30)),
                    f"{prefix}_mean_omega12": float(0.5 * (omega[0] + omega[1])),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _compatibility_rows(detail: pd.DataFrame) -> pd.DataFrame:
    hotspot = detail["D"].between(*HOTSPOT_BAND).to_numpy(dtype=bool)
    rows = []
    for witness in WITNESS_LIBRARY:
        witness_values = detail[witness].to_numpy(dtype=float)
        witness_anchor = _anchor_ratio(witness_values)
        witness_affine = _affine_profile(witness_values)
        for method in METHOD_SPECS.keys():
            prefix = method.replace("_aux", "")
            for proxy in PROXY_LIBRARY:
                col = f"{prefix}_{proxy}"
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
                        "method": method,
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
        best_rows = {
            method: sub.loc[sub["method"] == method]
            .sort_values(["anchor_l1", "affine_l1", "hotspot_anchor_l1"])
            .iloc[0]
            for method in METHOD_SPECS.keys()
        }
        rows.append(
            {
                "summary_type": "best_proxy_by_witness",
                "witness": witness,
                "uniform_best_proxy": str(best_rows["uniform_aux"]["proxy"]),
                "boosted_best_proxy": str(best_rows["lorentz_boosted_aux"]["proxy"]),
                "contract_best_proxy": str(best_rows["lorentz_contract_aux"]["proxy"]),
                "uniform_anchor_l1": float(best_rows["uniform_aux"]["anchor_l1"]),
                "boosted_anchor_l1": float(best_rows["lorentz_boosted_aux"]["anchor_l1"]),
                "contract_anchor_l1": float(best_rows["lorentz_contract_aux"]["anchor_l1"]),
                "uniform_affine_l1": float(best_rows["uniform_aux"]["affine_l1"]),
                "boosted_affine_l1": float(best_rows["lorentz_boosted_aux"]["affine_l1"]),
                "contract_affine_l1": float(best_rows["lorentz_contract_aux"]["affine_l1"]),
                "uniform_hotspot_anchor_l1": float(best_rows["uniform_aux"]["hotspot_anchor_l1"]),
                "boosted_hotspot_anchor_l1": float(best_rows["lorentz_boosted_aux"]["hotspot_anchor_l1"]),
                "contract_hotspot_anchor_l1": float(best_rows["lorentz_contract_aux"]["hotspot_anchor_l1"]),
                "best_method_anchor": min(
                    ("uniform_aux", float(best_rows["uniform_aux"]["anchor_l1"])),
                    ("lorentz_boosted_aux", float(best_rows["lorentz_boosted_aux"]["anchor_l1"])),
                    ("lorentz_contract_aux", float(best_rows["lorentz_contract_aux"]["anchor_l1"])),
                    key=lambda x: x[1],
                )[0],
                "best_method_affine": min(
                    ("uniform_aux", float(best_rows["uniform_aux"]["affine_l1"])),
                    ("lorentz_boosted_aux", float(best_rows["lorentz_boosted_aux"]["affine_l1"])),
                    ("lorentz_contract_aux", float(best_rows["lorentz_contract_aux"]["affine_l1"])),
                    key=lambda x: x[1],
                )[0],
            }
        )

    for method in METHOD_SPECS.keys():
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
                "best_proxy_by_mean_anchor": str(grouped.sort_values(["anchor_l1", "affine_l1"]).iloc[0]["proxy"]),
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
    proxy_map = {(str(r["witness"]), str(r["method"])): str(r["proxy"]) for _, r in best.iterrows()}
    rows = []
    for d in KEY_D:
        sub = detail.loc[np.isclose(detail["D"], d)].iloc[0]
        row = {"D": float(d)}
        for witness in ["chi_LR", "Gamma_ref", "g2_raw", "g3_raw"]:
            row[witness] = float(sub[witness])
            for method in METHOD_SPECS.keys():
                proxy = proxy_map[(witness, method)]
                prefix = method.replace("_aux", "")
                row[f"{method}_best_proxy_for_{witness}"] = proxy
                row[f"{method}_{witness}_proxy_value"] = float(sub[f"{prefix}_{proxy}"])
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
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), constrained_layout=True)

    colors = {
        "uniform_aux": "#4e79a7",
        "lorentz_boosted_aux": "#e15759",
        "lorentz_contract_aux": "#59a14f",
    }

    for ax, witness in zip(axes.flat, ["chi_LR", "Gamma_ref", "g2_raw", "g3_raw"]):
        y = _anchor_ratio(detail[witness].to_numpy(dtype=float))
        ax.plot(x, y, lw=2.2, color="#222222", label=f"{witness} witness")
        for method in METHOD_SPECS.keys():
            prefix = method.replace("_aux", "")
            proxy = proxy_map[(witness, method)]
            ym = _anchor_ratio(detail[f"{prefix}_{proxy}"].to_numpy(dtype=float))
            ax.plot(x, ym, lw=2, color=colors[method], label=f"{method}:{proxy}")
        for d in KEY_D:
            ax.axvline(d, color="0.82", ls="--", lw=1)
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
    meta_path.write_text(
        json.dumps(
            {
                "method_specs": METHOD_SPECS,
                "witness_paths": {
                    "chi": str(ROOT / "output" / "chi_fp_2d" / "localized_chi_Dgrid60_fine.csv"),
                    "gn": str(ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_Dgrid60.csv"),
                },
                "notes": (
                    "Compatibility audit for the two best Lorentz-like D reparameterization candidates discovered in the "
                    "uniform extraction audit. This remains an extraction-side numerical ansatz test."
                ),
            },
            indent=2,
        )
    )

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
