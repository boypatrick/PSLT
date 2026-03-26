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
    _affine_profile,
    _anchor_ratio,
    _load_witness,
    _shape_distance,
    _solve_profiles,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "prolate_hybrid_gn_chi_propagation_audit"


FAMILY_PROXY_MAP = {
    "uniform_family": {
        "chi_LR": ("uniform", "delta_omega12"),
        "Gamma_ref": ("uniform", "omega1"),
        "g2_raw": ("uniform", "omega3"),
        "g3_raw": ("uniform", "omega3"),
    },
    "prolate_family": {
        "chi_LR": ("prolate", "gap_fraction"),
        "Gamma_ref": ("prolate", "omega1"),
        "g2_raw": ("prolate", "omega2"),
        "g3_raw": ("prolate", "omega2"),
    },
    "hybrid_family": {
        "chi_LR": ("prolate", "gap_fraction"),
        "Gamma_ref": ("uniform", "omega1"),
        "g2_raw": ("prolate", "omega2"),
        "g3_raw": ("prolate", "omega2"),
    },
}

WITNESSES = ["chi_LR", "Gamma_ref", "g2_raw", "g3_raw"]


def _proxy_col(method: str, proxy: str) -> str:
    return f"{method}_{proxy}"


def _family_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hotspot = detail["D"].between(*HOTSPOT_BAND).to_numpy(dtype=bool)
    for family, mapping in FAMILY_PROXY_MAP.items():
        for witness in WITNESSES:
            method, proxy = mapping[witness]
            proxy_col = _proxy_col(method, proxy)
            witness_values = detail[witness].to_numpy(dtype=float)
            proxy_values = detail[proxy_col].to_numpy(dtype=float)

            witness_anchor = _anchor_ratio(witness_values)
            proxy_anchor = _anchor_ratio(proxy_values)
            witness_affine = _affine_profile(witness_values)
            proxy_affine = _affine_profile(proxy_values)

            anchor_stats = _shape_distance(proxy_anchor, witness_anchor)
            affine_stats = _shape_distance(proxy_affine, witness_affine)
            hotspot_anchor = _shape_distance(proxy_anchor[hotspot], witness_anchor[hotspot])
            hotspot_affine = _shape_distance(proxy_affine[hotspot], witness_affine[hotspot])

            rows.append(
                {
                    "family": family,
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
                }
            )
    return pd.DataFrame(rows)


def _summary(detail: pd.DataFrame, family_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, grp in family_detail.groupby("family", sort=False):
        rows.append(
            {
                "summary_type": "family_score",
                "family": family,
                "mean_anchor_l1": float(grp["anchor_l1"].mean()),
                "mean_affine_l1": float(grp["affine_l1"].mean()),
                "mean_hotspot_anchor_l1": float(grp["hotspot_anchor_l1"].mean()),
                "mean_hotspot_affine_l1": float(grp["hotspot_affine_l1"].mean()),
                "max_anchor_l1": float(grp["anchor_l1"].max()),
                "max_affine_l1": float(grp["affine_l1"].max()),
                "max_hotspot_anchor_l1": float(grp["hotspot_anchor_l1"].max()),
            }
        )

    pivot = family_detail.pivot(index="witness", columns="family", values="anchor_l1")
    for witness in pivot.index:
        u = float(pivot.loc[witness, "uniform_family"])
        p = float(pivot.loc[witness, "prolate_family"])
        h = float(pivot.loc[witness, "hybrid_family"])
        rows.append(
            {
                "summary_type": "witness_compare",
                "witness": witness,
                "uniform_anchor_l1": u,
                "prolate_anchor_l1": p,
                "hybrid_anchor_l1": h,
                "hybrid_over_uniform_ratio": float(h / max(u, 1e-30)),
                "hybrid_over_prolate_ratio": float(h / max(p, 1e-30)),
                "best_family_by_anchor": min(
                    [("uniform_family", u), ("prolate_family", p), ("hybrid_family", h)],
                    key=lambda kv: kv[1],
                )[0],
            }
        )
    return pd.DataFrame(rows)


def _slices(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in KEY_D:
        sub = detail.loc[np.isclose(detail["D"], d)].iloc[0]
        row = {"D": float(d)}
        for witness in WITNESSES:
            row[witness] = float(sub[witness])
            for family, mapping in FAMILY_PROXY_MAP.items():
                method, proxy = mapping[witness]
                row[f"{family}_{witness}_method"] = method
                row[f"{family}_{witness}_proxy"] = proxy
                row[f"{family}_{witness}_value"] = float(sub[_proxy_col(method, proxy)])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(detail: pd.DataFrame, out_png: Path) -> None:
    x = detail["D"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    for ax, witness in zip(axes.flat, WITNESSES):
        yw = _anchor_ratio(detail[witness].to_numpy(dtype=float))
        ax.plot(x, yw, lw=2, label=f"{witness} witness")
        for family, mapping in FAMILY_PROXY_MAP.items():
            method, proxy = mapping[witness]
            yp = _anchor_ratio(detail[_proxy_col(method, proxy)].to_numpy(dtype=float))
            ax.plot(x, yp, lw=2, label=f"{family}:{method}/{proxy}")
        ax.axvspan(HOTSPOT_BAND[0], HOTSPOT_BAND[1], color="#f3e7c7", alpha=0.35)
        for d in KEY_D:
            ax.axvline(d, color="0.8", ls="--", lw=1)
        ax.set_title(f"{witness} hybrid compatibility")
        ax.set_xlabel("D")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    witness = _load_witness()
    profiles = _solve_profiles(witness["D"].to_numpy(dtype=float))
    detail = witness.merge(profiles, on="D", how="inner").sort_values("D").reset_index(drop=True)
    family_detail = _family_detail(detail)
    summary = _summary(detail, family_detail)
    slices = _slices(detail)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(detail, png_path)
    meta = {
        "families": FAMILY_PROXY_MAP,
        "hotspot_band": list(HOTSPOT_BAND),
        "notes": (
            "Hybrid audit keeps Gamma_ref on uniform auxiliary proxy while routing chi_LR and raw g witnesses "
            "through calibrated prolate-compatible proxies. This remains extraction-side only."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
