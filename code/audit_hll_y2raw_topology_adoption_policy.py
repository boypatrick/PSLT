#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicit topology-changing adoption certificate for the U10 direct "
            "y_2^{raw}(D_*) scalar.  This does not reopen EYMH normalization: it "
            "checks whether the known D21xE21 topology change is a monotone, "
            "non-fragmenting acceptance expansion under the fixed bridge."
        )
    )
    parser.add_argument(
        "--impact-detail",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_adoption_impact_gate_detail.csv"
        ),
    )
    parser.add_argument(
        "--impact-summary",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_adoption_impact_gate_summary.csv"
        ),
    )
    parser.add_argument(
        "--continuation-detail",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_projector_continuation_detail.csv"
        ),
    )
    parser.add_argument(
        "--continuation-summary",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_projector_continuation_summary.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "hll_absolute_normalization"),
    )
    parser.add_argument("--copy-paper", action="store_true")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def fmt_values(values: Iterable[float]) -> str:
    return ";".join(f"{float(v):.12g}" for v in values)


def component_count(mask: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []
    n_d, n_eta = mask.shape
    for i in range(n_d):
        for j in range(n_eta):
            if visited[i, j] or not mask[i, j]:
                continue
            queue: deque[tuple[int, int]] = deque([(i, j)])
            visited[i, j] = True
            cells: list[tuple[int, int]] = []
            while queue:
                ci, cj = queue.popleft()
                cells.append((ci, cj))
                for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                    if 0 <= ni < n_d and 0 <= nj < n_eta and mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))
            components.append(
                {
                    "size": len(cells),
                    "min_d_index": min(i0 for i0, _ in cells),
                    "max_d_index": max(i0 for i0, _ in cells),
                    "min_eta_index": min(j0 for _, j0 in cells),
                    "max_eta_index": max(j0 for _, j0 in cells),
                }
            )
    return len(components), components


def main() -> None:
    args = parse_args()
    impact = pd.read_csv(args.impact_detail).sort_values(["D", "eta"]).reset_index(drop=True)
    impact_summary = pd.read_csv(args.impact_summary)
    continuation = pd.read_csv(args.continuation_detail).sort_values(["D", "eta"]).reset_index(drop=True)
    continuation_summary = pd.read_csv(args.continuation_summary)

    for col in ("accept_release", "accept_direct_y2", "acceptance_gain", "acceptance_loss"):
        impact[col] = bool_series(impact[col])

    merged = impact.merge(
        continuation[
            [
                "D",
                "eta",
                "crossing_type",
                "s_cross",
                "y2_cross_over_release",
                "amp_cross_over_release",
                "abs_mu_cross_validation_error",
            ]
        ],
        on=["D", "eta"],
        how="left",
        validate="one_to_one",
    )

    d_values = np.array(sorted(merged["D"].unique()), dtype=float)
    eta_values = np.array(sorted(merged["eta"].unique()), dtype=float)
    d_index = {float(v): idx for idx, v in enumerate(d_values)}
    eta_index = {float(v): idx for idx, v in enumerate(eta_values)}
    release_mask = np.zeros((len(d_values), len(eta_values)), dtype=bool)
    direct_mask = np.zeros_like(release_mask)
    gain_mask = np.zeros_like(release_mask)

    for row in merged.itertuples(index=False):
        i = d_index[float(row.D)]
        j = eta_index[float(row.eta)]
        release_mask[i, j] = bool(row.accept_release)
        direct_mask[i, j] = bool(row.accept_direct_y2)
        gain_mask[i, j] = bool(row.acceptance_gain)

    release_components, release_component_data = component_count(release_mask)
    direct_components, direct_component_data = component_count(direct_mask)
    gain_components, gain_component_data = component_count(gain_mask)

    accepted_release = merged[merged["accept_release"]]
    accepted_direct = merged[merged["accept_direct_y2"]]
    gains = merged[merged["acceptance_gain"]]
    losses = merged[merged["acceptance_loss"]]
    crossing = merged[merged["crossing_type"].fillna("none") != "none"]
    upper_crossing = merged[merged["crossing_type"].fillna("none") == "upper_gain"]
    lower_crossing = merged[merged["crossing_type"].fillna("none") == "lower_loss"]

    release_subset_direct = bool(np.all(~impact["accept_release"] | impact["accept_direct_y2"]))
    direct_has_no_loss = int(len(losses)) == 0
    no_new_fragment = direct_components <= max(release_components, 1)
    one_component_both = release_components == 1 and direct_components == 1
    gain_attaches_to_old_component = one_component_both and gain_components == 1

    eta_count = len(eta_values)
    release_by_d = accepted_release.groupby("D", as_index=False).agg(n_eta=("eta", "size"))
    direct_by_d = accepted_direct.groupby("D", as_index=False).agg(n_eta=("eta", "size"))
    gains_by_d = gains.groupby("D", as_index=False).agg(n_eta=("eta", "size"))
    release_full_slab_d = release_by_d.loc[release_by_d["n_eta"] == eta_count, "D"].to_list()
    direct_full_slab_d = direct_by_d.loc[direct_by_d["n_eta"] == eta_count, "D"].to_list()
    gain_full_slab_d = gains_by_d.loc[gains_by_d["n_eta"] == eta_count, "D"].to_list()
    slab_extension = (
        len(gains) > 0
        and len(gain_full_slab_d) == len(gains_by_d)
        and set(release_full_slab_d).issubset(set(direct_full_slab_d))
    )

    policy_pass = all(
        [
            release_subset_direct,
            direct_has_no_loss,
            no_new_fragment,
            gain_attaches_to_old_component,
            slab_extension,
        ]
    )
    verdict = (
        "adoption_safe_under_explicit_monotone_topology_expansion"
        if policy_pass
        else "not_adoption_safe_under_explicit_monotone_topology_expansion"
    )

    direct_summary = impact_summary.loc[
        impact_summary["case"] == "direct_inserted_history_y2"
    ].iloc[0]
    release_summary = impact_summary.loc[
        impact_summary["case"] == "release_interpolated_y2"
    ].iloc[0]
    crossing_summary = continuation_summary.loc[
        continuation_summary["section"] == "topology_crossing_certificate"
    ].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "section": "explicit_topology_policy",
                "policy": "monotone_acceptance_enlargement",
                "verdict": verdict,
                "release_subset_direct": release_subset_direct,
                "n_accept_release": int(len(accepted_release)),
                "n_accept_direct": int(len(accepted_direct)),
                "n_acceptance_gains": int(len(gains)),
                "n_acceptance_losses": int(len(losses)),
                "acceptance_fraction_release": float(release_summary["f_hmumu_chi2_le_4"]),
                "acceptance_fraction_direct": float(direct_summary["f_hmumu_chi2_le_4"]),
                "delta_acceptance_fraction": float(
                    direct_summary["f_hmumu_chi2_le_4"]
                    - release_summary["f_hmumu_chi2_le_4"]
                ),
                "accepted_component_count_release": release_components,
                "accepted_component_count_direct": direct_components,
                "gain_component_count": gain_components,
                "release_full_slab_D_values": fmt_values(release_full_slab_d),
                "direct_full_slab_D_values": fmt_values(direct_full_slab_d),
                "gain_full_slab_D_values": fmt_values(gain_full_slab_d),
                "gain_eta_min": float(gains["eta"].min()) if len(gains) else np.nan,
                "gain_eta_max": float(gains["eta"].max()) if len(gains) else np.nan,
                "s_cross_min": float(upper_crossing["s_cross"].min()) if len(upper_crossing) else np.nan,
                "s_cross_max": float(upper_crossing["s_cross"].max()) if len(upper_crossing) else np.nan,
                "n_upper_gain_crossings": int(len(upper_crossing)),
                "n_lower_loss_crossings": int(len(lower_crossing)),
                "max_abs_mu_cross_validation_error": float(
                    crossing["abs_mu_cross_validation_error"].max()
                )
                if len(crossing)
                else 0.0,
                "best_chi2_release": float(release_summary["best_chi2"]),
                "best_chi2_direct": float(direct_summary["best_chi2"]),
                "best_mu_release": float(release_summary["best_mu_mumu"]),
                "best_mu_direct": float(direct_summary["best_mu_mumu"]),
                "y2_over_release_direct": float(direct_summary["y2_over_release"]),
                "amp_over_release_direct": float(direct_summary["amp_over_release"]),
            }
        ]
    )

    component_summary = pd.DataFrame(
        [
            {
                "set_name": "release_acceptance",
                "component_count": release_components,
                "components_json": json.dumps(release_component_data, sort_keys=True),
            },
            {
                "set_name": "direct_acceptance",
                "component_count": direct_components,
                "components_json": json.dumps(direct_component_data, sort_keys=True),
            },
            {
                "set_name": "direct_minus_release_gains",
                "component_count": gain_components,
                "components_json": json.dumps(gain_component_data, sort_keys=True),
            },
        ]
    )

    detail = merged[
        [
            "D",
            "eta",
            "mu_mumu_release_bridge",
            "mu_mumu_direct_y2",
            "chi2_release_bridge",
            "chi2_direct_y2",
            "accept_release",
            "accept_direct_y2",
            "acceptance_gain",
            "acceptance_loss",
            "crossing_type",
            "s_cross",
            "y2_cross_over_release",
            "amp_cross_over_release",
            "abs_mu_cross_validation_error",
        ]
    ].copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_topology_adoption_policy_summary.csv"
    detail_out = outdir / "hll_y2raw_topology_adoption_policy_detail.csv"
    components_out = outdir / "hll_y2raw_topology_adoption_policy_components.csv"
    meta_out = outdir / "hll_y2raw_topology_adoption_policy_run_meta.json"
    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    component_summary.to_csv(components_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "impact_detail": str(args.impact_detail),
                "impact_summary": str(args.impact_summary),
                "continuation_detail": str(args.continuation_detail),
                "continuation_summary": str(args.continuation_summary),
                "policy": "monotone acceptance enlargement: release accepted set is contained in direct accepted set, no losses, no added disconnected accepted component, gain slab attaches to the old component",
                "verdict": verdict,
            },
            indent=2,
        )
    )

    if args.copy_paper:
        paper_dir = ROOT / "paper"
        for path in (summary_out, detail_out, components_out, meta_out):
            shutil.copy2(path, paper_dir / path.name)

    print(summary_out)
    print(detail_out)
    print(components_out)
    print(meta_out)
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()
