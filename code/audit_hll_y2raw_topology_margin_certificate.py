#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Margin-robustness certificate for the U13 explicit topology-changing "
            "direct-y2 adoption branch.  The audit does not change the EYMH bridge; "
            "it quantifies how far the release and direct accepted sets are from "
            "the H->mumu 2-sigma acceptance boundary."
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
        "--continuation-summary",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_projector_continuation_summary.csv"
        ),
    )
    parser.add_argument(
        "--topology-summary",
        default=str(
            ROOT
            / "output"
            / "hll_absolute_normalization"
            / "hll_y2raw_topology_adoption_policy_summary.csv"
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


def distance_inside_interval(mu: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.Series(np.minimum(mu - lower, upper - mu), index=mu.index)


def distance_outside_interval(mu: pd.Series, lower: float, upper: float) -> pd.Series:
    dist = np.where(mu < lower, lower - mu, np.where(mu > upper, mu - upper, 0.0))
    return pd.Series(dist, index=mu.index)


def summarize_branch(
    *,
    name: str,
    df: pd.DataFrame,
    mu_col: str,
    accept_col: str,
    lower: float,
    upper: float,
) -> dict[str, object]:
    accepted = df[df[accept_col]]
    rejected = df[~df[accept_col]]
    accepted_margin = distance_inside_interval(accepted[mu_col], lower, upper)
    rejected_margin = distance_outside_interval(rejected[mu_col], lower, upper)
    accepted_worst = accepted.loc[accepted_margin.idxmin()]
    rejected_worst = rejected.loc[rejected_margin.idxmin()]
    set_radius = min(float(accepted_margin.min()), float(rejected_margin.min()))
    return {
        "branch": name,
        "n_accept": int(len(accepted)),
        "n_reject": int(len(rejected)),
        "accepted_margin_min": float(accepted_margin.min()),
        "accepted_margin_worst_D": float(accepted_worst["D"]),
        "accepted_margin_worst_eta": float(accepted_worst["eta"]),
        "accepted_margin_worst_mu": float(accepted_worst[mu_col]),
        "rejected_margin_min": float(rejected_margin.min()),
        "rejected_margin_worst_D": float(rejected_worst["D"]),
        "rejected_margin_worst_eta": float(rejected_worst["eta"]),
        "rejected_margin_worst_mu": float(rejected_worst[mu_col]),
        "uniform_mu_set_invariance_radius": set_radius,
    }


def main() -> None:
    args = parse_args()
    impact = pd.read_csv(args.impact_detail).sort_values(["D", "eta"]).reset_index(drop=True)
    impact_summary = pd.read_csv(args.impact_summary)
    continuation_summary = pd.read_csv(args.continuation_summary)
    topology_summary = pd.read_csv(args.topology_summary).iloc[0]

    for col in ("accept_release", "accept_direct_y2", "acceptance_gain", "acceptance_loss"):
        impact[col] = bool_series(impact[col])

    params = continuation_summary.loc[
        continuation_summary["section"] == "projector_continuation_parameters"
    ].iloc[0]
    lower = float(params["accept_lower"])
    upper = float(params["accept_upper"])
    mu_obs = float(params["mu_obs"])
    sigma_obs = float(params["sigma_obs"])

    impact["release_inside_margin"] = distance_inside_interval(
        impact["mu_mumu_release_bridge"], lower, upper
    )
    impact["direct_inside_margin"] = distance_inside_interval(
        impact["mu_mumu_direct_y2"], lower, upper
    )
    impact["release_outside_margin"] = distance_outside_interval(
        impact["mu_mumu_release_bridge"], lower, upper
    )
    impact["direct_outside_margin"] = distance_outside_interval(
        impact["mu_mumu_direct_y2"], lower, upper
    )
    impact["z_release"] = np.abs(impact["mu_mumu_release_bridge"] - mu_obs) / sigma_obs
    impact["z_direct"] = np.abs(impact["mu_mumu_direct_y2"] - mu_obs) / sigma_obs
    impact["direct_minus_release_mu"] = (
        impact["mu_mumu_direct_y2"] - impact["mu_mumu_release_bridge"]
    )
    impact["release_minus_direct_mu"] = -impact["direct_minus_release_mu"]

    release_branch = summarize_branch(
        name="release",
        df=impact,
        mu_col="mu_mumu_release_bridge",
        accept_col="accept_release",
        lower=lower,
        upper=upper,
    )
    direct_branch = summarize_branch(
        name="direct",
        df=impact,
        mu_col="mu_mumu_direct_y2",
        accept_col="accept_direct_y2",
        lower=lower,
        upper=upper,
    )

    gains = impact[impact["acceptance_gain"]].copy()
    release_accepted = impact[impact["accept_release"]].copy()
    direct_accepted = impact[impact["accept_direct_y2"]].copy()
    direct_rejected = impact[~impact["accept_direct_y2"]].copy()

    direct_radius = float(direct_branch["uniform_mu_set_invariance_radius"])
    release_radius = float(release_branch["uniform_mu_set_invariance_radius"])
    pointwise_direct_le_release = bool((impact["direct_minus_release_mu"] <= 1e-14).all())
    direct_margin_stronger = bool(direct_radius > release_radius)
    no_losses = int(impact["acceptance_loss"].sum()) == 0
    topology_verdict = str(topology_summary["verdict"])
    verdict = (
        "margin_robust_explicit_topology_branch"
        if (
            topology_verdict == "adoption_safe_under_explicit_monotone_topology_expansion"
            and pointwise_direct_le_release
            and no_losses
            and direct_margin_stronger
        )
        else "margin_certificate_failed"
    )

    release_case = impact_summary.loc[impact_summary["case"] == "release_interpolated_y2"].iloc[0]
    direct_case = impact_summary.loc[impact_summary["case"] == "direct_inserted_history_y2"].iloc[0]
    cert_row = continuation_summary.loc[
        continuation_summary["section"] == "topology_crossing_certificate"
    ].iloc[0]

    gain_direct_upper_margin = upper - gains["mu_mumu_direct_y2"]
    gain_release_upper_excess = gains["mu_mumu_release_bridge"] - upper
    gain_direct_sigma_slack = 2.0 - gains["z_direct"]
    gain_release_sigma_excess = gains["z_release"] - 2.0

    summary = pd.DataFrame(
        [
            {
                "section": "margin_robust_topology_branch",
                "verdict": verdict,
                "accept_lower": lower,
                "accept_upper": upper,
                "mu_obs": mu_obs,
                "sigma_obs": sigma_obs,
                "n_points": int(len(impact)),
                "n_accept_release": int(len(release_accepted)),
                "n_accept_direct": int(len(direct_accepted)),
                "n_gains": int(impact["acceptance_gain"].sum()),
                "n_losses": int(impact["acceptance_loss"].sum()),
                "pointwise_direct_le_release": pointwise_direct_le_release,
                "max_direct_minus_release_mu": float(impact["direct_minus_release_mu"].max()),
                "max_release_minus_direct_mu": float(impact["release_minus_direct_mu"].max()),
                "release_uniform_mu_set_invariance_radius": release_radius,
                "direct_uniform_mu_set_invariance_radius": direct_radius,
                "direct_over_release_radius_factor": float(direct_radius / release_radius),
                "direct_accepted_margin_min": float(direct_branch["accepted_margin_min"]),
                "direct_accepted_margin_worst_D": float(direct_branch["accepted_margin_worst_D"]),
                "direct_accepted_margin_worst_eta": float(direct_branch["accepted_margin_worst_eta"]),
                "direct_accepted_margin_worst_mu": float(direct_branch["accepted_margin_worst_mu"]),
                "direct_rejected_margin_min": float(direct_branch["rejected_margin_min"]),
                "direct_rejected_margin_worst_D": float(direct_branch["rejected_margin_worst_D"]),
                "direct_rejected_margin_worst_eta": float(direct_branch["rejected_margin_worst_eta"]),
                "direct_rejected_margin_worst_mu": float(direct_branch["rejected_margin_worst_mu"]),
                "gain_direct_upper_margin_min": float(gain_direct_upper_margin.min()),
                "gain_direct_upper_margin_max": float(gain_direct_upper_margin.max()),
                "gain_direct_sigma_slack_min": float(gain_direct_sigma_slack.min()),
                "gain_direct_sigma_slack_max": float(gain_direct_sigma_slack.max()),
                "gain_release_upper_excess_min": float(gain_release_upper_excess.min()),
                "gain_release_upper_excess_max": float(gain_release_upper_excess.max()),
                "gain_release_sigma_excess_min": float(gain_release_sigma_excess.min()),
                "gain_release_sigma_excess_max": float(gain_release_sigma_excess.max()),
                "release_best_chi2": float(release_case["best_chi2"]),
                "direct_best_chi2": float(direct_case["best_chi2"]),
                "best_chi2_improvement_factor": float(
                    float(release_case["best_chi2"]) / float(direct_case["best_chi2"])
                ),
                "direct_best_mu": float(direct_case["best_mu_mumu"]),
                "release_best_mu": float(release_case["best_mu_mumu"]),
                "first_cross_s": float(cert_row["first_cross_s"]),
                "direct_endpoint_s": 1.0,
                "cross_validation_error": float(cert_row["max_abs_mu_cross_validation_error"]),
                "topology_policy_verdict": topology_verdict,
            }
        ]
    )

    branch_summary = pd.DataFrame([release_branch, direct_branch])
    detail = impact[
        [
            "D",
            "eta",
            "mu_mumu_release_bridge",
            "mu_mumu_direct_y2",
            "accept_release",
            "accept_direct_y2",
            "acceptance_gain",
            "acceptance_loss",
            "release_inside_margin",
            "direct_inside_margin",
            "release_outside_margin",
            "direct_outside_margin",
            "z_release",
            "z_direct",
            "direct_minus_release_mu",
            "release_minus_direct_mu",
        ]
    ].copy()
    detail["direct_uniform_mu_set_invariance_radius"] = direct_radius
    detail["release_uniform_mu_set_invariance_radius"] = release_radius

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_topology_margin_certificate_summary.csv"
    branch_out = outdir / "hll_y2raw_topology_margin_certificate_branches.csv"
    detail_out = outdir / "hll_y2raw_topology_margin_certificate_detail.csv"
    meta_out = outdir / "hll_y2raw_topology_margin_certificate_run_meta.json"
    summary.to_csv(summary_out, index=False)
    branch_summary.to_csv(branch_out, index=False)
    detail.to_csv(detail_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "impact_detail": str(args.impact_detail),
                "impact_summary": str(args.impact_summary),
                "continuation_summary": str(args.continuation_summary),
                "topology_summary": str(args.topology_summary),
                "policy": "uniform-mu boundary margin around the U13 direct topology branch",
                "verdict": verdict,
            },
            indent=2,
        )
    )

    if args.copy_paper:
        paper_dir = ROOT / "paper"
        for path in (summary_out, branch_out, detail_out, meta_out):
            shutil.copy2(path, paper_dir / path.name)

    print(summary_out)
    print(branch_out)
    print(detail_out)
    print(meta_out)
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()
