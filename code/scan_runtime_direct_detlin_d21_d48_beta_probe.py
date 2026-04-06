#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv"
VALIDATOR = ROOT / "code" / "scan_runtime_direct_detlin_generalization_fast_validation.py"
BASELINE_DETAIL = Path(
    os.environ.get(
        "BASELINE_DETAIL",
        str(
            ROOT
            / "output"
            / "kinetic_action_chain"
            / "runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64d48d64amp_v20_casesplit72_fixed.csv"
        ),
    )
)
TAG = os.environ.get("D21_D48_PROBE_TAG", "v1").strip() or "v1"
OUT_SAMPLES = OUTDIR / f"runtime_direct_detlin_d21_d48_beta_probe_{TAG}_samples.csv"
OUT_SUMMARY = OUTDIR / f"runtime_direct_detlin_d21_d48_beta_probe_{TAG}_summary.json"

BETAS = [float(x) for x in os.environ.get("BETAS", "0.68,0.69,0.70,0.71,0.72").split(",") if x.strip()]
TARGET_D = 4.8

ENV_CONST = {
    "STRICT_GNORM_BETA_D21": "0.31",
    "STRICT_GNORM_CENTER_D21": "0.07",
    "STRICT_GNORM_HALF_WIDTH_D21": "0.045",
    "STRICT_GNORM_BETA_D60": "0.2468174085",
    "STRICT_GNORM_CENTER_D60": "0.0488372924",
    "STRICT_GNORM_HALF_WIDTH_D60": "0.0251312702",
    "STRICT_D40_BETA_D21": "1.20",
    "STRICT_D40_BETA_D60": "1.304",
    "STRICT_D48_BETA_D60": "0.56",
    "STRICT_D64_BETA_D21": "0.0",
    "STRICT_D64_BETA_D60": "0.05",
    "STRICT_D72_BETA_D21": "0.35",
    "STRICT_D72_BETA_D60": "0.455",
}


def load_validator():
    spec = importlib.util.spec_from_file_location("valmod", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    for key, value in ENV_CONST.items():
        os.environ[key] = value

    mod = load_validator()
    df = pd.read_csv(MAP)
    target_D = float(sorted(df["D"].unique(), key=lambda x: abs(float(x) - TARGET_D))[0])
    sub = df[np.isclose(df["D"], target_D)].copy()

    baseline = pd.read_csv(BASELINE_DETAIL)
    baseline = baseline[(baseline["case"] == "D21E21_holdout") & np.isclose(baseline["eval_D"], target_D)]
    baseline_p95 = float(np.percentile(baseline["abs_delta_mu_mumu"], 95.0))
    baseline_max = float(baseline["abs_delta_mu_mumu"].max())

    rows = []
    for i, beta in enumerate(BETAS, start=1):
        os.environ["STRICT_D48_BETA_D21"] = str(beta)
        mod = load_validator()
        kin = mod.build_kinetics("D21E21_holdout", float(df["D"].min()), float(df["D"].max()), int(df["D"].nunique()))
        deltas = []
        signed = []
        for row in sub.itertuples(index=False):
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(row.D),
                    eta=float(row.eta),
                    t_coh=mod.T_COH,
                    ref_D=mod.REF_D,
                    ref_eta=mod.REF_ETA,
                    observable_mode="eft_wilson_uv_rge",
                    N_max=kin.params.hll_observable_nmax,
                )
            )
            diff = mu - float(row.mu_mumu)
            deltas.append(abs(diff))
            signed.append(diff)
        result = {
            "beta_d21_d48": float(beta),
            "p95_abs_delta_mu_mumu": float(np.percentile(deltas, 95.0)),
            "max_abs_delta_mu_mumu": float(np.max(deltas)),
            "mean_signed_delta_mu_mumu": float(np.mean(signed)),
            "min_signed_delta_mu_mumu": float(np.min(signed)),
            "max_signed_delta_mu_mumu": float(np.max(signed)),
            "delta_p95_vs_v20": float(np.percentile(deltas, 95.0) - baseline_p95),
            "delta_max_vs_v20": float(np.max(deltas) - baseline_max),
        }
        rows.append(result)
        with OUT_SAMPLES.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        best_so_far = min(rows, key=lambda r: r["p95_abs_delta_mu_mumu"])
        OUT_SUMMARY.write_text(
            json.dumps(
                {
                    "target_D": float(target_D),
                    "baseline": {
                        "detail_path": str(BASELINE_DETAIL),
                        "p95_abs_delta_mu_mumu": baseline_p95,
                        "max_abs_delta_mu_mumu": baseline_max,
                    },
                    "best_probe_so_far": best_so_far,
                    "completed_probes": rows,
                },
                indent=2,
            )
        )
        print(
            f"[progress] {i}/{len(BETAS)} beta={beta:.3f} "
            f"p95={result['p95_abs_delta_mu_mumu']:.6f} "
            f"delta_vs_v20={result['delta_p95_vs_v20']:+.6f} "
            f"mean_signed={result['mean_signed_delta_mu_mumu']:+.6f}",
            flush=True,
        )

    best = min(rows, key=lambda r: r["p95_abs_delta_mu_mumu"])
    summary = {
        "target_D": float(target_D),
        "baseline": {
            "detail_path": str(BASELINE_DETAIL),
            "p95_abs_delta_mu_mumu": baseline_p95,
            "max_abs_delta_mu_mumu": baseline_max,
        },
        "best_probe": best,
        "all_probes": rows,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    for path in [OUT_SAMPLES, OUT_SUMMARY]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
