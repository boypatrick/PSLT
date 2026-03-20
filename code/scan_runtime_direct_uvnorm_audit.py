#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"

FULL_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"
TAIL_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_tailm2_D21E41.csv"

OUT_MAP = OUTDIR / "runtime_direct_uvnorm_audit_map.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_uvnorm_audit_summary.csv"
OUT_PNG = OUTDIR / "runtime_direct_uvnorm_audit.png"
OUT_META = OUTDIR / "runtime_direct_uvnorm_audit_run_meta.json"


def _load_map(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    rows = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = (float(row["D"]), float(row["eta"]))
            rows[key] = {
                "mu_mumu": float(row["mu_mumu"]),
                "chi2_mumu": float(row["chi2_mumu"]),
            }
    return rows


def _build_kinetics(mode: str):
    return make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=mode,
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=float(PAPER_BASELINE["hll_uv_blend"]),
        uv_m2_power=float(PAPER_BASELINE["hll_uv_m2_power"]),
        uv_match_kappa_diag=float(PAPER_BASELINE["hll_uv_match_kappa_diag"]),
        uv_match_kappa_offdiag=float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"]),
        uv_match_mode=str(PAPER_BASELINE.get("hll_uv_match_mode", "constant")),
        uv_match_input_diag_scale=float(PAPER_BASELINE.get("hll_uv_match_input_diag_scale", 0.0)),
        uv_match_input_offdiag_scale=float(PAPER_BASELINE.get("hll_uv_match_input_offdiag_scale", 0.0)),
        uv_rge_mu_low=float(PAPER_BASELINE["hll_uv_rge_mu_low"]),
        uv_rge_gamma_diag=float(PAPER_BASELINE["hll_uv_rge_gamma_diag"]),
        uv_rge_gamma_offdiag=float(PAPER_BASELINE["hll_uv_rge_gamma_offdiag"]),
        uv_rge_log_clip=float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
    )


def _g_diag(kin, D: float) -> np.ndarray:
    return np.diag(kin._hll_g_uv_matrix(D)).astype(float)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    full_map = _load_map(FULL_MAP)
    tail_map = _load_map(TAIL_MAP)
    d_vals = sorted({d for d, _ in full_map})
    eta_vals = sorted({e for _, e in full_map})

    kin_full = _build_kinetics("full_direct")
    kin_tail = _build_kinetics("cell_direct_runtime_release_tailm2")

    ref_D = float(PAPER_BASELINE["ref_D"])
    g_full_ref = _g_diag(kin_full, ref_D)
    g_tail_ref = _g_diag(kin_tail, ref_D)

    rows: list[dict[str, float]] = []
    for D in d_vals:
        g_full = _g_diag(kin_full, D)
        g_tail = _g_diag(kin_tail, D)

        log_gnorm_full = float(np.mean(np.log(g_full)))
        log_gnorm_tail = float(np.mean(np.log(g_tail)))
        delta_log_gnorm = float(log_gnorm_tail - log_gnorm_full)

        log_g2_anchor_full = float(np.log(g_full[1] / g_full_ref[1]))
        log_g2_anchor_tail = float(np.log(g_tail[1] / g_tail_ref[1]))
        delta_log_g2_anchor = float(log_g2_anchor_tail - log_g2_anchor_full)

        log_side_anchor_full = float(np.log(np.sqrt(g_full[0] * g_full[2]) / g_full[1]))
        log_side_anchor_tail = float(np.log(np.sqrt(g_tail[0] * g_tail[2]) / g_tail[1]))
        delta_log_side_anchor = float(log_side_anchor_tail - log_side_anchor_full)

        slice_deltas = []
        slice_mismatch = 0
        for eta in eta_vals:
            f = full_map[(D, eta)]
            t = tail_map[(D, eta)]
            slice_deltas.append(abs(t["mu_mumu"] - f["mu_mumu"]))
            if (f["chi2_mumu"] <= 4.0) != (t["chi2_mumu"] <= 4.0):
                slice_mismatch += 1
        slice_deltas = np.asarray(slice_deltas, dtype=float)

        rows.append(
            {
                "D": D,
                "g1_full": float(g_full[0]),
                "g2_full": float(g_full[1]),
                "g3_full": float(g_full[2]),
                "g1_tail": float(g_tail[0]),
                "g2_tail": float(g_tail[1]),
                "g3_tail": float(g_tail[2]),
                "delta_log_gnorm": delta_log_gnorm,
                "delta_log_g2_anchor": delta_log_g2_anchor,
                "delta_log_side_anchor": delta_log_side_anchor,
                "slice_max_abs_delta_mu_mumu": float(np.max(slice_deltas)),
                "slice_p95_abs_delta_mu_mumu": float(np.percentile(slice_deltas, 95.0)),
                "slice_acceptance_mismatch": float(slice_mismatch / max(len(eta_vals), 1)),
            }
        )

    with OUT_MAP.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    abs_gnorm = np.array([abs(r["delta_log_gnorm"]) for r in rows], dtype=float)
    abs_g2 = np.array([abs(r["delta_log_g2_anchor"]) for r in rows], dtype=float)
    abs_side = np.array([abs(r["delta_log_side_anchor"]) for r in rows], dtype=float)
    abs_mu = np.array([r["slice_max_abs_delta_mu_mumu"] for r in rows], dtype=float)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    worst = max(rows, key=lambda r: r["slice_max_abs_delta_mu_mumu"])
    summary = {
        "ref_D": ref_D,
        "worst_D": worst["D"],
        "worst_slice_max_abs_delta_mu_mumu": worst["slice_max_abs_delta_mu_mumu"],
        "worst_slice_acceptance_mismatch": worst["slice_acceptance_mismatch"],
        "p95_abs_delta_log_gnorm": float(np.percentile(abs_gnorm, 95.0)),
        "p95_abs_delta_log_g2_anchor": float(np.percentile(abs_g2, 95.0)),
        "p95_abs_delta_log_side_anchor": float(np.percentile(abs_side, 95.0)),
        "corr_abs_mu_vs_abs_delta_log_gnorm": _corr(abs_mu, abs_gnorm),
        "corr_abs_mu_vs_abs_delta_log_g2_anchor": _corr(abs_mu, abs_g2),
        "corr_abs_mu_vs_abs_delta_log_side_anchor": _corr(abs_mu, abs_side),
    }

    with OUT_SUMMARY.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    x = np.array([r["D"] for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(x, [r["slice_max_abs_delta_mu_mumu"] for r in rows], marker="o", label="slice max |Δμ_μμ|")
    axes[0].plot(x, [r["slice_p95_abs_delta_mu_mumu"] for r in rows], marker="s", label="slice p95 |Δμ_μμ|")
    axes[0].set_ylabel("|Δμ_μμ|")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [abs(r["delta_log_gnorm"]) for r in rows], marker="o", label="|Δ log g_norm|")
    axes[1].plot(x, [abs(r["delta_log_g2_anchor"]) for r in rows], marker="s", label="|Δ log g2(anchor)|")
    axes[1].plot(x, [abs(r["delta_log_side_anchor"]) for r in rows], marker="^", label="|Δ log side/g2|")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("anchored log drift")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Runtime-Direct UV Normalization Audit: full_direct vs tailm2")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    meta = {
        "full_map": str(FULL_MAP),
        "tail_map": str(TAIL_MAP),
        "rows": len(rows),
        "eta_count": len(eta_vals),
        "summary": summary,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    for path in (OUT_MAP, OUT_SUMMARY, OUT_PNG, OUT_META):
        shutil.copy2(path, PAPERDIR / path.name)
        print(f"[saved] {path}")

    print(",".join(summary.keys()))
    print(",".join(str(summary[k]) for k in summary))


if __name__ == "__main__":
    main()
