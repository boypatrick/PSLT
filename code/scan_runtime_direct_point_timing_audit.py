#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import numpy as np

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"


@dataclass
class TimingRow:
    chain_mode: str
    phase: str
    d: float
    eta: float
    mu_mumu: float
    wall_seconds: float


def _clear_runtime_caches(kinetics: Any) -> None:
    cache_names = [
        "_gamma_prefactor_cache",
        "_g_fp_2d_spectrum_interp_cache",
        "_g_fp_2d_full_hat_cache",
        "_g_runtime_direct_spectrum_cache",
        "_g_fp_2d_full_hat_runtime_cache",
        "_chi_open_cache",
        "_chi_runtime_direct_cache",
        "_gamma_runtime_direct_cache",
        "_b_eft_norm_cache",
        "_b_runtime_direct_input_cache",
    ]
    for name in cache_names:
        obj = getattr(kinetics, name, None)
        if hasattr(obj, "clear"):
            obj.clear()


def _instrument_methods(kinetics: Any, names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {name: {"calls": 0.0, "total": 0.0, "exclusive": 0.0} for name in names}
    call_stack: list[dict[str, float]] = []

    def make_wrapper(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            frame = {"child": 0.0}
            call_stack.append(frame)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                call_stack.pop()
                exclusive = elapsed - frame["child"]
                stats[name]["calls"] += 1.0
                stats[name]["total"] += elapsed
                stats[name]["exclusive"] += exclusive
                if call_stack:
                    call_stack[-1]["child"] += elapsed

        return wrapper

    for name in names:
        original = getattr(kinetics, name)
        setattr(kinetics, name, MethodType(make_wrapper(name, original), kinetics))
    return stats


def _run_point(chain_mode: str, d: float, eta: float, phase: str) -> tuple[TimingRow, dict[str, dict[str, float]], float]:
    build_t0 = time.perf_counter()
    kinetics = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=chain_mode,
        d_min=4.0,
        d_max=8.0,
        d_num=6,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
        uv_match_mode="constant",
        uv_match_input_diag_scale=0.0,
        uv_match_input_offdiag_scale=0.0,
        uv_rge_mu_low=1.0,
        uv_rge_gamma_diag=2.0,
        uv_rge_gamma_offdiag=1.0,
        uv_rge_log_clip=6.0,
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=2.5,
        runtime_direct_chi_z_margin=5.0,
        runtime_direct_chi_n_mu=40,
        runtime_direct_chi_tol=1e-6,
        runtime_direct_chi_maxiter=6000,
        runtime_direct_chi_sigma=2.5,
    )
    build_seconds = time.perf_counter() - build_t0

    method_names = [
        "hll_mu_pred",
        "hll_channel_amplitude",
        "hll_total_width_ratio_uv_rge",
        "hll_wilson_coeff_uv_rge",
        "hll_wilson_matrix_uv_rge",
        "hll_wilson_matrix_uv_rge_with_meta",
        "hll_wilson_matrix_uv_tree",
        "_hll_g_uv_matrix",
        "_hll_pkin_vector",
        "_hll_m2_vector",
        "_runtime_direct_b_operator_inputs",
        "_runtime_direct_chi",
        "_runtime_direct_a12",
        "layer_weight",
        "layer_kinetic_weight",
    ]
    stats = _instrument_methods(kinetics, method_names)

    _clear_runtime_caches(kinetics)

    if phase == "warm":
        kinetics.hll_mu_pred(
            2,
            d,
            eta,
            float(PAPER_BASELINE["t_coh"]),
            ref_D=8.0,
            ref_eta=1.0,
            observable_mode="eft_wilson_uv_rge",
            N_max=int(PAPER_BASELINE["hll_observable_nmax"]),
        )
        for entry in stats.values():
            entry["calls"] = 0.0
            entry["total"] = 0.0
            entry["exclusive"] = 0.0

    t0 = time.perf_counter()
    mu_val = kinetics.hll_mu_pred(
        2,
        d,
        eta,
        float(PAPER_BASELINE["t_coh"]),
        ref_D=8.0,
        ref_eta=1.0,
        observable_mode="eft_wilson_uv_rge",
        N_max=int(PAPER_BASELINE["hll_observable_nmax"]),
    )
    wall = time.perf_counter() - t0

    row = TimingRow(
        chain_mode=chain_mode,
        phase=phase,
        d=float(d),
        eta=float(eta),
        mu_mumu=float(mu_val),
        wall_seconds=float(wall),
    )
    return row, stats, float(build_seconds)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    points = [(4.8, 0.2), (6.4, 4.0)]
    chain_modes = ["full_direct", "cell_direct_runtime_release_tailm2"]
    phases = ["cold", "warm"]

    detail_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    build_rows: list[dict[str, float | str]] = []

    for chain_mode in chain_modes:
        for d, eta in points:
            for phase in phases:
                row, stats, build_seconds = _run_point(chain_mode=chain_mode, d=d, eta=eta, phase=phase)
                build_rows.append(
                    {
                        "chain_mode": chain_mode,
                        "phase": phase,
                        "D": d,
                        "eta": eta,
                        "build_seconds": build_seconds,
                    }
                )
                summary_rows.append(
                    {
                        "chain_mode": row.chain_mode,
                        "phase": row.phase,
                        "D": row.d,
                        "eta": row.eta,
                        "mu_mumu": row.mu_mumu,
                        "wall_seconds": row.wall_seconds,
                    }
                )
                for name, entry in stats.items():
                    detail_rows.append(
                        {
                            "chain_mode": row.chain_mode,
                            "phase": row.phase,
                            "D": row.d,
                            "eta": row.eta,
                            "method": name,
                            "calls": int(entry["calls"]),
                            "total_seconds": float(entry["total"]),
                            "exclusive_seconds": float(entry["exclusive"]),
                            "share_of_wall": float(entry["total"] / max(row.wall_seconds, 1e-30)),
                        }
                    )

    detail_path = OUTDIR / "runtime_direct_point_timing_audit_detail.csv"
    summary_path = OUTDIR / "runtime_direct_point_timing_audit_summary.csv"
    build_path = OUTDIR / "runtime_direct_point_timing_audit_build.csv"
    meta_path = OUTDIR / "runtime_direct_point_timing_audit_run_meta.json"

    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chain_mode",
                "phase",
                "D",
                "eta",
                "method",
                "calls",
                "total_seconds",
                "exclusive_seconds",
                "share_of_wall",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chain_mode", "phase", "D", "eta", "mu_mumu", "wall_seconds"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with build_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chain_mode", "phase", "D", "eta", "build_seconds"],
        )
        writer.writeheader()
        writer.writerows(build_rows)

    meta = {
        "points": [{"D": d, "eta": eta} for d, eta in points],
        "chain_modes": chain_modes,
        "phases": phases,
        "observable_mode": "eft_wilson_uv_rge",
        "ref_D": 8.0,
        "ref_eta": 1.0,
        "runtime_direct_smoke_settings": {
            "runtime_direct_chi_rho_max": 2.5,
            "runtime_direct_chi_z_margin": 5.0,
            "runtime_direct_chi_n_mu": 40,
            "runtime_direct_chi_tol": 1e-6,
            "runtime_direct_chi_maxiter": 6000,
            "runtime_direct_chi_sigma": 2.5,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    for path in (detail_path, summary_path, build_path, meta_path):
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(f"[done] wrote {detail_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {build_path}")
    print(f"[done] wrote {meta_path}")


if __name__ == "__main__":
    main()
