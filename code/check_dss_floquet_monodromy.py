#!/usr/bin/env python3
"""D2 toy Floquet monodromy gate for the large-d DSS parent queue.

The goal is deliberately narrow: use the D1 periodic beta(tau) profile to build
a smallest auditable Hill/Floquet toy problem and test whether the periodic DSS
background supports a stable ordering of Floquet quasi-frequencies.

This is not a PSLT baseline calculation and it does not define Gamma_N.  The
Hamiltonian toy has unit-modulus Floquet multipliers, so a positive result here
only supports phase/band ordering, not growth/decay rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "dss_floquet"
SUMMARY_CSV = OUTDIR / "dss_floquet_monodromy_summary.csv"
DETAIL_CSV = OUTDIR / "dss_floquet_monodromy_detail.csv"
DETAIL_JSON = OUTDIR / "dss_floquet_monodromy_detail.json"


def beta(tau: float, A: float) -> float:
    return math.cos(2.0 * math.pi * tau) + math.sin(6.0 * math.pi * tau) / A


def ilogeo(tau: float, A: float) -> float:
    """Spatially averaged LO geometry strength int_0^1 Omega_LO dx.

    Omega_LO = beta^2/(1+beta^2 x^2), hence
    int_0^1 Omega_LO dx = |beta| atan(|beta|), with the zero case continuous.
    """
    b = abs(beta(tau, A))
    return b * math.atan(b)


def mean_ilogeo(A: float, n: int = 20000) -> float:
    # Periodic trapezoid rule on [0,1).
    return sum(ilogeo(k / n, A) for k in range(n)) / n


def rk4_monodromy(mode: int, kappa: float, A_param: float, q_mean: float, steps: int) -> np.ndarray:
    """Integrate one-period fundamental matrix for y'' + Q_N(tau)y = 0."""
    omega0 = math.pi * (mode - 0.5)
    h = 1.0 / steps
    Y = np.eye(2, dtype=float)

    def matrix(t: float) -> np.ndarray:
        q = ilogeo(t % 1.0, A_param) - q_mean
        coeff = omega0 * omega0 + kappa * q
        return np.array([[0.0, 1.0], [-coeff, 0.0]], dtype=float)

    def rhs(t: float, M: np.ndarray) -> np.ndarray:
        return matrix(t) @ M

    t = 0.0
    for _ in range(steps):
        k1 = rhs(t, Y)
        k2 = rhs(t + 0.5 * h, Y + 0.5 * h * k1)
        k3 = rhs(t + 0.5 * h, Y + 0.5 * h * k2)
        k4 = rhs(t + h, Y + h * k3)
        Y = Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return Y


def unwrap_quasi_frequency(theta_principal: float, omega0: float) -> float:
    """Choose the Floquet phase branch closest to the unperturbed frequency."""
    candidates: list[float] = []
    for sign in (-1.0, 1.0):
        for k in range(-8, 16):
            cand = sign * theta_principal + 2.0 * math.pi * k
            if cand > 0.0:
                candidates.append(cand)
    return min(candidates, key=lambda x: abs(x - omega0))


def analyze_mode(mode: int, kappa: float, A_param: float, q_mean: float, steps: int) -> dict[str, object]:
    M = rk4_monodromy(mode, kappa, A_param, q_mean, steps)
    eigvals = np.linalg.eigvals(M)
    det = float(np.linalg.det(M))
    trace = float(np.trace(M))
    moduli = np.abs(eigvals)
    log_abs = np.log(moduli)
    max_log_abs = float(np.max(np.abs(log_abs)))
    stable = bool(np.max(np.abs(moduli - 1.0)) < 5e-6)
    omega0 = math.pi * (mode - 0.5)
    # For a stable 2x2 symplectic monodromy, the principal phase is determined
    # by either the eigenvalue angles or arccos(trace/2).  Clip trace to avoid
    # harmless roundoff outside [-2,2].
    trace_half = max(-1.0, min(1.0, 0.5 * trace))
    theta_from_trace = abs(math.acos(trace_half))
    angles = np.angle(eigvals)
    theta_from_eigs = float(np.max(np.abs(angles)))
    theta = theta_from_trace if stable else theta_from_eigs
    quasi_frequency = unwrap_quasi_frequency(theta, omega0)
    return {
        "mode": mode,
        "kappa": kappa,
        "omega0": omega0,
        "trace": trace,
        "det": det,
        "det_residual": abs(det - 1.0),
        "eigval_0": complex(eigvals[0]),
        "eigval_1": complex(eigvals[1]),
        "max_abs_log_multiplier": max_log_abs,
        "stable_unit_circle": stable,
        "theta_principal": theta,
        "quasi_frequency_unwrapped": quasi_frequency,
        "quasi_shift": quasi_frequency - omega0,
    }


def parse_kappas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=15.947599366191568, help="D1 solved A for Delta=1")
    parser.add_argument("--modes", type=int, default=6, help="number of toy modes to test")
    parser.add_argument("--steps", type=int, default=8000, help="RK4 steps per period")
    parser.add_argument("--kappas", default="0.0,0.25,0.5,1.0", help="comma-separated modulation strengths")
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json
    for path in (summary_path, detail_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    kappas = parse_kappas(args.kappas)
    q_mean = mean_ilogeo(args.A)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for kappa in kappas:
        mode_rows = [analyze_mode(mode, kappa, args.A, q_mean, args.steps) for mode in range(1, args.modes + 1)]
        qfreqs = [float(r["quasi_frequency_unwrapped"]) for r in mode_rows]
        gaps = [qfreqs[i + 1] - qfreqs[i] for i in range(len(qfreqs) - 1)]
        stable_all = all(bool(r["stable_unit_circle"]) for r in mode_rows)
        ordering_pass = all(g > 0.0 for g in gaps)
        min_gap = min(gaps) if gaps else float("nan")
        max_log_abs = max(float(r["max_abs_log_multiplier"]) for r in mode_rows)
        max_det_residual = max(float(r["det_residual"]) for r in mode_rows)
        max_abs_shift = max(abs(float(r["quasi_shift"])) for r in mode_rows)
        verdict = (
            "TOY_FLOQUET_PHASE_ORDERING_STABLE_NO_GROWTH_CLAIM"
            if stable_all and ordering_pass
            else "TOY_FLOQUET_ORDERING_NOT_STABLE_DIAGNOSTIC_ONLY"
        )

        summary_rows.append(
            {
                "gate": "D2",
                "kappa": f"{kappa:.8g}",
                "modes": args.modes,
                "steps": args.steps,
                "A": f"{args.A:.16e}",
                "q_mean": f"{q_mean:.16e}",
                "stable_all": stable_all,
                "ordering_pass": ordering_pass,
                "min_quasi_frequency_gap": f"{min_gap:.16e}",
                "max_abs_log_multiplier": f"{max_log_abs:.16e}",
                "max_det_residual": f"{max_det_residual:.16e}",
                "max_abs_quasi_shift": f"{max_abs_shift:.16e}",
                "verdict": verdict,
            }
        )

        previous_qfreq = None
        for r in mode_rows:
            qfreq = float(r["quasi_frequency_unwrapped"])
            gap_prev = float("nan") if previous_qfreq is None else qfreq - previous_qfreq
            previous_qfreq = qfreq
            detail_rows.append(
                {
                    "gate": "D2",
                    "kappa": f"{kappa:.8g}",
                    "mode": r["mode"],
                    "omega0": f"{float(r['omega0']):.16e}",
                    "quasi_frequency_unwrapped": f"{qfreq:.16e}",
                    "quasi_shift": f"{float(r['quasi_shift']):.16e}",
                    "gap_from_previous": "" if math.isnan(gap_prev) else f"{gap_prev:.16e}",
                    "stable_unit_circle": r["stable_unit_circle"],
                    "max_abs_log_multiplier": f"{float(r['max_abs_log_multiplier']):.16e}",
                    "det_residual": f"{float(r['det_residual']):.16e}",
                    "trace": f"{float(r['trace']):.16e}",
                    "theta_principal": f"{float(r['theta_principal']):.16e}",
                    "eigval_0": str(r["eigval_0"]),
                    "eigval_1": str(r["eigval_1"]),
                }
            )

    with summary_path.open("w", newline="") as f:
        fieldnames = list(summary_rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_path.open("w", newline="") as f:
        fieldnames = list(detail_rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    overall_pass = all(row["stable_all"] and row["ordering_pass"] for row in summary_rows)
    overall_verdict = (
        "D2_TOY_FLOQUET_PHASE_ORDERING_STABLE_NO_GROWTH_CLAIM"
        if overall_pass
        else "D2_TOY_FLOQUET_MONODROMY_DIAGNOSTIC_ONLY"
    )
    detail_json = {
        "claim_firewall": {
            "baseline_change": False,
            "main_tex_change": False,
            "defines_Gamma_N": False,
            "interpretation": "toy Hamiltonian Floquet phase ordering only; no growth-rate claim",
        },
        "profile": {
            "A": args.A,
            "Delta": 1.0,
            "I_LO": "int_0^1 Omega_LO dx = |beta| atan(|beta|)",
            "q_tau": "I_LO(tau) - mean(I_LO)",
            "q_mean": q_mean,
        },
        "toy_operator": "y_N'' + [omega_N^2 + kappa q(tau)] y_N = 0, omega_N=pi(N-1/2)",
        "overall_verdict": overall_verdict,
        "summary_rows": summary_rows,
    }
    with json_path.open("w") as f:
        json.dump(detail_json, f, indent=2)
        f.write("\n")

    print(overall_verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"detail_json={json_path}")
    for row in summary_rows:
        print(
            "kappa={kappa} stable_all={stable_all} ordering_pass={ordering_pass} "
            "min_gap={min_quasi_frequency_gap} max_log_abs={max_abs_log_multiplier}".format(**row)
        )


if __name__ == "__main__":
    main()
