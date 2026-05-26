#!/usr/bin/env python3
"""D3 projected-conformal bridge gate for the DSS/Floquet feasibility queue.

This is a hard necessary-condition test, not a PSLT baseline calculation.
If a projected large-d DSS parent is to approximate the PSLT two-center
conformal factor in a way that can control V_eff, it must at least control the
axial restriction in C^2, because V_eff contains Omega and nabla^2 Omega.

The candidate family is intentionally generous:

    Omega_DSS^R(z) = c0 + c1 [F(|z-D_sep/2|/R) + F(|z+D_sep/2|/R)],

where F(x)=<beta(tau)^2/(1+beta(tau)^2 x^2)>_tau is the period-averaged LO DSS
geometry strength.  For each window we scan R and fit c0,c1 against Omega,
d_z Omega, and d_z^2 Omega with scale-normalized least squares.  Two policies
are reported:

* domain_safe: keeps all lobe coordinates in the LO DSS x<=1 chart.
* analytic_extension: allows x>1 as a diagnostic analytic continuation.

Passing D3 requires the required windows to pass C0, C1, and C2 relative
sup-norm guards under the domain_safe policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "dss_floquet"
SUMMARY_CSV = OUTDIR / "dss_projected_conformal_bridge_summary.csv"
DETAIL_CSV = OUTDIR / "dss_projected_conformal_bridge_detail.csv"
DETAIL_JSON = OUTDIR / "dss_projected_conformal_bridge_detail.json"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    core_exclusion: float
    required: bool


def beta_samples(A: float, n_tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.arange(n_tau, dtype=float) / float(n_tau)
    beta = np.cos(2.0 * math.pi * tau) + np.sin(6.0 * math.pi * tau) / A
    b2 = beta * beta
    return b2, b2 * b2, b2 * b2 * b2


def dss_period_average_features(
    x: np.ndarray,
    b2: np.ndarray,
    b4: np.ndarray,
    b6: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return F, F', F'' for F(x)=<b^2/(1+b^2 x^2)>."""
    shape = x.shape
    xr = np.asarray(x, dtype=float).reshape(1, -1)
    den = 1.0 + b2[:, None] * xr * xr
    den2 = den * den
    den3 = den2 * den
    F = np.mean(b2[:, None] / den, axis=0)
    Fp = np.mean(-2.0 * b4[:, None] * xr / den2, axis=0)
    Fpp = np.mean(
        -2.0 * b4[:, None] / den2 + 8.0 * b6[:, None] * xr * xr / den3,
        axis=0,
    )
    return F.reshape(shape), Fp.reshape(shape), Fpp.reshape(shape)


def target_two_center_features(
    z: np.ndarray,
    D_sep: float,
    eps: float,
    a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Omega, d_z Omega, and d_z^2 Omega on the rho=0 axis."""

    def K(d: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(d * d + eps * eps)

    def Kp(d: np.ndarray) -> np.ndarray:
        return -d / np.power(d * d + eps * eps, 1.5)

    def Kpp(d: np.ndarray) -> np.ndarray:
        return (2.0 * d * d - eps * eps) / np.power(d * d + eps * eps, 2.5)

    d_plus = z - 0.5 * D_sep
    d_minus = z + 0.5 * D_sep
    omega = 1.0 + a * (K(d_plus) + K(d_minus))
    omega_p = a * (Kp(d_plus) + Kp(d_minus))
    omega_pp = a * (Kpp(d_plus) + Kpp(d_minus))
    return omega, omega_p, omega_pp


def dss_two_lobe_features(
    z: np.ndarray,
    R: float,
    D_sep: float,
    b2: np.ndarray,
    b4: np.ndarray,
    b6: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.zeros_like(z)
    phi_p = np.zeros_like(z)
    phi_pp = np.zeros_like(z)
    for center in (-0.5 * D_sep, 0.5 * D_sep):
        dz = z - center
        u = np.abs(dz) / R
        F, Fp, Fpp = dss_period_average_features(u, b2, b4, b6)
        phi += F
        phi_p += Fp * np.sign(dz) / R
        phi_pp += Fpp / (R * R)
    return phi, phi_p, phi_pp


def normalized_fit(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    T0, T1, T2 = target
    P0, P1, P2 = features
    s0 = max(float(np.max(np.abs(T0))), 1.0e-14)
    s1 = max(float(np.max(np.abs(T1))), 1.0e-14)
    s2 = max(float(np.max(np.abs(T2))), 1.0e-14)

    X = np.vstack(
        [
            np.column_stack([np.ones_like(P0) / s0, P0 / s0]),
            np.column_stack([np.zeros_like(P1), P1 / s1]),
            np.column_stack([np.zeros_like(P2), P2 / s2]),
        ]
    )
    y = np.concatenate([T0 / s0, T1 / s1, T2 / s2])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    C0 = coeffs[0] + coeffs[1] * P0
    C1 = coeffs[1] * P1
    C2 = coeffs[1] * P2
    e0 = np.max(np.abs(C0 - T0)) / s0
    e1 = np.max(np.abs(C1 - T1)) / s1
    e2 = np.max(np.abs(C2 - T2)) / s2
    rms = math.sqrt(
        float(
            np.mean(
                np.concatenate(
                    [
                        ((C0 - T0) / s0) ** 2,
                        ((C1 - T1) / s1) ** 2,
                        ((C2 - T2) / s2) ** 2,
                    ]
                )
            )
        )
    )
    metrics = {
        "c0": float(coeffs[0]),
        "c1": float(coeffs[1]),
        "rel_C0_sup": float(e0),
        "rel_C1_sup": float(e1),
        "rel_C2_sup": float(e2),
        "rel_C012_max": float(max(e0, e1, e2)),
        "rel_C012_rms": rms,
        "min_candidate_Omega": float(np.min(C0)),
        "max_candidate_Omega": float(np.max(C0)),
        "min_target_Omega": float(np.min(T0)),
        "max_target_Omega": float(np.max(T0)),
    }
    return coeffs, metrics


def analyze_window(
    spec: WindowSpec,
    policy: str,
    args: argparse.Namespace,
    b2: np.ndarray,
    b4: np.ndarray,
    b6: np.ndarray,
) -> dict[str, object]:
    left = -0.5 * args.D_sep + spec.core_exclusion
    right = 0.5 * args.D_sep - spec.core_exclusion
    if not left < right:
        raise ValueError(f"empty window {spec.name}: [{left}, {right}]")
    z = np.linspace(left, right, args.z_points)
    target = target_two_center_features(z, args.D_sep, args.eps, args.a)
    max_lobe_distance = max(
        float(np.max(np.abs(z - 0.5 * args.D_sep))),
        float(np.max(np.abs(z + 0.5 * args.D_sep))),
    )
    if policy == "domain_safe":
        R_min = max(args.R_min, max_lobe_distance)
    elif policy == "analytic_extension":
        R_min = args.R_min
    else:
        raise ValueError(f"unknown policy: {policy}")
    R_max = max(args.R_max, R_min)

    best: dict[str, object] | None = None
    for R in np.linspace(R_min, R_max, args.R_count):
        features = dss_two_lobe_features(z, float(R), args.D_sep, b2, b4, b6)
        _, metrics = normalized_fit(target, features)
        pass_C0 = metrics["rel_C0_sup"] <= args.tol_C0
        pass_C1 = metrics["rel_C1_sup"] <= args.tol_C1
        pass_C2 = metrics["rel_C2_sup"] <= args.tol_C2
        pass_positive = metrics["min_candidate_Omega"] > args.positive_floor
        score = metrics["rel_C012_max"]
        row = {
            "gate": "D3",
            "policy": policy,
            "window": spec.name,
            "required": spec.required,
            "core_exclusion": spec.core_exclusion,
            "D_sep": args.D_sep,
            "eps": args.eps,
            "a": args.a,
            "R": float(R),
            "R_scan_min": float(R_min),
            "R_scan_max": float(R_max),
            "max_lobe_distance": max_lobe_distance,
            "max_x_used": max_lobe_distance / float(R),
            "domain_safe": max_lobe_distance / float(R) <= 1.0 + 1.0e-12,
            **metrics,
            "pass_C0": pass_C0,
            "pass_C1": pass_C1,
            "pass_C2": pass_C2,
            "pass_positive": pass_positive,
            "window_pass": pass_C0 and pass_C1 and pass_C2 and pass_positive,
        }
        if best is None or float(score) < float(best["rel_C012_max"]):
            best = row
    assert best is not None
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=15.947599366191568, help="D1 solved A for Delta=1")
    parser.add_argument("--D-sep", type=float, default=10.0, help="PSLT two-center separation")
    parser.add_argument("--eps", type=float, default=0.2, help="Plummer regulator used by the PSLT target")
    parser.add_argument("--a", type=float, default=1.0, help="PSLT conformal-source amplitude")
    parser.add_argument("--tau-samples", type=int, default=1200, help="period samples for DSS averaging")
    parser.add_argument("--z-points", type=int, default=801, help="axial points per window")
    parser.add_argument("--R-min", type=float, default=0.1, help="minimum DSS lobe scale to scan")
    parser.add_argument("--R-max", type=float, default=30.0, help="maximum DSS lobe scale to scan")
    parser.add_argument("--R-count", type=int, default=180, help="number of scale samples")
    parser.add_argument("--tol-C0", type=float, default=5.0e-2, help="relative C0 sup-norm guard")
    parser.add_argument("--tol-C1", type=float, default=1.0e-1, help="relative C1 sup-norm guard")
    parser.add_argument("--tol-C2", type=float, default=2.0e-1, help="relative C2 sup-norm guard")
    parser.add_argument("--positive-floor", type=float, default=0.0, help="minimum allowed candidate Omega")
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

    b2, b4, b6 = beta_samples(args.A, args.tau_samples)
    windows = [
        WindowSpec("full_axial", 0.0, True),
        WindowSpec("core_excluded_2p5eps", 2.5 * args.eps, True),
        WindowSpec("shoulder_excluded_5eps", 5.0 * args.eps, True),
        WindowSpec("central_barrier_only_10eps", 10.0 * args.eps, False),
    ]
    rows = [
        analyze_window(spec, policy, args, b2, b4, b6)
        for policy in ("domain_safe", "analytic_extension")
        for spec in windows
    ]
    required_rows = [row for row in rows if bool(row["required"]) and row["policy"] == "domain_safe"]
    extension_required_rows = [
        row for row in rows if bool(row["required"]) and row["policy"] == "analytic_extension"
    ]
    required_pass = all(bool(row["window_pass"]) for row in required_rows)
    overall_verdict = (
        "D3_PROJECTED_CONFORMAL_C2_BRIDGE_PASSED"
        if required_pass
        else "D3_PROJECTED_CONFORMAL_C2_BRIDGE_FAILED_DIAGNOSTIC_ONLY"
    )

    summary_row = {
        "gate": "D3",
        "verdict": overall_verdict,
        "required_windows_pass": required_pass,
        "tol_C0": args.tol_C0,
        "tol_C1": args.tol_C1,
        "tol_C2": args.tol_C2,
        "D_sep": args.D_sep,
        "eps": args.eps,
        "a": args.a,
        "A": args.A,
        "tau_samples": args.tau_samples,
        "z_points": args.z_points,
        "R_min": args.R_min,
        "R_max": args.R_max,
        "R_count": args.R_count,
        "max_required_rel_C0_sup": max(float(row["rel_C0_sup"]) for row in required_rows),
        "max_required_rel_C1_sup": max(float(row["rel_C1_sup"]) for row in required_rows),
        "max_required_rel_C2_sup": max(float(row["rel_C2_sup"]) for row in required_rows),
        "max_required_rel_C012_max": max(float(row["rel_C012_max"]) for row in required_rows),
        "analytic_extension_required_windows_pass": all(
            bool(row["window_pass"]) for row in extension_required_rows
        ),
        "max_extension_required_rel_C0_sup": max(float(row["rel_C0_sup"]) for row in extension_required_rows),
        "max_extension_required_rel_C1_sup": max(float(row["rel_C1_sup"]) for row in extension_required_rows),
        "max_extension_required_rel_C2_sup": max(float(row["rel_C2_sup"]) for row in extension_required_rows),
        "max_extension_required_rel_C012_max": max(
            float(row["rel_C012_max"]) for row in extension_required_rows
        ),
    }

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row))
        writer.writeheader()
        writer.writerow(summary_row)

    with detail_path.open("w", newline="") as f:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detail = {
        "claim_firewall": {
            "baseline_change": False,
            "main_tex_change": False,
            "uses_master_equation": False,
            "interpretation": "necessary axial C2 bridge test only; domain_safe failure blocks D4 promotion",
        },
        "candidate_family": "Omega_DSS^R(z)=c0+c1[F(|z-D_sep/2|/R)+F(|z+D_sep/2|/R)]",
        "F_definition": "F(x)=<beta(tau)^2/(1+beta(tau)^2 x^2)>_tau",
        "mathematical_reason_for_C2": "delta V_eff contains delta Omega and nabla^2 delta Omega",
        "summary": summary_row,
        "best_windows": rows,
    }
    with json_path.open("w") as f:
        json.dump(detail, f, indent=2)
        f.write("\n")

    print(overall_verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"detail_json={json_path}")
    for row in rows:
        print(
            "policy={policy} window={window} required={required} pass={window_pass} R={R:.8g} "
            "relC0={rel_C0_sup:.6g} relC1={rel_C1_sup:.6g} relC2={rel_C2_sup:.6g}".format(**row)
        )


if __name__ == "__main__":
    main()
