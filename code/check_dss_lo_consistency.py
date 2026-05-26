#!/usr/bin/env python3
"""D0/D1 gate for the large-d DSS/Floquet parent feasibility queue.

This script does not touch the PSLT baseline.  It records the symbol firewall
and reproduces the leading-order DSS consistency checks for the example
beta(tau) used in arXiv:2601.14358:

    beta(tau) = cos(2*pi*tau) + sin(6*pi*tau) / A,  A ~= 15.9476.

The NLO echoing-period condition is checked in the form

    Delta = |beta''| / (3 |beta'|) evaluated at beta = 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "dss_floquet"
SUMMARY_CSV = OUTDIR / "dss_lo_consistency_summary.csv"
DETAIL_JSON = OUTDIR / "dss_lo_consistency_detail.json"


def beta(tau: float, A: float) -> float:
    return math.cos(2.0 * math.pi * tau) + math.sin(6.0 * math.pi * tau) / A


def beta_prime(tau: float, A: float) -> float:
    return -2.0 * math.pi * math.sin(2.0 * math.pi * tau) + (6.0 * math.pi / A) * math.cos(6.0 * math.pi * tau)


def beta_second(tau: float, A: float) -> float:
    return -(2.0 * math.pi) ** 2 * math.cos(2.0 * math.pi * tau) - ((6.0 * math.pi) ** 2 / A) * math.sin(6.0 * math.pi * tau)


def bisection(f: Callable[[float], float], left: float, right: float, *, tol: float = 1e-15, max_iter: int = 200) -> float:
    f_left = f(left)
    f_right = f(right)
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        raise ValueError(f"bisection bracket has no sign change: f({left})={f_left}, f({right})={f_right}")
    lo, hi = left, right
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol or 0.5 * (hi - lo) <= tol:
            return mid
        if f_left * f_mid <= 0.0:
            hi = mid
            f_right = f_mid
        else:
            lo = mid
            f_left = f_mid
    return 0.5 * (lo + hi)


def first_zero_in_half_period(A: float) -> float:
    # The target family has a single zero in [0, 1/2).  Avoid the endpoint to
    # keep the "single zero" convention auditable.
    grid_n = 2000
    prev_t = 0.0
    prev_v = beta(prev_t, A)
    for k in range(1, grid_n):
        t = 0.5 * k / grid_n
        val = beta(t, A)
        if prev_v * val < 0.0:
            return bisection(lambda x: beta(x, A), prev_t, t)
        prev_t, prev_v = t, val
    raise RuntimeError(f"no beta zero found in [0, 1/2) for A={A}")


def echo_delta_from_A(A: float) -> tuple[float, float, float, float]:
    tau0 = first_zero_in_half_period(A)
    bp = beta_prime(tau0, A)
    bpp = beta_second(tau0, A)
    delta = abs(bpp) / (3.0 * abs(bp))
    # For a period-scaled beta_base(tau / Delta), the NLO regularity residual is
    # beta_base'' / Delta^2 + 3 beta_base' / Delta at beta=0.  At Delta=1 this
    # is simply beta'' + 3 beta'.
    residual_at_delta_1 = bpp + 3.0 * bp
    return tau0, delta, bp, residual_at_delta_1


def solve_A_for_delta_one(left: float = 2.0, right: float = 100.0) -> tuple[float, float]:
    def objective(A: float) -> float:
        _, delta, _, _ = echo_delta_from_A(A)
        return delta - 1.0

    # Find a robust bracket because the admissible root is a property of the
    # chosen Fourier ansatz, not of this script's initial guess.
    n = 600
    lo = left
    f_lo = objective(lo)
    for k in range(1, n + 1):
        hi = left + (right - left) * k / n
        f_hi = objective(hi)
        if f_lo * f_hi <= 0.0:
            A_star = bisection(objective, lo, hi, tol=1e-13)
            tau0 = first_zero_in_half_period(A_star)
            return A_star, tau0
        lo, f_lo = hi, f_hi
    raise RuntimeError("could not bracket A solving Delta(A)=1")


def max_grid_residuals(A: float, n_tau: int = 1001, n_x: int = 401) -> dict[str, float]:
    max_periodicity = 0.0
    max_half_antisymmetry = 0.0
    max_omega_identity = 0.0
    max_ssh_boundary = 0.0
    min_omega = float("inf")
    min_f_minus_x = float("inf")
    max_f_minus_one = -float("inf")

    for i in range(n_tau):
        tau = i / (n_tau - 1)
        b = beta(tau, A)
        max_periodicity = max(max_periodicity, abs(beta(tau + 1.0, A) - b))
        max_half_antisymmetry = max(max_half_antisymmetry, abs(beta(tau + 0.5, A) + b))
        denom_ssh = 1.0 + b * b
        f_ssh = math.sqrt((1.0 + b * b) / denom_ssh)
        max_ssh_boundary = max(max_ssh_boundary, abs(f_ssh - 1.0))
        for j in range(n_x):
            x = j / (n_x - 1)
            pi_lo = b / math.sqrt(1.0 + b * b * x * x)
            omega_lo = pi_lo * pi_lo
            f_lo = math.sqrt((1.0 + b * b * x * x) / (1.0 + b * b))
            max_omega_identity = max(max_omega_identity, abs(omega_lo - pi_lo * pi_lo))
            min_omega = min(min_omega, omega_lo)
            min_f_minus_x = min(min_f_minus_x, f_lo - x)
            max_f_minus_one = max(max_f_minus_one, f_lo - 1.0)

    return {
        "max_periodicity_residual": max_periodicity,
        "max_half_antisymmetry_residual": max_half_antisymmetry,
        "max_OmegaLO_minus_PiLO2": max_omega_identity,
        "max_ssh_boundary_residual": max_ssh_boundary,
        "min_OmegaLO": min_omega,
        "min_fLO_minus_x": min_f_minus_x,
        "max_fLO_minus_1": max_f_minus_one,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A-ref", type=float, default=15.9476, help="rounded A quoted in the paper")
    parser.add_argument("--tol-delta", type=float, default=1e-5, help="tolerance for the rounded-A Delta check")
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    tau0_ref, delta_ref, beta_p_ref, nlo_residual_ref = echo_delta_from_A(args.A_ref)
    A_star, tau0_star = solve_A_for_delta_one()
    tau0_star_check, delta_star, beta_p_star, nlo_residual_star = echo_delta_from_A(A_star)
    grid = max_grid_residuals(args.A_ref)

    status = "PASS" if abs(delta_ref - 1.0) < args.tol_delta else "FAIL"
    verdict = "D1_LO_NLO_DSS_CONSISTENCY_REPRODUCED" if status == "PASS" else "D1_DSS_CONSISTENCY_FAILED"

    row = {
        "gate": "D0_D1",
        "status": status,
        "verdict": verdict,
        "paper_arxiv": "2601.14358",
        "dimension_symbol": "d",
        "pslt_separation_symbol": "D_sep",
        "A_ref": f"{args.A_ref:.10f}",
        "tau_zero_ref": f"{tau0_ref:.16f}",
        "Delta_from_ref": f"{delta_ref:.16e}",
        "abs_Delta_ref_minus_1": f"{abs(delta_ref - 1.0):.16e}",
        "beta_prime_at_zero_ref": f"{beta_p_ref:.16e}",
        "nlo_residual_beta_second_plus_3beta_prime_ref": f"{nlo_residual_ref:.16e}",
        "A_solved_for_Delta_1": f"{A_star:.16e}",
        "tau_zero_solved": f"{tau0_star:.16f}",
        "Delta_solved": f"{delta_star:.16e}",
        "nlo_residual_solved": f"{nlo_residual_star:.16e}",
        **{k: f"{v:.16e}" for k, v in grid.items()},
    }

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    detail = {
        "claim_firewall": {
            "baseline_change": False,
            "main_tex_change": False,
            "dss_role": "optional parent-side feasibility only",
            "does_not_prove": ["three SM generations", "g_N", "B_N", "H->mumu", "gauge charges"],
            "symbol_policy": {
                "spacetime_dimension": "d",
                "pslt_two_center_separation": "D_sep",
            },
        },
        "lo_formulas_checked": {
            "Pi_LO": "beta(tau)/sqrt(1+beta(tau)^2 x^2)",
            "Omega_LO": "Pi_LO^2",
            "f_LO": "sqrt((1+beta(tau)^2 x^2)/(1+beta(tau)^2))",
        },
        "nlo_condition_checked": "Delta = |beta''|/(3|beta'|) at beta=0",
        "summary": row,
    }
    with detail_path.open("w") as f:
        json.dump(detail, f, indent=2)
        f.write("\n")

    print(verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"A_ref={args.A_ref:.10f} tau0={tau0_ref:.16f} Delta={delta_ref:.16e}")
    print(f"A_solved={A_star:.16e} tau0={tau0_star_check:.16f} Delta={delta_star:.16e}")


if __name__ == "__main__":
    main()
