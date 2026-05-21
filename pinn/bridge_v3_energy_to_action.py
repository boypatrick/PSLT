#!/usr/bin/env python3
"""V3.1 energy-to-action bridge for the PINN spectral export.

The bridge computes a central WKB action from the action-derived axial
potential

    U(0,z;D) = m0^2 (Omega^2 - 1) + (1 - 6 xi) Delta Omega / Omega

by integrating over the connected component of {U(z;D) > E} closest to the
central barrier.  It also cross-checks the same adapter against the canonical
deterministic single-track artifact.

This is intentionally an adapter audit, not a replacement for the deterministic
single-track or finite-volume certificates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRAL = ROOT / "pinn" / "v3_downstream_spectral_table.csv"
DEFAULT_TRUE = ROOT / "output" / "true_single_track" / "true_results.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectral-table", type=Path, default=DEFAULT_SPECTRAL)
    p.add_argument("--true-artifact", type=Path, default=DEFAULT_TRUE)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--m0", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--rho-axis", type=float, default=0.02)
    p.add_argument("--n-points", type=int, default=3000)
    p.add_argument("--out-prefix", default="v3_energy_to_action")
    return p.parse_args()


def omega_func(rho: np.ndarray | float, z: np.ndarray | float, D: float, a: float, eps: float):
    r1 = np.sqrt(rho**2 + (z - D / 2.0) ** 2 + eps**2)
    r2 = np.sqrt(rho**2 + (z + D / 2.0) ** 2 + eps**2)
    return 1.0 + a * (1.0 / r1 + 1.0 / r2)


def laplacian_omega_func(rho: np.ndarray | float, z: np.ndarray | float, D: float, a: float, eps: float):
    r1 = np.sqrt(rho**2 + (z - D / 2.0) ** 2 + eps**2)
    r2 = np.sqrt(rho**2 + (z + D / 2.0) ** 2 + eps**2)
    return a * (-3.0 * eps**2 / (r1**5) - 3.0 * eps**2 / (r2**5))


def u_axis(z: np.ndarray, D: float, *, rho: float, a: float, eps: float, m0: float, xi: float):
    om = omega_func(rho, z, D, a, eps)
    lap = laplacian_omega_func(rho, z, D, a, eps)
    return m0 * m0 * (om * om - 1.0) + (1.0 - 6.0 * xi) * lap / om


def positive_components(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            comps.append((start, i - 1))
            start = None
    if start is not None:
        comps.append((start, len(mask) - 1))
    return comps


def simpson_uniform(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson integral on an almost-uniform grid.

    If the number of samples is even, use Simpson on the first n-1 samples and
    one trapezoid on the final interval.  This avoids adding SciPy as a PINN
    sandbox dependency.
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n == 2:
        return float(np.trapz(y, x=x))
    h = float((x[-1] - x[0]) / (n - 1))
    if n % 2 == 1:
        return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])))
    simpson_part = h / 3.0 * (y[0] + y[-2] + 4.0 * np.sum(y[1:-2:2]) + 2.0 * np.sum(y[2:-3:2]))
    trap_part = 0.5 * h * (y[-2] + y[-1])
    return float(simpson_part + trap_part)


def central_action(D: float, E: float, *, rho: float, a: float, eps: float, m0: float, xi: float, n_points: int):
    L_z = max(18.0, D + 8.0)
    z = np.linspace(-L_z, L_z, n_points)
    U = u_axis(z, D, rho=rho, a=a, eps=eps, m0=m0, xi=xi)
    diff = U - E
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    # The canonical single-track artifact uses the inner two turning crossings
    # for negative-energy bound states.  Keep this branch exactly compatible so
    # the V3.1 adapter is cross-checked against the deterministic artifact
    # before we use it diagnostically on the PINN spectral table.
    if E < 0.0 and len(sign_changes) >= 4:
        i = int(sign_changes[1])
        j = int(sign_changes[2])
        zz = z[i : j + 1]
        integrand = np.sqrt(np.maximum(U[i : j + 1] - E, 0.0))
        S = simpson_uniform(integrand, zz)
        return {
            "S": S,
            "n_turning": int(len(sign_changes)),
            "status": "CANONICAL_NEGATIVE_BOUND_BARRIER",
            "z_left": float(z[i]),
            "z_right": float(z[j]),
            "component_width": float(z[j] - z[i]),
            "U_center_minus_E": float(u_axis(np.array([0.0]), D, rho=rho, a=a, eps=eps, m0=m0, xi=xi)[0] - E),
        }

    mask = diff > 0.0
    comps = positive_components(mask)
    if not comps:
        return {
            "S": 0.0,
            "n_turning": int(len(sign_changes)),
            "status": "NO_FORBIDDEN_COMPONENT",
            "z_left": math.nan,
            "z_right": math.nan,
            "component_width": 0.0,
            "U_center_minus_E": float(u_axis(np.array([0.0]), D, rho=rho, a=a, eps=eps, m0=m0, xi=xi)[0] - E),
        }

    def score(comp: tuple[int, int]):
        i, j = comp
        mid = 0.5 * (z[i] + z[j])
        contains_center = z[i] <= 0.0 <= z[j]
        # Prefer the central barrier component; if no component contains zero,
        # choose the closest forbidden island.
        return (0 if contains_center else 1, abs(mid), -(j - i))

    i, j = sorted(comps, key=score)[0]
    if j <= i:
        return {
            "S": 0.0,
            "n_turning": int(len(sign_changes)),
            "status": "DEGENERATE_COMPONENT",
            "z_left": float(z[i]),
            "z_right": float(z[j]),
            "component_width": float(z[j] - z[i]),
            "U_center_minus_E": float(u_axis(np.array([0.0]), D, rho=rho, a=a, eps=eps, m0=m0, xi=xi)[0] - E),
        }
    zz = z[i : j + 1]
    integrand = np.sqrt(np.maximum(U[i : j + 1] - E, 0.0))
    S = simpson_uniform(integrand, zz)
    status = "CENTRAL_BARRIER" if z[i] <= 0.0 <= z[j] else "NEAREST_FORBIDDEN_ISLAND"
    return {
        "S": S,
        "n_turning": int(len(sign_changes)),
        "status": status,
        "z_left": float(z[i]),
        "z_right": float(z[j]),
        "component_width": float(z[j] - z[i]),
        "U_center_minus_E": float(u_axis(np.array([0.0]), D, rho=rho, a=a, eps=eps, m0=m0, xi=xi)[0] - E),
    }


def read_spectral_table(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_true_artifact(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for i, D in enumerate(data["D"]):
        rows.append({
            "D": float(D),
            "E": float(data["E_bound"][i]),
            "S_ref": float(data["S_N"][i]),
            "n_turning_ref": int(data["n_turning"][i]),
        })
    return rows


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spectral_rows = read_spectral_table(args.spectral_table)
    true_rows = read_true_artifact(args.true_artifact)

    true_detail = []
    for row in true_rows:
        result = central_action(
            row["D"],
            row["E"],
            rho=args.rho_axis,
            a=args.a,
            eps=args.eps,
            m0=args.m0,
            xi=args.xi,
            n_points=args.n_points,
        )
        S = float(result["S"])
        S_ref = float(row["S_ref"])
        true_detail.append({
            "D": row["D"],
            "E_artifact": row["E"],
            "S_adapter": S,
            "S_ref": S_ref,
            "S_abs_error": abs(S - S_ref),
            "S_rel_error": abs(S - S_ref) / max(abs(S_ref), 1.0e-12),
            "n_turning_adapter": result["n_turning"],
            "n_turning_ref": row["n_turning_ref"],
            "status": result["status"],
            "z_left": result["z_left"],
            "z_right": result["z_right"],
        })

    bridge_rows = []
    modes = [0, 1, 2]
    for row in spectral_rows:
        D = float(row["D"])
        for mode in modes:
            E = float(row[f"E{mode}_ritz"])
            result = central_action(
                D,
                E,
                rho=args.rho_axis,
                a=args.a,
                eps=args.eps,
                m0=args.m0,
                xi=args.xi,
                n_points=args.n_points,
            )
            S = float(result["S"])
            bridge_rows.append({
                "D": D,
                "mode": mode,
                "E_ritz": E,
                "omega_ritz": float(row[f"omega{mode}_ritz"]),
                "S_adapter": S,
                "r_eta1": math.exp(-2.0 * S) if S > 0.0 else 1.0,
                "n_turning": result["n_turning"],
                "status": result["status"],
                "z_left": result["z_left"],
                "z_right": result["z_right"],
                "component_width": result["component_width"],
                "U_center_minus_E": result["U_center_minus_E"],
                "source_spectral_suspicious": row.get("suspicious", "False"),
            })

    true_max_abs = max(r["S_abs_error"] for r in true_detail)
    true_max_rel = max(r["S_rel_error"] for r in true_detail)
    status_counts: dict[str, int] = {}
    mode_status_counts: dict[str, dict[str, int]] = {}
    central_D_ranges: dict[str, dict[str, float | None]] = {}
    for row in bridge_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        mode_key = str(row["mode"])
        mode_status_counts.setdefault(mode_key, {})
        mode_status_counts[mode_key][row["status"]] = mode_status_counts[mode_key].get(row["status"], 0) + 1
    promoted_rows = [r for r in bridge_rows if r["status"] == "CENTRAL_BARRIER"]
    positive_action_rows = [r for r in bridge_rows if float(r["S_adapter"]) > 0.0]
    for mode in modes:
        ds = [float(r["D"]) for r in bridge_rows if int(r["mode"]) == mode and r["status"] == "CENTRAL_BARRIER"]
        central_D_ranges[str(mode)] = {
            "min_D": min(ds) if ds else None,
            "max_D": max(ds) if ds else None,
        }

    summary = {
        "target": "PINN V3.1 energy-to-action bridge",
        "status": "DIAGNOSTIC_BRIDGE_ONLY",
        "spectral_table": str(args.spectral_table.relative_to(ROOT)),
        "true_artifact": str(args.true_artifact.relative_to(ROOT)),
        "rho_axis": args.rho_axis,
        "n_points": args.n_points,
        "true_crosscheck_max_abs_S_error": true_max_abs,
        "true_crosscheck_max_rel_S_error": true_max_rel,
        "true_crosscheck_pass": bool(true_max_abs < 5.0e-3 and true_max_rel < 5.0e-4),
        "n_bridge_rows": len(bridge_rows),
        "status_counts": status_counts,
        "n_positive_action_rows": len(positive_action_rows),
        "n_central_barrier_rows": len(promoted_rows),
        "mode_status_counts": mode_status_counts,
        "central_barrier_D_ranges_by_mode": central_D_ranges,
        "policy": (
            "Adapter is validated against the canonical single-track S artifact, "
            "but the PINN V3.0 self-adjoint spectrum is not the same spectral "
            "object as the legacy negative-energy single-track artifact.  Do not "
            "promote to Gamma_N or P_N without a separate operator-identification gate."
        ),
    }

    prefix = ROOT / "pinn" / args.out_prefix
    write_csv(prefix.with_name(prefix.name + "_true_crosscheck.csv"), true_detail)
    write_csv(prefix.with_name(prefix.name + "_detail.csv"), bridge_rows)
    prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
