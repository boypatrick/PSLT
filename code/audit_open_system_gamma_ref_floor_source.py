#!/usr/bin/env python3
"""
Audit source for the gamma_ref floor on the open-system micro bridge.

The current open-system observable chain uses gamma_ref only through

    chi_eff = 2 * gamma_mix * Cmax / gamma_ref.

This script verifies that on the exported exact micro bridge,

    gamma_ref = omega1^11 = tau_env^-11,

so the gamma_ref floor is equivalent to either a first-mode floor on omega1
or a ceiling on tau_env.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def load_bridge_map() -> pd.DataFrame:
    for path in [
        OUTDIR / "chi_open_system_micro_bridge_map.csv",
        PAPER_DIR / "chi_open_system_micro_bridge_map.csv",
    ]:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Could not find chi_open_system_micro_bridge_map.csv")


def main() -> None:
    df = load_bridge_map().copy()
    df["gamma_ref_from_omega11"] = df["omega1"].astype(float) ** 11
    df["gamma_ref_from_tauceil"] = df["tau_env"].astype(float) ** (-11)
    df["tau_env_from_gaplock"] = 1.0 / df["omega1"].astype(float)

    df["gamma_ref_omega11_residual_abs"] = (
        df["gamma_ref"].astype(float) - df["gamma_ref_from_omega11"]
    ).abs()
    df["gamma_ref_tau_residual_abs"] = (
        df["gamma_ref"].astype(float) - df["gamma_ref_from_tauceil"]
    ).abs()
    df["tau_gaplock_residual_abs"] = (
        df["tau_env"].astype(float) - df["tau_env_from_gaplock"]
    ).abs()

    out_detail = OUTDIR / "chi_open_system_gamma_ref_floor_source_detail.csv"
    out_summary = OUTDIR / "chi_open_system_gamma_ref_floor_source_summary.csv"

    df.to_csv(out_detail, index=False)

    gamma_ref = df["gamma_ref"].astype(float)
    omega1 = df["omega1"].astype(float)
    tau_env = df["tau_env"].astype(float)

    omega1_floor = float(omega1.min())
    tau_env_ceiling = float(tau_env.max())
    gamma_ref_floor = float(gamma_ref.min())

    summary = pd.DataFrame(
        [
            {
                "max_abs_gamma_ref_omega11_residual": float(df["gamma_ref_omega11_residual_abs"].max()),
                "max_abs_gamma_ref_tau_residual": float(df["gamma_ref_tau_residual_abs"].max()),
                "max_abs_tau_gaplock_residual": float(df["tau_gaplock_residual_abs"].max()),
                "omega1_floor": omega1_floor,
                "omega1_floor_D": float(df.loc[omega1.idxmin(), "D"]),
                "tau_env_ceiling": tau_env_ceiling,
                "tau_env_ceiling_D": float(df.loc[tau_env.idxmax(), "D"]),
                "actual_gamma_ref_floor": gamma_ref_floor,
                "actual_gamma_ref_floor_D": float(df.loc[gamma_ref.idxmin(), "D"]),
                "gamma_ref_floor_from_omega11": float(omega1_floor**11),
                "gamma_ref_floor_from_tau_ceiling": float(tau_env_ceiling**(-11)),
                "actual_over_omega11_floor": float(gamma_ref_floor / max(omega1_floor**11, 1e-300)),
                "actual_over_tau_ceiling_floor": float(gamma_ref_floor / max(tau_env_ceiling**(-11), 1e-300)),
            }
        ]
    )
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_detail}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
