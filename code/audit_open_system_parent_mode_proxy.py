#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_parent_proxy_detail(parent_df: pd.DataFrame) -> pd.DataFrame:
    detail = parent_df.copy()
    detail["parent_proxy_x"] = detail["parent_blockdet_k11"] - 1.0
    detail["parent_proxy_g_schur"] = detail["parent_blockdet_schur_norm"]
    detail["parent_proxy_y"] = detail["parent_proxy_g_schur"] - 1.0
    detail["parent_proxy_mix_norm_sq"] = (
        detail["parent_blockdet_c_parent"] ** 2 / np.maximum(detail["parent_blockdet_k_bg"], 1e-30)
    )
    detail["parent_proxy_rank1_residual_abs"] = np.abs(
        detail["parent_proxy_mix_norm_sq"] - detail["parent_proxy_x"] * detail["parent_proxy_y"]
    )

    k11 = detail["parent_blockdet_k11"].to_numpy(dtype=float)
    g_schur = detail["parent_proxy_g_schur"].to_numpy(dtype=float)
    k12_norm = detail["parent_blockdet_c_parent"].to_numpy(dtype=float) / np.sqrt(
        np.maximum(detail["parent_blockdet_k_bg"].to_numpy(dtype=float), 1e-30)
    )
    trace = k11 + g_schur
    radical = np.sqrt(np.maximum((k11 - g_schur) ** 2 + 4.0 * k12_norm**2, 0.0))
    lam1_direct = 0.5 * (trace - radical)
    lam2_direct = 0.5 * (trace + radical)
    lam2_formula = 1.0 + detail["parent_proxy_x"].to_numpy(dtype=float) + detail["parent_proxy_y"].to_numpy(dtype=float)

    detail["parent_proxy_lambda1"] = lam1_direct
    detail["parent_proxy_lambda2"] = lam2_direct
    detail["parent_proxy_lambda1_residual_abs"] = np.abs(lam1_direct - 1.0)
    detail["parent_proxy_lambda2_formula"] = lam2_formula
    detail["parent_proxy_lambda2_residual_abs"] = np.abs(lam2_direct - lam2_formula)
    detail["parent_proxy_g"] = np.sqrt(np.maximum(lam2_direct, 0.0))
    detail["parent_proxy_active"] = detail["parent_blockdet_action"] > 1e-12
    return detail


def attach_open_system_window(
    proxy_df: pd.DataFrame,
    open_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = proxy_df.merge(
        open_df[["D", "omega1", "delta_micro", "tau_env", "Szz_0", "Sxx_delta"]],
        on="D",
        how="left",
    )
    merged["parent_proxy_Omega1"] = merged["omega1"]
    merged["parent_proxy_Omega2"] = merged["omega1"] * merged["parent_proxy_g"]
    merged["parent_proxy_r_on_shell"] = merged["delta_micro"] / np.maximum(merged["omega1"], 1e-30)
    merged["parent_proxy_q_on_shell"] = merged["Sxx_delta"] / np.maximum(merged["Szz_0"], 1e-30)
    merged["parent_proxy_q_single_pole"] = lorentz_factor(merged["parent_proxy_r_on_shell"].to_numpy(dtype=float))
    merged["parent_proxy_delta_q_abs"] = np.abs(
        merged["parent_proxy_q_on_shell"] - merged["parent_proxy_q_single_pole"]
    )
    merged["parent_proxy_gap_separator"] = lorentz_factor(
        (merged["parent_proxy_r_on_shell"] / np.maximum(merged["parent_proxy_g"], 1e-30)).to_numpy(dtype=float)
    ) - merged["parent_proxy_q_single_pole"].to_numpy(dtype=float)
    merged["parent_proxy_eps_delta_factor"] = (
        1.0 + merged["parent_proxy_r_on_shell"] ** 2
    ) / (
        merged["parent_proxy_g"] ** 2 + merged["parent_proxy_r_on_shell"] ** 2
    )
    return merged


def build_tolerance_detail(
    merged: pd.DataFrame,
    delta_q_tolerances: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    active = merged[merged["parent_proxy_active"]].copy()
    active = active[np.isfinite(active["parent_proxy_gap_separator"])].copy()
    for row in active.itertuples(index=False):
        separator = float(row.parent_proxy_gap_separator)
        eps_factor = float(row.parent_proxy_eps_delta_factor)
        for tol in delta_q_tolerances:
            if tol < separator:
                tail_upper = tol / separator
                eps0_upper = tol / (separator - tol)
                eps_delta_upper = eps0_upper * eps_factor
            else:
                tail_upper = np.nan
                eps0_upper = np.nan
                eps_delta_upper = np.nan
            rows.append(
                {
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "parent_proxy_g": float(row.parent_proxy_g),
                    "parent_proxy_gap_separator": separator,
                    "parent_proxy_eps_delta_factor": eps_factor,
                    "delta_q_tol": float(tol),
                    "tail_mass_upper": tail_upper,
                    "eps0_upper": eps0_upper,
                    "eps_delta_upper": eps_delta_upper,
                }
            )
    return pd.DataFrame(rows)


def summarize_parent_proxy(merged: pd.DataFrame) -> pd.DataFrame:
    all_rows = merged.copy()
    active = merged[merged["parent_proxy_active"]].copy()
    rows: list[dict[str, float]] = []
    for label, sub in [("all_rows", all_rows), ("active_rows", active)]:
        row = {
            "subset": label,
            "n_rows": int(len(sub)),
            "max_rank1_residual_abs": float(np.max(np.abs(sub["parent_proxy_rank1_residual_abs"]))),
            "max_lambda1_residual_abs": float(np.max(np.abs(sub["parent_proxy_lambda1_residual_abs"]))),
            "max_lambda2_residual_abs": float(np.max(np.abs(sub["parent_proxy_lambda2_residual_abs"]))),
            "g_proxy_min": float(sub["parent_proxy_g"].min()),
            "g_proxy_p10": float(sub["parent_proxy_g"].quantile(0.10)),
            "g_proxy_p50": float(sub["parent_proxy_g"].quantile(0.50)),
            "g_proxy_p90": float(sub["parent_proxy_g"].quantile(0.90)),
            "g_proxy_max": float(sub["parent_proxy_g"].max()),
        }
        if sub["parent_proxy_gap_separator"].notna().any():
            row.update(
                {
                    "gap_separator_min": float(sub["parent_proxy_gap_separator"].min()),
                    "gap_separator_max": float(sub["parent_proxy_gap_separator"].max()),
                    "eps_delta_factor_min": float(sub["parent_proxy_eps_delta_factor"].min()),
                    "eps_delta_factor_max": float(sub["parent_proxy_eps_delta_factor"].max()),
                    "max_delta_q_abs": float(sub["parent_proxy_delta_q_abs"].max()),
                }
            )
        else:
            row.update(
                {
                    "gap_separator_min": np.nan,
                    "gap_separator_max": np.nan,
                    "eps_delta_factor_min": np.nan,
                    "eps_delta_factor_max": np.nan,
                    "max_delta_q_abs": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_eta(merged: pd.DataFrame) -> pd.DataFrame:
    active = merged[merged["parent_proxy_active"]].copy()
    summary = (
        active.groupby("eta", as_index=False)
        .agg(
            D_min=("D", "min"),
            D_max=("D", "max"),
            g_proxy_min=("parent_proxy_g", "min"),
            g_proxy_p50=("parent_proxy_g", "median"),
            g_proxy_max=("parent_proxy_g", "max"),
            gap_separator_min=("parent_proxy_gap_separator", "min"),
            gap_separator_max=("parent_proxy_gap_separator", "max"),
        )
    )
    return summary


def summarize_tolerances(tol_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        tol_df.groupby("delta_q_tol", as_index=False)
        .agg(
            g_proxy_min=("parent_proxy_g", "min"),
            g_proxy_p50=("parent_proxy_g", "median"),
            g_proxy_max=("parent_proxy_g", "max"),
            gap_separator_min=("parent_proxy_gap_separator", "min"),
            gap_separator_max=("parent_proxy_gap_separator", "max"),
            tail_mass_upper_max=("tail_mass_upper", "max"),
            eps0_upper_max=("eps0_upper", "max"),
            eps_delta_upper_max=("eps_delta_upper", "max"),
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a first parent-side mode-gap proxy from the projected EYMH "
            "complement block and connect it to the open-system first-mode theorem target."
        )
    )
    parser.add_argument(
        "--parent-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_blockdet_audit_map_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--open-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
        ),
    )
    parser.add_argument(
        "--delta-q-tols",
        default="1e-3,5e-3,1e-2",
        help="Comma-separated hypothetical on-shell ratio defects for the proxy gap audit.",
    )
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_parent_mode_proxy_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_parent_mode_proxy_summary.csv"
        ),
    )
    parser.add_argument(
        "--eta-summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_parent_mode_proxy_eta_summary.csv"
        ),
    )
    parser.add_argument(
        "--tolerance-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_parent_mode_proxy_tolerance_summary.csv"
        ),
    )
    args = parser.parse_args()

    parent_df = pd.read_csv(args.parent_input)
    open_df = pd.read_csv(args.open_input)
    delta_q_tolerances = parse_csv_floats(args.delta_q_tols)

    detail = build_parent_proxy_detail(parent_df)
    detail = attach_open_system_window(detail, open_df)
    tol_df = build_tolerance_detail(detail, delta_q_tolerances)

    summary = summarize_parent_proxy(detail)
    eta_summary = summarize_by_eta(detail)
    tolerance_summary = summarize_tolerances(tol_df)

    outputs = [
        args.detail_output,
        args.summary_output,
        args.eta_summary_output,
        args.tolerance_output,
    ]
    for output in outputs:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    eta_summary.to_csv(args.eta_summary_output, index=False)
    tolerance_summary.to_csv(args.tolerance_output, index=False)

    print(summary.to_string(index=False))
    print()
    print(eta_summary.to_string(index=False))
    print()
    print(tolerance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
