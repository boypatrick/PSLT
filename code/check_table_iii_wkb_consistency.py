#!/usr/bin/env python3
"""Check and optionally regenerate the Table III WKB consistency row set.

Canonical source:
    output/true_single_track/true_results.json

The paper table labels the last numerical column as r_1 = exp(-2 S_1).
This script enforces that the displayed table values are generated from the
same canonical S_1 values, up to the declared display-rounding tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "output" / "true_single_track" / "true_results.json"
DEFAULT_TABLE = ROOT / "paper" / "true_unified_table.tex"

ROW_RE = re.compile(
    r"^\s*(?P<D>\d+(?:\.\d+)?)\s*&\s*"
    r"\$?(?P<E>[-+0-9.]+)\$?\s*&\s*"
    r"(?P<omega>[-+0-9.]+)\s*&\s*"
    r"(?P<S>[-+0-9.]+)\s*&\s*"
    r"\$(?P<mant>[-+0-9.]+)\s*\\times\s*10\^\{(?P<exp>[-+0-9]+)\}\$\s*&\s*"
    r"(?P<tp>\d+)\s*&\s*(?P<nbound>\d+)\s*\\\\"
)


def load_artifact(path: Path) -> list[dict[str, float | int]]:
    data = json.loads(path.read_text())
    required = ["D", "E_bound", "omega", "S_N", "n_turning", "n_bound"]
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"{path} is missing required keys: {missing}")

    n = len(data["D"])
    rows = []
    for key in required:
        if len(data[key]) != n:
            raise SystemExit(f"{path}:{key} has length {len(data[key])}, expected {n}")

    for i in range(n):
        rows.append(
            {
                "D": float(data["D"][i]),
                "E": float(data["E_bound"][i]),
                "omega": float(data["omega"][i]),
                "S": float(data["S_N"][i]),
                "tp": int(data["n_turning"][i]),
                "nbound": int(data["n_bound"][i]),
            }
        )
    return rows


def sci_tex(value: float, sig: int = 4) -> str:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Expected a positive finite value, got {value!r}")
    exponent = math.floor(math.log10(value))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.{sig}g} \times 10^{{{exponent}}}$"


def table_text(rows: list[dict[str, float | int]], artifact_path: Path) -> str:
    rel_artifact = artifact_path.relative_to(ROOT)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Single-track results from action-derived $V_{\rm eff} = m_0^2\Omega^2 + (1-6\xi)\Omega^{-1}\nabla^2\Omega$. All quantities are computed from the same potential. $E_1 = \omega_1^2 - m_0^2 < 0$ indicates bound states. Parameters: $a = 1$, $\varepsilon = 0.2$, $m_0 = 1$, $\xi = 0$.}",
        r"\label{tab:true_unified}",
        rf"% Generated from {rel_artifact} by code/check_table_iii_wkb_consistency.py.",
        r"\begin{tabular}{ccccccc}",
        r"\hline\hline",
        r"$D$ & $E_1$ & $\omega_1$ & $S_1$ & $r_1 = e^{-2S_1}$ & tp & $n_{\rm bound}$ \\",
        r"\hline",
    ]

    for row in rows:
        D = int(round(float(row["D"])))
        E = float(row["E"])
        omega = float(row["omega"])
        S = float(row["S"])
        r_value = math.exp(-2.0 * S)
        tp = int(row["tp"])
        nbound = int(row["nbound"])
        lines.append(
            f"{D}  & ${E:.4f}$ & {omega:.4f} & {S:.4f} & "
            f"{sci_tex(r_value)} & {tp} & {nbound} \\\\"
        )

    lines.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_table(path: Path) -> list[dict[str, float | int]]:
    rows = []
    for line in path.read_text().splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        mant = float(match.group("mant"))
        exponent = int(match.group("exp"))
        rows.append(
            {
                "D": float(match.group("D")),
                "E": float(match.group("E")),
                "omega": float(match.group("omega")),
                "S": float(match.group("S")),
                "r": mant * (10.0**exponent),
                "tp": int(match.group("tp")),
                "nbound": int(match.group("nbound")),
            }
        )
    if not rows:
        raise SystemExit(f"No Table III rows parsed from {path}")
    return rows


def check_table(
    artifact_rows: list[dict[str, float | int]],
    table_rows: list[dict[str, float | int]],
    *,
    epsilon: float,
    rel_r_tol: float,
) -> None:
    if len(artifact_rows) != len(table_rows):
        raise SystemExit(f"Row count mismatch: artifact={len(artifact_rows)}, table={len(table_rows)}")

    max_log_resid = 0.0
    max_rel_r_error = 0.0
    max_value_round_error = 0.0

    for artifact, table in zip(artifact_rows, table_rows):
        D = float(artifact["D"])
        if abs(float(table["D"]) - D) > 1.0e-9:
            raise SystemExit(f"D mismatch: artifact={D}, table={table['D']}")
        for key in ("E", "omega", "S"):
            err = abs(float(table[key]) - float(artifact[key]))
            max_value_round_error = max(max_value_round_error, err)
            if err > 5.5e-5:
                raise SystemExit(f"{key} display mismatch at D={D:g}: err={err:.3e}")
        for key in ("tp", "nbound"):
            if int(table[key]) != int(artifact[key]):
                raise SystemExit(f"{key} mismatch at D={D:g}: artifact={artifact[key]}, table={table[key]}")

        exact_r = math.exp(-2.0 * float(artifact["S"]))
        table_r = float(table["r"])
        rel_r_error = abs(table_r / exact_r - 1.0)
        max_rel_r_error = max(max_rel_r_error, rel_r_error)
        if rel_r_error > rel_r_tol:
            raise SystemExit(
                f"r_1 display mismatch at D={D:g}: rel_error={rel_r_error:.3e}, tol={rel_r_tol:.3e}"
            )

        log_resid = abs(math.log(table_r) + 2.0 * float(table["S"]))
        max_log_resid = max(max_log_resid, log_resid)
        if log_resid > epsilon:
            raise SystemExit(
                f"display WKB residual too large at D={D:g}: "
                f"|log r_1 + 2 S_1|={log_resid:.3e}, epsilon={epsilon:.3e}"
            )

    print("Table III WKB consistency check: PASS")
    print(f"  rows: {len(table_rows)}")
    print(f"  max |log r_1 + 2 S_1| from displayed values: {max_log_resid:.6e}")
    print(f"  max relative r_1 display error vs artifact exp(-2 S_1): {max_rel_r_error:.6e}")
    print(f"  max rounded value error for E_1, omega_1, S_1: {max_value_round_error:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--epsilon", type=float, default=2.0e-3)
    parser.add_argument("--rel-r-tol", type=float, default=5.0e-4)
    parser.add_argument("--rewrite-table", action="store_true")
    args = parser.parse_args()

    artifact_path = args.artifact.resolve()
    table_path = args.table.resolve()
    artifact_rows = load_artifact(artifact_path)

    if args.rewrite_table:
        table_path.write_text(table_text(artifact_rows, artifact_path))
        print(f"Rewrote {table_path.relative_to(ROOT)} from {artifact_path.relative_to(ROOT)}")

    table_rows = parse_table(table_path)
    check_table(artifact_rows, table_rows, epsilon=args.epsilon, rel_r_tol=args.rel_r_tol)


if __name__ == "__main__":
    main()
