#!/usr/bin/env python3
"""
Build a pre-submission freeze report for PRD/JHEP-targeted PSLT artifacts.

Outputs:
  - JSON machine-readable report
  - Markdown human-readable report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_git(root: Path, args: List[str]) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        return out
    except Exception:
        return "unknown"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: List[Dict[str, str]], key: str, value: str) -> Dict[str, str] | None:
    for r in rows:
        if str(r.get(key, "")).strip() == value:
            return r
    return None


def to_float(v: str | None, default: float = float("nan")) -> float:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def fmt_num(v: float) -> str:
    if v != v:  # NaN
        return "NaN"
    if abs(v) >= 1e-2:
        return f"{v:.6f}"
    return f"{v:.6e}"


def build_report(root: Path, run_id: str) -> Tuple[Dict[str, object], str]:
    required_files = [
        root / "paper" / "main.tex",
        root / "paper" / "main.pdf",
        root / "paper" / "hll_signal_strength_summary.csv",
        root / "paper" / "hll_uv_to_eft_summary.csv",
        root / "paper" / "hll_rge_sensitivity.csv",
        root / "paper" / "full_direct_map_release_summary.csv",
        root / "output" / "first_principles_migration" / "first_principles_migration_summary.csv",
        root / "output" / "hll_signal_strength" / "hll_signal_strength_run_meta_baseline.json",
    ]

    checks: List[CheckResult] = []
    file_meta: List[Dict[str, object]] = []
    for p in required_files:
        exists = p.exists() and p.is_file() and p.stat().st_size > 0
        checks.append(CheckResult(name=f"file:{p.relative_to(root)}", ok=exists, detail="exists and non-empty" if exists else "missing or empty"))
        if exists:
            file_meta.append(
                {
                    "path": str(p.relative_to(root)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256sum(p),
                }
            )

    tex = root / "paper" / "main.tex"
    pdf = root / "paper" / "main.pdf"
    pdf_newer = pdf.exists() and tex.exists() and pdf.stat().st_mtime >= tex.stat().st_mtime
    checks.append(
        CheckResult(
            name="paper_sync",
            ok=pdf_newer,
            detail="main.pdf mtime >= main.tex mtime" if pdf_newer else "main.pdf older than main.tex",
        )
    )

    hll_sum = read_csv_rows(root / "paper" / "hll_signal_strength_summary.csv")
    mumu_row = find_row(hll_sum, "channel", "mumu")
    mumu_f_chi2 = to_float(mumu_row.get("f_chi2_le_4") if mumu_row else None)
    mumu_best = to_float(mumu_row.get("best_chi2") if mumu_row else None)
    checks.append(
        CheckResult(
            name="hll_summary_row_mumu",
            ok=mumu_row is not None,
            detail="mumu row found" if mumu_row is not None else "missing mumu row",
        )
    )

    uv_sum = read_csv_rows(root / "paper" / "hll_uv_to_eft_summary.csv")
    uv_row = uv_sum[0] if uv_sum else None
    uv_f_chi2_rge = to_float(uv_row.get("f_chi2_le_4_uv_rge") if uv_row else None)
    uv_max_dmu = to_float(uv_row.get("max_abs_delta_mu_mumu") if uv_row else None)
    uv_max_dcmatch = to_float(uv_row.get("max_abs_delta_C_match_mumu") if uv_row else None)
    checks.append(
        CheckResult(
            name="uv_summary_present",
            ok=uv_row is not None,
            detail="uv summary row found" if uv_row is not None else "missing uv summary row",
        )
    )

    if uv_row is not None and mumu_row is not None:
        close = abs(uv_f_chi2_rge - mumu_f_chi2) <= 1e-12
        checks.append(
            CheckResult(
                name="chi2_fraction_consistency",
                ok=close,
                detail=f"f_chi2_le_4_uv_rge={uv_f_chi2_rge:.6f}, hll_mumu_f_chi2_le_4={mumu_f_chi2:.6f}",
            )
        )

    rge_rows = read_csv_rows(root / "paper" / "hll_rge_sensitivity.csv")
    params = {r.get("parameter", "").strip() for r in rge_rows}
    expected = {"mu_low", "gamma_diag", "gamma_offdiag", "kappa_diag", "kappa_offdiag"}
    checks.append(
        CheckResult(
            name="rge_sensitivity_parameters",
            ok=expected.issubset(params),
            detail=f"found={sorted(params)}",
        )
    )

    mig_rows = read_csv_rows(root / "output" / "first_principles_migration" / "first_principles_migration_summary.csv")
    base = find_row(mig_rows, "scenario", "baseline_fp2d_full_action_grid")
    r3_base = to_float(base.get("f_R3_gt_0p90") if base else None)
    checks.append(
        CheckResult(
            name="baseline_migration_row",
            ok=base is not None,
            detail="baseline_fp2d_full_action_grid row found" if base is not None else "missing baseline migration row",
        )
    )

    run_meta = json.loads((root / "output" / "hll_signal_strength" / "hll_signal_strength_run_meta_baseline.json").read_text())
    kdiag = float(run_meta.get("uv_match_kappa_diag", float("nan")))
    koff = float(run_meta.get("uv_match_kappa_offdiag", float("nan")))
    checks.append(
        CheckResult(
            name="run_meta_has_finite_match_knobs",
            ok=("uv_match_kappa_diag" in run_meta and "uv_match_kappa_offdiag" in run_meta),
            detail=f"kappa_diag={kdiag}, kappa_offdiag={koff}",
        )
    )

    release_rows = read_csv_rows(root / "paper" / "full_direct_map_release_summary.csv")
    row_small_direct = find_row(release_rows, "scenario", "small_surface_complete_direct_bias")
    row_large_direct = find_row(release_rows, "scenario", "large_surface_spotcheck_direct_bias")
    row_small_tuned = find_row(
        release_rows,
        "scenario",
        "chain_mode_parity_full_direct_vs_cell_direct_runtime_bnorm_release_tuned_profile_anchored",
    )
    row_large_tuned = find_row(
        release_rows,
        "scenario",
        "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_bnorm_release_tuned_profile_anchored",
    )

    checks.append(
        CheckResult(
            name="release_gate_rows_present",
            ok=(
                row_small_direct is not None
                and row_large_direct is not None
                and row_small_tuned is not None
                and row_large_tuned is not None
            ),
            detail=(
                "small/large direct-bias rows and profile-anchored tuned parity rows found"
                if (
                    row_small_direct is not None
                    and row_large_direct is not None
                    and row_small_tuned is not None
                    and row_large_tuned is not None
                )
                else "missing one or more release-gate rows in paper/full_direct_map_release_summary.csv"
            ),
        )
    )

    full_direct_thresholds = {
        "small_frac_winner_mismatch": 0.01,
        "small_max_abs_delta_R3": 1e-3,
        "small_max_abs_delta_mu_mumu": 0.10,
        "large_frac_winner_mismatch": 0.01,
        "large_max_abs_delta_R3": 1e-3,
        "large_max_abs_delta_mu_mumu": 0.10,
    }
    tuned_promotion_thresholds = {
        "small_frac_acceptance_mismatch": 0.01,
        "small_max_abs_delta_mu_mumu": 1.0,
        "large_frac_acceptance_mismatch": 0.01,
        "large_max_abs_delta_mu_mumu": 1.0,
    }

    full_direct_metrics = {
        "small_frac_winner_mismatch": to_float(row_small_direct.get("frac_winner_mismatch") if row_small_direct else None),
        "small_max_abs_delta_R3": to_float(row_small_direct.get("max_abs_delta_R3") if row_small_direct else None),
        "small_max_abs_delta_mu_mumu": to_float(row_small_direct.get("max_abs_delta_mu_mumu") if row_small_direct else None),
        "large_frac_winner_mismatch": to_float(row_large_direct.get("frac_winner_mismatch") if row_large_direct else None),
        "large_max_abs_delta_R3": to_float(row_large_direct.get("max_abs_delta_R3") if row_large_direct else None),
        "large_max_abs_delta_mu_mumu": to_float(row_large_direct.get("max_abs_delta_mu_mumu") if row_large_direct else None),
    }
    tuned_promotion_metrics = {
        "small_frac_acceptance_mismatch": to_float(row_small_tuned.get("frac_winner_mismatch") if row_small_tuned else None),
        "small_max_abs_delta_mu_mumu": to_float(row_small_tuned.get("max_abs_delta_mu_mumu") if row_small_tuned else None),
        "large_frac_acceptance_mismatch": to_float(row_large_tuned.get("frac_winner_mismatch") if row_large_tuned else None),
        "large_max_abs_delta_mu_mumu": to_float(row_large_tuned.get("max_abs_delta_mu_mumu") if row_large_tuned else None),
    }

    full_direct_gate_ok = all(
        (full_direct_metrics[k] == full_direct_metrics[k]) and (full_direct_metrics[k] <= v)
        for k, v in full_direct_thresholds.items()
    )
    full_direct_gate_decision = "GO" if full_direct_gate_ok else "HOLD"
    checks.append(
        CheckResult(
            name="gate_full_direct_release",
            ok=full_direct_gate_ok,
            detail=(
                f"decision={full_direct_gate_decision}; "
                + ", ".join(
                    f"{k}={fmt_num(full_direct_metrics[k])}<= {fmt_num(v)}"
                    for k, v in full_direct_thresholds.items()
                )
            ),
        )
    )

    tuned_promotion_ok = all(
        (tuned_promotion_metrics[k] == tuned_promotion_metrics[k]) and (tuned_promotion_metrics[k] <= v)
        for k, v in tuned_promotion_thresholds.items()
    )
    tuned_promotion_decision = "GO" if tuned_promotion_ok else "HOLD"
    checks.append(
        CheckResult(
            name="gate_runtime_release_tuned_promotion_rows",
            ok=(row_small_tuned is not None and row_large_tuned is not None),
            detail=(
                f"decision={tuned_promotion_decision}; "
                + ", ".join(
                    f"{k}={fmt_num(tuned_promotion_metrics[k])}<= {fmt_num(v)}"
                    for k, v in tuned_promotion_thresholds.items()
                )
            ),
        )
    )

    all_ok = all(c.ok for c in checks)
    commit = run_git(root, ["rev-parse", "HEAD"])
    branch = run_git(root, ["branch", "--show-current"])
    dirty = run_git(root, ["status", "--porcelain"])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "git": {
            "branch": branch,
            "commit": commit,
            "dirty": bool(dirty.strip()),
        },
        "key_metrics": {
            "f_R3_gt_0p90_baseline": r3_base,
            "f_chi2_le_4_mumu": mumu_f_chi2,
            "best_chi2_mumu": mumu_best,
            "uv_max_abs_delta_mu_mumu": uv_max_dmu,
            "uv_max_abs_delta_C_match_mumu": uv_max_dcmatch,
            "gate_full_direct_release": full_direct_gate_decision,
            "gate_runtime_release_tuned_promotion": tuned_promotion_decision,
        },
        "release_gates": {
            "full_direct_release": {
                "decision": full_direct_gate_decision,
                "pass": full_direct_gate_ok,
                "thresholds": full_direct_thresholds,
                "metrics": full_direct_metrics,
            },
            "runtime_release_tuned_promotion": {
                "decision": tuned_promotion_decision,
                "pass": tuned_promotion_ok,
                "thresholds": tuned_promotion_thresholds,
                "metrics": tuned_promotion_metrics,
            },
        },
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "files": file_meta,
        "all_checks_passed": all_ok,
    }

    lines = []
    lines.append("# PRD Pre-Submission Freeze Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {report['generated_at_utc']}")
    lines.append(f"- Run ID: {run_id}")
    lines.append(f"- Git branch: {branch}")
    lines.append(f"- Git commit: {commit}")
    lines.append(f"- Git dirty at report time: {report['git']['dirty']}")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(f"- Baseline `f(R3>0.90)`: {r3_base:.6f}" if r3_base == r3_base else "- Baseline `f(R3>0.90)`: NaN")
    lines.append(f"- `f(chi2_mumu<4)`: {mumu_f_chi2:.6f}" if mumu_f_chi2 == mumu_f_chi2 else "- `f(chi2_mumu<4)`: NaN")
    lines.append(f"- Best `chi2_mumu`: {mumu_best:.6e}" if mumu_best == mumu_best else "- Best `chi2_mumu`: NaN")
    lines.append(f"- UV audit `max |delta mu_mumu|`: {uv_max_dmu:.6e}" if uv_max_dmu == uv_max_dmu else "- UV audit `max |delta mu_mumu|`: NaN")
    lines.append(f"- UV audit `max |delta C_match_mumu|`: {uv_max_dcmatch:.6e}" if uv_max_dcmatch == uv_max_dcmatch else "- UV audit `max |delta C_match_mumu|`: NaN")
    lines.append(f"- Full-direct release gate: **{full_direct_gate_decision}**")
    lines.append(f"- Runtime-tuned promotion gate: **{tuned_promotion_decision}**")
    lines.append("")
    lines.append("## Release Gates")
    lines.append("")
    lines.append("| Gate | Decision | Pass | Key metrics vs thresholds |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| `full_direct_release` | "
        f"{full_direct_gate_decision} | {'PASS' if full_direct_gate_ok else 'FAIL'} | "
        + "; ".join(
            f"{k}={fmt_num(full_direct_metrics[k])}<= {fmt_num(v)}"
            for k, v in full_direct_thresholds.items()
        )
        + " |"
    )
    lines.append(
        "| `runtime_release_tuned_promotion` | "
        f"{tuned_promotion_decision} | {'PASS' if tuned_promotion_ok else 'FAIL'} | "
        + "; ".join(
            f"{k}={fmt_num(tuned_promotion_metrics[k])}<= {fmt_num(v)}"
            for k, v in tuned_promotion_thresholds.items()
        )
        + " |"
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| `{c.name}` | {'PASS' if c.ok else 'FAIL'} | {c.detail} |")
    lines.append("")
    lines.append(f"**Overall:** {'PASS' if all_ok else 'FAIL'}")

    return report, "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PRD pre-submission freeze report.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-md", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id.strip() or "presubmit"
    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    report, md = build_report(root=root, run_id=run_id)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    out_md.write_text(md)

    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(f"[status] all_checks_passed={report['all_checks_passed']}")


if __name__ == "__main__":
    main()
