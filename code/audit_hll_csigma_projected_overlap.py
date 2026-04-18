#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_csigma_source import d2_rho, d2_z, finite_box_amplitude  # noqa: E402
from extract_chi_localized_2d import Level, PhysicalParams  # noqa: E402
from extract_y_eff_2d_three_channel import (  # noqa: E402
    OverlapConfig,
    chirality_profiles,
    microcanonical_average,
    mode_overlap_values,
    solve_modes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether projected-Yukawa overlap closure selects the same "
            "c_sigma band as the core-to-box kernel matching."
        )
    )
    parser.add_argument(
        "--overlap-detail-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/mode_cache",
    )
    parser.add_argument(
        "--Ds",
        default="",
        help="Optional comma-separated D subset to benchmark. Default uses every row in overlap-detail input.",
    )
    parser.add_argument("--h2", type=float, default=1.0e-3)
    parser.add_argument("--cmin", type=float, default=22.0)
    parser.add_argument("--cmax", type=float, default=22.6)
    parser.add_argument("--nscan", type=int, default=601)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=50000)
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Shift-invert target used in the original overlap extractor; negative disables shift-invert.",
    )
    return parser.parse_args()


def microcanonical_weights(evals: np.ndarray, center_idx: int, cfg: OverlapConfig) -> tuple[np.ndarray, float, int]:
    n = len(evals)
    if n <= 1:
        weights = np.zeros_like(evals, dtype=float)
        weights[center_idx] = 1.0
        return weights, 1.0, 1
    gaps = []
    if center_idx > 0:
        gaps.append(abs(float(evals[center_idx] - evals[center_idx - 1])))
    if center_idx < n - 1:
        gaps.append(abs(float(evals[center_idx + 1] - evals[center_idx])))
    gap = float(max(min(gaps), 1.0e-6)) if gaps else 1.0
    sigma = float(max(cfg.window_floor, cfg.window_gap_scale * gap))
    idx = np.arange(n, dtype=int)
    dlam = np.abs(evals - evals[center_idx])
    mask = (np.abs(idx - center_idx) <= cfg.window_k) | (dlam <= cfg.window_sigma_mult * sigma)
    if not np.any(mask):
        mask[center_idx] = True
    weights = np.exp(-0.5 * (dlam / max(sigma, 1.0e-30)) ** 2) * mask.astype(float)
    norm = float(np.sum(weights))
    if norm <= 1.0e-30:
        weights = np.zeros_like(weights)
        weights[center_idx] = 1.0
        return weights, sigma, int(np.sum(mask))
    weights /= norm
    return weights, sigma, int(np.sum(mask))


def evaluate_projected_curve(
    c_grid: np.ndarray,
    kappa_iso: float,
    D: float,
    rho_max: float,
    z_margin: float,
    rr: np.ndarray,
    zz: np.ndarray,
    wpsi: np.ndarray,
    weights_mc: np.ndarray,
    chunk: int,
) -> np.ndarray:
    r2 = (rr * rr + zz * zz).reshape(-1)
    y2_vals = np.zeros_like(c_grid, dtype=float)
    pref = kappa_iso ** -0.25
    for start in range(0, len(c_grid), chunk):
        stop = min(start + chunk, len(c_grid))
        c_sub = c_grid[start:stop]
        sigma_sub = c_sub * pref
        amp_sub = np.array(
            [finite_box_amplitude(D, float(s), rho_max, z_margin) for s in sigma_sub],
            dtype=float,
        )
        kernel_chunk = amp_sub[:, None] * np.exp(-r2[None, :] / (sigma_sub[:, None] ** 2))
        overlaps = np.abs(kernel_chunk @ wpsi)
        y2_vals[start:stop] = overlaps @ weights_mc
    return y2_vals


def parse_d_subset(raw: str) -> set[float]:
    raw = str(raw).strip()
    if not raw:
        return set()
    return {float(x.strip()) for x in raw.split(",") if x.strip()}


def cache_path_for_row(cache_dir: Path, row: pd.Series, sigma: float | None) -> Path:
    sigma_tag = "none" if sigma is None else f"{float(sigma):g}"
    name = (
        f"D{float(row['D']):g}_level{row['level']}"
        f"_dr{float(row['dr']):g}_dz{float(row['dz']):g}"
        f"_rho{float(row['rho_max']):g}_zmg{float(row['z_max'] - row['D'] / 2.0):g}"
        f"_sig{sigma_tag}"
        f"_neigs{int(row['n_eigs'])}.npz"
    )
    safe = name.replace(".", "p").replace("-", "m")
    return cache_dir / safe


def load_or_solve_mode_block(
    *,
    row: pd.Series,
    p: PhysicalParams,
    level: Level,
    cfg: OverlapConfig,
    tol: float,
    maxiter: int,
    sigma: float | None,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_path_for_row(cache_dir, row, sigma)
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as data:
            print(f"[cache hit] D={float(row['D']):g} -> {cpath.name}")
            return {
                "rho": data["rho"],
                "rr": data["rr"],
                "zz": data["zz"],
                "psi": data["psi"],
                "evals": data["evals"],
            }

    D = float(row["D"])
    rho_max = float(row["rho_max"])
    z_margin = float(row["z_max"] - row["D"] / 2.0)
    n_eigs = int(row["n_eigs"])
    print(f"[cache miss] solving modes for D={D:g}")
    solved = solve_modes(
        d_val=D,
        level=level,
        p=p,
        rho_max=rho_max,
        z_margin=z_margin,
        tol=tol,
        maxiter=maxiter,
        sigma=sigma,
        n_eigs=n_eigs,
    )
    np.savez_compressed(
        cpath,
        rho=np.asarray(solved["rho"], dtype=float),
        rr=np.asarray(solved["rr"], dtype=float),
        zz=np.asarray(solved["zz"], dtype=float),
        psi=np.asarray(solved["psi"], dtype=float),
        evals=np.asarray(solved["evals"], dtype=float),
    )
    return {
        "rho": np.asarray(solved["rho"], dtype=float),
        "rr": np.asarray(solved["rr"], dtype=float),
        "zz": np.asarray(solved["zz"], dtype=float),
        "psi": np.asarray(solved["psi"], dtype=float),
        "evals": np.asarray(solved["evals"], dtype=float),
    }


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    d_subset = parse_d_subset(args.Ds)
    if d_subset:
        detail = detail[detail["D"].isin(d_subset)].copy().sort_values("D").reset_index(drop=True)
        if detail.empty:
            raise ValueError("--Ds filter removed every row from overlap-detail input.")
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])

    p = PhysicalParams()
    c_grid = np.linspace(args.cmin, args.cmax, int(args.nscan), dtype=float)
    cache_dir = Path(args.cache_dir)
    c_ref = None
    if "ref_D" in summary_row.index:
        ref_row = detail.loc[(detail["D"] - ref_d).abs().idxmin()]
        kappa_ref = (2.0 * d2_rho(float(ref_row["D"]), p, args.h2) + d2_z(float(ref_row["D"]), p, args.h2)) / 3.0
        c_ref = 0.5 * (float(ref_row["sigma_l"]) + float(ref_row["sigma_r"])) * (kappa_ref ** 0.25)
    c_amp = 22.3100

    out_rows: list[dict[str, float]] = []
    global_sq_ref = 0.0
    global_sq_amp = 0.0
    global_sq_best = np.zeros_like(c_grid, dtype=float)
    canonical_rebuild_resids = []

    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        n_eigs = int(row["n_eigs"])
        sigma_exact = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))
        level = Level(name=str(row["level"]), dr=dr, dz=dz)
        cfg = OverlapConfig(
            sigma_l=float(row["sigma_l"]),
            sigma_r=float(row["sigma_r"]),
            frame_power=float(row["frame_power"]),
            n_track=3,
            n_eigs=n_eigs,
            window_k=int(row["window_k"]),
            window_gap_scale=float(row["window_gap_scale"]),
            window_sigma_mult=float(row["window_sigma_mult"]),
            window_floor=float(row["window_floor"]),
        )

        solved = load_or_solve_mode_block(
            row=row,
            p=p,
            level=level,
            cfg=cfg,
            tol=args.tol,
            maxiter=args.maxiter,
            sigma=None if args.sigma < 0 else float(args.sigma),
            cache_dir=cache_dir,
        )
        rho = np.asarray(solved["rho"], dtype=float)
        rr = np.asarray(solved["rr"], dtype=float)
        zz = np.asarray(solved["zz"], dtype=float)
        psi = np.asarray(solved["psi"], dtype=float)
        evals = np.asarray(solved["evals"], dtype=float)
        center_idx = int(row["track_idx_2"])
        weights_mc, sigma_mc, nwin = microcanonical_weights(evals, center_idx, cfg)
        w = 2.0 * math.pi * rho[:, None] * dr * dz
        wpsi = np.stack([(w * psi[:, :, k]).reshape(-1) for k in range(psi.shape[2])], axis=1)

        f_l, f_r = chirality_profiles(
            rr=rr,
            zz=zz,
            rho=rho,
            dr=dr,
            dz=dz,
            d_val=D,
            sigma_l=sigma_exact,
            sigma_r=sigma_exact,
        )
        kernel_exact = f_l * f_r
        y_modes_exact = mode_overlap_values(psi=psi, kernel=kernel_exact, rho=rho, dr=dr, dz=dz)
        y2_rebuild, _, _ = microcanonical_average(evals=evals, y_modes=y_modes_exact, center_idx=center_idx, cfg=cfg)
        y2_exact = float(row["y_eff_raw_2"])
        canonical_relerr = float(abs(y2_rebuild / max(y2_exact, 1.0e-300) - 1.0))
        canonical_rebuild_resids.append(canonical_relerr)

        kappa_iso = (2.0 * d2_rho(D, p, args.h2) + d2_z(D, p, args.h2)) / 3.0
        y2_curve = evaluate_projected_curve(
            c_grid=c_grid,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )
        log_resid = np.abs(np.log(np.clip(y2_curve, 1.0e-300, None)) - np.log(max(y2_exact, 1.0e-300)))
        idx_best = int(np.argmin(log_resid))
        c_local = float(c_grid[idx_best])
        y2_best = float(y2_curve[idx_best])
        sigma_local = c_local * (kappa_iso ** -0.25)

        idx_ref = int(np.argmin(np.abs(c_grid - c_ref)))
        idx_amp = int(np.argmin(np.abs(c_grid - c_amp)))
        y2_ref = float(y2_curve[idx_ref])
        y2_amp = float(y2_curve[idx_amp])

        global_sq_best += (np.log(np.clip(y2_curve, 1.0e-300, None)) - np.log(max(y2_exact, 1.0e-300))) ** 2
        global_sq_ref += (math.log(max(y2_ref, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2
        global_sq_amp += (math.log(max(y2_amp, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2

        out_rows.append(
            {
                "D": D,
                "kappa_iso": kappa_iso,
                "sigma_exact": sigma_exact,
                "sigma_mc": sigma_mc,
                "nwin": nwin,
                "y2_exact": y2_exact,
                "y2_rebuild_exact_kernel": y2_rebuild,
                "canonical_rebuild_relerr": canonical_relerr,
                "c_sigma_projected_local": c_local,
                "sigma_projected_local": sigma_local,
                "y2_projected_local": y2_best,
                "rel_y2_resid_local": float(abs(y2_best / max(y2_exact, 1.0e-300) - 1.0)),
                "c_sigma_ref": float(c_ref),
                "c_sigma_amp": float(c_amp),
                "y2_projected_ref": y2_ref,
                "y2_projected_amp": y2_amp,
                "rel_y2_resid_ref": float(abs(y2_ref / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_y2_resid_amp": float(abs(y2_amp / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_drift_local_vs_ref": float(abs(c_local / max(float(c_ref), 1.0e-300) - 1.0)),
                "rel_drift_local_vs_amp": float(abs(c_local / c_amp - 1.0)),
            }
        )

    out_detail = pd.DataFrame(out_rows).sort_values("D").reset_index(drop=True)
    idx_best_global = int(np.argmin(global_sq_best))
    c_proj_global = float(c_grid[idx_best_global])
    nearest = out_detail.loc[(out_detail["D"] - ref_d).abs().idxmin()]

    summary = pd.DataFrame(
        [
            {
                "section": "canonical_overlap_rebuild",
                "max_canonical_rebuild_relerr": float(np.max(canonical_rebuild_resids)),
                "mean_canonical_rebuild_relerr": float(np.mean(canonical_rebuild_resids)),
            },
            {
                "section": "projected_overlap_selector",
                "c_sigma_ref": float(c_ref),
                "c_sigma_amp": float(c_amp),
                "c_sigma_projected_global": c_proj_global,
                "rel_drift_proj_global_vs_ref": float(abs(c_proj_global / max(float(c_ref), 1.0e-300) - 1.0)),
                "rel_drift_proj_global_vs_amp": float(abs(c_proj_global / c_amp - 1.0)),
                "c_sigma_projected_local_min": float(out_detail["c_sigma_projected_local"].min()),
                "c_sigma_projected_local_max": float(out_detail["c_sigma_projected_local"].max()),
                "c_sigma_projected_local_mean": float(out_detail["c_sigma_projected_local"].mean()),
                "c_sigma_projected_local_rel_span": float(
                    (out_detail["c_sigma_projected_local"].max() - out_detail["c_sigma_projected_local"].min())
                    / out_detail["c_sigma_projected_local"].mean()
                ),
            },
            {
                "section": "reference_selector_metrics",
                "c": float(c_ref),
                "y2_log_rmse": float(np.sqrt(global_sq_ref / len(out_detail))),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_ref"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_ref"].mean()),
            },
            {
                "section": "amplitude_selector_metrics",
                "c": float(c_amp),
                "y2_log_rmse": float(np.sqrt(global_sq_amp / len(out_detail))),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_amp"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_amp"].mean()),
            },
            {
                "section": "projected_overlap_global_fit_metrics",
                "c": c_proj_global,
                "y2_log_rmse": float(np.sqrt(global_sq_best[idx_best_global] / len(out_detail))),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_local"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_local"].mean()),
            },
            {
                "section": "reference_point_projected_overlap",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "nearest_detail_D": float(nearest["D"]),
                "c_sigma_projected_local_ref": float(nearest["c_sigma_projected_local"]),
                "y2_exact_ref": float(nearest["y2_exact"]),
                "y2_projected_ref_ref": float(nearest["y2_projected_ref"]),
                "y2_projected_amp_ref": float(nearest["y2_projected_amp"]),
                "y2_projected_local_ref": float(nearest["y2_projected_local"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "projected_overlap_global_selector",
                "formula": "argmin_c sum_D [log y_2^pred(D;c)-log y_2^raw(D)]^2",
                "status": "projected-Yukawa overlap block source candidate",
                "value": c_proj_global,
            },
            {
                "candidate": "finite_box_amplitude_selector",
                "formula": "argmin_c sum_D [log A_mu^disc(D)-log A_mu^box(D;c)]^2",
                "status": "current kernel-side best selector",
                "value": float(c_amp),
            },
            {
                "candidate": "reference_curvature_selector",
                "formula": "c_sigma(D_*)=sigma_mu(D_*) * kappa_iso(D_*)^(1/4)",
                "status": "single-point curvature calibration",
                "value": float(c_ref),
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_csigma_projected_overlap_summary.csv"
    detail_out = outdir / "hll_csigma_projected_overlap_detail.csv"
    constants_out = outdir / "hll_csigma_projected_overlap_constants.csv"
    meta_out = outdir / "hll_csigma_projected_overlap_run_meta.json"

    summary.to_csv(summary_out, index=False)
    out_detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "cmin": float(args.cmin),
                "cmax": float(args.cmax),
                "nscan": int(args.nscan),
                "chunk": int(args.chunk),
                "tol": float(args.tol),
                "maxiter": int(args.maxiter),
                "sigma": None if args.sigma < 0 else float(args.sigma),
                "h2": float(args.h2),
                "Ds_filter": sorted(d_subset),
                "cache_dir": str(cache_dir),
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()
