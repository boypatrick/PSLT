#!/usr/bin/env python3
"""
PSLT Generation-Matching Demonstrator (paper-consistent, reproducible)

Implements the PSLT closure:
  Q_N(D,eta;t) = B_N * g_N(c_eff,nu) * (1 - exp(-Gamma_N(D,eta) * t))
  P_N = Q_N / sum_K Q_K
  N*(D,eta) = argmax_N Q_N
  Delta(D,eta) = log Q_(1) - log Q_(2)  (top-2 competition margin)
  chi_D = |d<N>/dD|

Kinetic ansatz (finite-time / freeze-out selection):
  Gamma_N(D,eta) = eta * Gamma_ref * N^{-p} * exp(-a * D * (N-1))

Cardy-like microstate growth:
  g_N(c_eff,nu) = N^{-nu} * exp(2*pi*sqrt(c_eff*N/6))

This repository contains *two* explicit presets (documented in main.tex):
  - "generation": low-N three-peak demonstrator (Fig. P_N windows; Fig. kernels scaling)
  - "band": transition-band scan for interface diagnostics (Fig. transition-band; overlay)

Run:
  python transition_band_scan.py all --preset generation
  python transition_band_scan.py all --preset band
or generate one figure at a time:
  python transition_band_scan.py pnt-windows --preset generation
  python transition_band_scan.py kernels-scaling --preset generation
  python transition_band_scan.py transition-band --preset band

All figure scripts print the full parameter set used.
"""
from __future__ import annotations
import argparse
import math
import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Geometry-closed t_coh helper -------------------------

class TcohCurve:
    """
    Provides t_coh_geom(D) by interpolating a precomputed curve table.

    Expected CSV columns:
      - D
      - tcoh_geom   (preferred)
      - or log10_tcoh_geom

    Extrapolation modes:
      - "linear-log": linear extrapolation in ln(tcoh) vs D using endpoint slopes
      - "clamp": clamp D to [Dmin,Dmax]
      - "error": raise ValueError outside domain
    """
    def __init__(self, csv_path: str, extrap: str = "linear-log"):
        self.csv_path = str(csv_path)
        self.extrap = str(extrap).strip().lower()

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"tcoh curve CSV not found: {self.csv_path}")

        data = np.genfromtxt(self.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if "D" not in data.dtype.names:
            raise ValueError("tcoh curve CSV must include column 'D'.")

        D = np.array(data["D"], dtype=float)

        if "tcoh_geom" in data.dtype.names:
            t = np.array(data["tcoh_geom"], dtype=float)
        elif "log10_tcoh_geom" in data.dtype.names:
            t = np.power(10.0, np.array(data["log10_tcoh_geom"], dtype=float))
        else:
            raise ValueError("tcoh curve CSV must include 'tcoh_geom' or 'log10_tcoh_geom'.")

        if np.any(~np.isfinite(D)) or np.any(~np.isfinite(t)) or np.any(t <= 0):
            raise ValueError("Invalid values in tcoh curve CSV (non-finite or non-positive).")

        idx = np.argsort(D)
        self.D = D[idx]
        self.logt = np.log(t[idx])
        self.Dmin = float(self.D[0])
        self.Dmax = float(self.D[-1])

        if len(self.D) < 2:
            raise ValueError("tcoh curve must have at least 2 points.")
        self._slope_lo = float((self.logt[1] - self.logt[0]) / (self.D[1] - self.D[0]))
        self._slope_hi = float((self.logt[-1] - self.logt[-2]) / (self.D[-1] - self.D[-2]))

    def tcoh(self, D: float) -> float:
        D = float(D)
        if self.Dmin <= D <= self.Dmax:
            logt = float(np.interp(D, self.D, self.logt))
            return float(math.exp(logt))

        if self.extrap == "clamp":
            Dc = min(max(D, self.Dmin), self.Dmax)
            logt = float(np.interp(Dc, self.D, self.logt))
            return float(math.exp(logt))

        if self.extrap == "error":
            raise ValueError(f"D={D} outside curve domain [{self.Dmin},{self.Dmax}] and extrap='error'.")

        # default: linear-log
        if D < self.Dmin:
            logt = float(self.logt[0] + self._slope_lo * (D - self.Dmin))
            return float(math.exp(logt))
        else:
            logt = float(self.logt[-1] + self._slope_hi * (D - self.Dmax))
            return float(math.exp(logt))


def _get_tcoh_fn(par: "Params", t_scalar: float):
    """
    Returns a callable tcoh(D) given current tcoh mode.
    - fixed: tcoh(D)=t_scalar
    - geom:  tcoh(D)=t_scalar * tcoh_geom(D)  (t_scalar acts as a multiplier)
    """
    mode = getattr(par, "_tcoh_mode", "fixed")
    if mode == "geom":
        curve = getattr(par, "_tcoh_curve", None)
        if curve is None:
            raise RuntimeError("tcoh_mode=geom but no curve loaded.")
        mult = float(t_scalar)
        return lambda D: mult * float(curve.tcoh(float(D)))
    else:
        t = float(t_scalar)
        return lambda D: t


def _effective_t_at_Dfix(par: "Params", t_scalar: float) -> float:
    """Effective tcoh evaluated at D_fix (used for x-axis display)."""
    mode = getattr(par, "_tcoh_mode", "fixed")
    if mode == "geom":
        curve = getattr(par, "_tcoh_curve", None)
        return float(t_scalar) * float(curve.tcoh(float(par.D_fix)))
    return float(t_scalar)

# ------------------------- Model pieces -------------------------

def B_matching(N: int, B1: float, B2: float, B3: float) -> float:
    if N == 1: return float(B1)
    if N == 2: return float(B2)
    if N == 3: return float(B3)
    return 1.0

def g_cardy(N: int, c_eff: float, nu: float) -> float:
    # Cardy-like asymptotic + a controllable low-N softening
    return (N ** (-float(nu))) * math.exp(2.0 * math.pi * math.sqrt(float(c_eff) * N / 6.0))

def Gamma_N(N: int, D: float, eta: float, Gamma_ref: float, p: float, a: float, use_Nminus1: bool = True) -> float:
    Nf = float(N)
    expo = (Nf - 1.0) if use_Nminus1 else Nf
    return float(eta) * float(Gamma_ref) * (Nf ** (-float(p))) * math.exp(-float(a) * float(D) * expo)

def logF_from_logx(logx: np.ndarray) -> np.ndarray:
    """
    Compute log(1-exp(-x)) stably where logx = log(x) and x>0.
    Uses:
      - small-x: 1-exp(-x) ~ x => logF ~ logx
      - large-x: logF ~ 0
      - mid: log1p(-exp(-x)) with x in float domain
    """
    logx = np.asarray(logx, dtype=float)
    out = np.empty_like(logx)

    # thresholds tuned for double precision
    thresh_small = math.log(1e-8)
    thresh_large = math.log(50.0)

    small = logx < thresh_small
    large = logx > thresh_large
    mid = ~(small | large)

    out[small] = logx[small]
    out[large] = 0.0
    if np.any(mid):
        x_mid = np.exp(logx[mid])
        out[mid] = np.log1p(-np.exp(-x_mid))
    return out

@dataclass
class Params:
    # shared
    Nmin: int = 1
    Nmax: int = 20
    c_eff: float = 1.2
    nu: float = 0.0
    p: float = 2.0
    a: float = 0.03
    Gamma_ref: float = 500.0
    B1: float = 500.0
    B2: float = 50.0
    B3: float = 5.0
    use_Nminus1: bool = True

    # scan grids
    Dmin: float = 6.0
    Dmax: float = 34.0
    nD: int = 100
    etamin: float = 0.2
    etamax: float = 2.0
    neta: int = 80

    # band / time
    Delta0: float = 0.05
    t_rep: float = 10.0
    t_list: Tuple[float, ...] = (0.5, 3.0, 10.0, 50.0, 200.0)

    # fixed-point figures
    D_fix: float = 10.0
    eta_fix: float = 2.0
    t_windows: Tuple[float, ...] = (2.0, 20.0, 100.0)

def preset(name: str) -> Params:
    name = name.strip().lower()
    if name in ("band", "transition-band", "scan"):
        # interface diagnostics preset (chosen to produce multiple domains in (D,eta))
        return Params(
            Nmin=1, Nmax=20,
            c_eff=1.2, nu=0.0,
            p=2.0, a=0.03,
            Gamma_ref=500.0,
            B1=500.0, B2=50.0, B3=5.0,
            use_Nminus1=True,
            Dmin=6.0, Dmax=34.0, nD=100,
            etamin=0.2, etamax=2.0, neta=80,
            Delta0=0.05, t_rep=10.0,
            t_list=(0.5, 3.0, 10.0, 50.0, 200.0),
            D_fix=10.0, eta_fix=1.0,
            t_windows=(2.0, 20.0, 100.0)
        )
    if name in ("generation", "gen", "three-generation"):
        # generation-matching preset (paper Eq.(bench) + rate rescaling for visible windows)
        # Note: only the dimensionless products Gamma_N * t matter for the windows; Gamma_ref can be
        # absorbed into t in this demonstrator.
        return Params(
            Nmin=1, Nmax=20,
            c_eff=0.5, nu=1.5,
            p=2.0, a=0.03,
            Gamma_ref=10.0,   # equivalent to rescaling t by 10 compared to Gamma_ref=1
            B1=19.0, B2=13.0, B3=9.0,
            use_Nminus1=True,
            Dmin=6.0, Dmax=34.0, nD=100,
            etamin=0.2, etamax=2.0, neta=80,
            Delta0=0.05, t_rep=10.0,
            t_list=(0.5, 3.0, 10.0, 50.0, 200.0),
            D_fix=10.0, eta_fix=2.0,
            t_windows=(2.0, 20.0, 100.0)
        )
    raise ValueError(f"Unknown preset '{name}'. Use 'band' or 'generation'.")

# ------------------------- Core computations -------------------------

def compute_phase(par: Params, tcoh) -> Dict[str, Any]:
    D_grid = np.linspace(par.Dmin, par.Dmax, par.nD)
    eta_grid = np.linspace(par.etamin, par.etamax, par.neta)
    Ns = np.arange(par.Nmin, par.Nmax + 1, dtype=int)

    # precompute N-dependent terms
    logg = np.log([g_cardy(int(N), par.c_eff, par.nu) for N in Ns])
    logB = np.log([B_matching(int(N), par.B1, par.B2, par.B3) for N in Ns])
    logGamma_base = math.log(par.Gamma_ref) - float(par.p) * np.log(Ns.astype(float))

    winner = np.zeros((len(eta_grid), len(D_grid)), dtype=int)
    margin = np.zeros_like(winner, dtype=float)
    H = np.zeros_like(margin)
    Nmean = np.zeros_like(margin)

    # allow scalar tcoh or callable tcoh(D)
    if callable(tcoh):
        logt_grid = np.log([max(float(tcoh(float(D))), 1e-300) for D in D_grid])
    else:
        logt_grid = float(math.log(max(float(tcoh), 1e-300))) * np.ones_like(D_grid, dtype=float)

    for ie, eta in enumerate(eta_grid):
        logeta = math.log(float(eta))
        for idd, D in enumerate(D_grid):
            expo = (Ns.astype(float) - 1.0) if par.use_Nminus1 else Ns.astype(float)
            logGamma = logeta + logGamma_base - float(par.a) * float(D) * expo
            logx = logGamma + logt_grid[idd]

            logF = logF_from_logx(logx)
            logQ = logB + logg + logF

            idx = np.argsort(logQ)[::-1]
            top, second = int(idx[0]), int(idx[1])
            winner[ie, idd] = int(Ns[top])
            margin[ie, idd] = float(logQ[top] - logQ[second])

            # softmax for <N> and entropy
            mmax = float(np.max(logQ))
            w = np.exp(logQ - mmax)
            P = w / w.sum()
            Nmean[ie, idd] = float((P * Ns).sum())
            # avoid log(0)
            Ppos = np.where(P > 0, P, 1.0)
            H[ie, idd] = float(-(P * np.log(Ppos)).sum())

    chiD = np.abs(np.gradient(Nmean, D_grid, axis=1))
    return dict(D=D_grid, eta=eta_grid, winner=winner, margin=margin, H=H, chiD=chiD, Nmean=Nmean)

def P_distribution_at_point(par: Params, D: float, eta: float, tcoh) -> Tuple[np.ndarray, np.ndarray]:
    Ns = np.arange(par.Nmin, par.Nmax + 1, dtype=int)
    Q = np.zeros_like(Ns, dtype=float)
    for i, N in enumerate(Ns):
        g = g_cardy(int(N), par.c_eff, par.nu)
        Bn = B_matching(int(N), par.B1, par.B2, par.B3)
        Gam = Gamma_N(int(N), D, eta, par.Gamma_ref, par.p, par.a, par.use_Nminus1)
        t_eff = float(tcoh(float(D))) if callable(tcoh) else float(tcoh)
        F = 1.0 - math.exp(-Gam * t_eff)
        Q[i] = Bn * g * F
    P = Q / Q.sum()
    return Ns.astype(float), P

# ------------------------- Figures -------------------------

def print_params(tag: str, par: Params) -> None:
    d = asdict(par)
    # compact print, stable ordering
    keys = ["Nmin","Nmax","c_eff","nu","p","a","Gamma_ref","B1","B2","B3","use_Nminus1",
            "Dmin","Dmax","nD","etamin","etamax","neta","Delta0","t_rep","t_list","D_fix","eta_fix","t_windows"]
    msg = ", ".join([f"{k}={d[k]}" for k in keys if k in d])
    print(f"[{tag}] {msg}")

def fig_transition_band(par: Params, out_png: str = "fig_transition_band.png", out_overlay: str = "fig_transition_band_overlay.png") -> None:
    print_params("transition-band", par)
    tcoh_fn = _get_tcoh_fn(par, par.t_rep)
    ph = compute_phase(par, tcoh_fn)

    band_mask = ph["margin"] < par.Delta0
    DD, EE = np.meshgrid(ph["D"], ph["eta"])

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    im0 = axs[0, 0].pcolormesh(DD, EE, ph["winner"], shading="auto")
    axs[0, 0].set_title("Winner map $N_\\star(D,\\eta)$")
    axs[0, 0].set_xlabel("D")
    axs[0, 0].set_ylabel("eta")
    fig.colorbar(im0, ax=axs[0, 0], label="$N_\\star$")

    im1 = axs[0, 1].pcolormesh(DD, EE, ph["margin"], shading="auto")
    axs[0, 1].contour(DD, EE, band_mask.astype(float), levels=[0.5], linewidths=1.2)
    axs[0, 1].set_title(r"Margin $\Delta=\log Q_{(1)}-\log Q_{(2)}$; band: $\Delta<\Delta_0$")
    axs[0, 1].set_xlabel("D")
    axs[0, 1].set_ylabel("eta")
    fig.colorbar(im1, ax=axs[0, 1], label=r"$\Delta$")

    im2 = axs[1, 0].pcolormesh(DD, EE, ph["chiD"], shading="auto")
    axs[1, 0].contour(DD, EE, band_mask.astype(float), levels=[0.5], linewidths=1.2)
    axs[1, 0].set_title(r"Interfacial susceptibility $\chi_D=|\partial_D\langle N\rangle|$")
    axs[1, 0].set_xlabel("D")
    axs[1, 0].set_ylabel("eta")
    fig.colorbar(im2, ax=axs[1, 0], label=r"$\chi_D$")

    # band-length diagnostic vs t_coh
    t_list = list(par.t_list)
    avgL = []
    for t in t_list:
        ph_t = compute_phase(par, _get_tcoh_fn(par, float(t)))
        band = (ph_t["margin"] < par.Delta0)
        D = ph_t["D"]
        # fraction of D in-band, averaged over eta rows
        L = [row.mean() * (D[-1] - D[0]) for row in band]
        avgL.append(float(np.mean(L)))
    x_list = [_effective_t_at_Dfix(par, float(t)) for t in t_list]
    axs[1, 1].plot(x_list, avgL, marker="o")
    axs[1, 1].set_xscale("log")
    if getattr(par, "_tcoh_mode", "fixed") == "geom":
        axs[1, 1].set_title(r"Mean band-length vs $t_{\rm coh}(D_{\rm fix})$ (geom-closed)")
    else:
        axs[1, 1].set_title(r"Mean band-length along D vs $t_{\rm coh}$")
    axs[1, 1].set_xlabel(r"$t_{\rm coh}(D_{\rm fix})$")
    axs[1, 1].set_ylabel(r"$\langle L_{\rm band}\rangle_D$")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")

    # overlay
    fig2, ax = plt.subplots(1, 1, figsize=(7.6, 5.6))
    im = ax.pcolormesh(DD, EE, ph["winner"], shading="auto")
    ax.set_title("Overlay: band + entropy + susceptibility ridges")
    ax.set_xlabel("D")
    ax.set_ylabel("eta")
    fig2.colorbar(im, ax=ax, label="$N_\\star$")

    ax.contour(DD, EE, ph["margin"], levels=[par.Delta0], linewidths=1.8, linestyles="-", colors="k")
    H = ph["H"]; chi = ph["chiD"]
    H_level = float(np.quantile(H, 0.90))
    chi_level = float(np.quantile(chi, 0.90))
    ax.contour(DD, EE, H, levels=[H_level], linewidths=1.6, linestyles="--")
    ax.contour(DD, EE, chi, levels=[chi_level], linewidths=1.6, linestyles=":")

    # Geometry-closed t_coh contours (log10) for interpretability
    if getattr(par, "_tcoh_mode", "fixed") == "geom":
        tvec = np.array([_get_tcoh_fn(par, par.t_rep)(float(d)) for d in ph["D"]], dtype=float)
        log10t = np.log10(np.maximum(tvec, 1e-300))
        LOG10 = np.tile(log10t[None, :], (len(ph["eta"]), 1))
        lo = float(np.floor(np.min(log10t)))
        hi = float(np.ceil(np.max(log10t)))
        levels = np.linspace(lo, hi, 4) if (hi - lo) >= 3 else np.linspace(lo, hi, 3)
        ax.contour(DD, EE, LOG10, levels=levels, linewidths=1.2)

    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0],[0], color="k", lw=1.8, ls="-", label=r"Band: $\Delta=\Delta_0$"),
        Line2D([0],[0], color="k", lw=1.6, ls="--", label=r"Entropy ridge $H$ (90% q)"),
        Line2D([0],[0], color="k", lw=1.6, ls=":", label=r"Susceptibility ridge $\chi_D$ (90% q)"),
    ]
    if getattr(par, "_tcoh_mode", "fixed") == "geom":
        proxies.append(Line2D([0],[0], color="k", lw=1.2, ls="-", label=r"$\log_{10} t_{\rm coh}$ contours"))
    ax.legend(handles=proxies, loc="upper right", frameon=True)
    fig2.tight_layout()
    fig2.savefig(out_overlay, dpi=220)
    print(f"Wrote {out_overlay}")

def fig_pnt_three_windows(par: Params, out_png: str = "fig_PNt_three_windows.png") -> None:
    print_params("pnt-windows", par)
    D = par.D_fix
    eta = par.eta_fix
    t_list = list(par.t_windows)

    fig, axs = plt.subplots(1, len(t_list), figsize=(14, 4), sharey=False)
    if len(t_list) == 1:
        axs = [axs]

    for ax, t in zip(axs, t_list):
        t_fn = _get_tcoh_fn(par, float(t))
        Ns, P = P_distribution_at_point(par, D, eta, t_fn)
        ax.bar(Ns, P)
        t_eff = _effective_t_at_Dfix(par, float(t))
        ax.set_title(rf"$t_{{\rm coh}}(D_{{\rm fix}}) = {t_eff:g}$")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$P_N$")
        ax.set_xlim(par.Nmin - 0.5, par.Nmax + 0.5)

    fig.suptitle(rf"Layer probabilities at fixed point $(D,\eta)=({D:g},{eta:g})$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")

def fig_kernels_scaling(par: Params, out_png: str = "fig_kernels_scaling.png") -> None:
    print_params("kernels-scaling", par)
    D = par.D_fix
    eta = par.eta_fix
    Ns = np.arange(par.Nmin, par.Nmax + 1, dtype=int)

    g = np.array([g_cardy(int(N), par.c_eff, par.nu) for N in Ns], dtype=float)
    Gam = np.array([Gamma_N(int(N), D, eta, par.Gamma_ref, par.p, par.a, par.use_Nminus1) for N in Ns], dtype=float)
    B = np.array([B_matching(int(N), par.B1, par.B2, par.B3) for N in Ns], dtype=float)

    # normalized curves
    gN = g / g.max()
    GamN = Gam / Gam.max()
    ker = (B * g * Gam) / (B * g * Gam).max()

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    ax.plot(Ns, gN, marker="o", label=r"$g_N/g_{\max}$")
    ax.plot(Ns, GamN, marker="s", label=r"$\Gamma_N/\Gamma_{\max}$")
    ax.plot(Ns, ker, marker="^", label=r"$(B_N g_N \Gamma_N)/(\cdot)_{\max}$")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("normalized scale")
    ax.set_title(rf"Kernel scalings at fixed point $(D,\eta)=({D:g},{eta:g})$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")

# ------------------------- CLI -------------------------


# ------------------------- Convergence diagnostics -------------------------

def compute_1d_scan(par: Params, tcoh, eta: float, nD: int = None, Nmax: int = None) -> Dict[str, Any]:
    """1D scan along D at fixed eta and tcoh."""
    nD = int(nD) if nD is not None else par.nD
    Nmax_eff = int(Nmax) if Nmax is not None else par.Nmax

    D_grid = np.linspace(par.Dmin, par.Dmax, nD)
    Ns = np.arange(par.Nmin, Nmax_eff + 1, dtype=int)

    logg = np.log([g_cardy(int(N), par.c_eff, par.nu) for N in Ns])
    logB = np.log([B_matching(int(N), par.B1, par.B2, par.B3) for N in Ns])

    logGamma = np.empty((nD, len(Ns)), dtype=float)
    for j, N in enumerate(Ns):
        geomN = (N - 1) if par.use_Nminus1 else N
        logGamma[:, j] = math.log(max(eta, 1e-300)) + math.log(par.Gamma_ref) - par.p * math.log(N) - par.a * D_grid * geomN

    # allow scalar tcoh or callable tcoh(D)
    if callable(tcoh):
        logt = np.log([max(float(tcoh(float(D))), 1e-300) for D in D_grid])
    else:
        logt = float(math.log(max(float(tcoh), 1e-300))) * np.ones_like(D_grid, dtype=float)
    logx = logGamma + logt[:, None]
    logF = logF_from_logx(logx)

    logQ = logB[None, :] + logg[None, :] + logF
    mmax = np.max(logQ, axis=1, keepdims=True)
    w = np.exp(logQ - mmax)
    P = w / np.sum(w, axis=1, keepdims=True)

    Nmean = (P * Ns[None, :]).sum(axis=1)

    idx = np.argsort(-logQ, axis=1)
    win_idx = idx[:, 0]
    runner_idx = idx[:, 1]
    winner = Ns[win_idx]
    Delta = logQ[np.arange(nD), win_idx] - logQ[np.arange(nD), runner_idx]

    chiD = np.abs(np.gradient(Nmean, D_grid))

    return dict(D=D_grid, Nmean=Nmean, chiD=chiD, winner=winner, Delta=Delta, Ns=Ns, Nmax=Nmax_eff, nD=nD)


def fig_grid_convergence(par: Params, tag: str) -> None:
    """Grid convergence test: refine D grid and compare Nmean(D), chiD(D)."""
    t = par.t_rep
    eta = par.eta_fix

    grids = [
        ("coarse", max(40, int(par.nD * 0.6))),
        ("default", par.nD),
        ("fine", int(par.nD * 1.6)),
    ]

    t_fn = _get_tcoh_fn(par, t)
    scans = [(name, compute_1d_scan(par, t_fn, eta, nD=nD)) for name, nD in grids]

    metrics = {"preset": "band" if par.c_eff == 1.2 else "generation", "t_rep": t, "eta_fix": eta, "grids": []}
    for name, sc in scans:
        ipeak = int(np.argmax(sc["chiD"]))
        metrics["grids"].append({
            "label": name,
            "nD": int(sc["nD"]),
            "D_peak_chiD": float(sc["D"][ipeak]),
            "chiD_max": float(sc["chiD"][ipeak]),
        })
    with open(f"grid_convergence_{tag}_metrics.json","w",encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2))

    plt.figure(figsize=(7.0, 4.2))
    for name, sc in scans:
        plt.plot(sc["D"], sc["Nmean"], label=f"{name} (nD={sc['nD']})")
    plt.xlabel("D")
    plt.ylabel(r"$\langle N \rangle(D)$")
    plt.title(rf"Grid convergence: $\langle N \rangle$ at $t={t:g}$, $\eta={eta:g}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"fig_grid_convergence_{tag}_Nmean.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    for name, sc in scans:
        plt.plot(sc["D"], sc["chiD"], label=f"{name} (nD={sc['nD']})")
        ipeak = int(np.argmax(sc["chiD"]))
        plt.scatter([sc["D"][ipeak]], [sc["chiD"][ipeak]], s=18)
    plt.xlabel("D")
    plt.ylabel(r"$\chi_D = |\partial_D \langle N \rangle|$")
    plt.title(rf"Grid convergence: $\chi_D$ at $t={t:g}$, $\eta={eta:g}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"fig_grid_convergence_{tag}_chiD.png", dpi=220)
    plt.close()

    print(f"[grid-convergence] wrote fig_grid_convergence_{tag}_Nmean.png, fig_grid_convergence_{tag}_chiD.png")
    print("[grid-convergence] metrics:", metrics["grids"])


def fig_nmax_convergence(par: Params, tag: str) -> None:
    """Nmax convergence test: increase Nmax and compare Nmean(D), chiD(D)."""
    t = par.t_rep
    eta = par.eta_fix

    nmax_list = [15, par.Nmax, 30, 40]
    # unique, sorted
    nmax_list = sorted({n for n in nmax_list if n >= par.Nmin + 2})

    scans = []
    for Nmax in nmax_list:
        sc = compute_1d_scan(par, _get_tcoh_fn(par, t), eta, nD=par.nD, Nmax=Nmax)
        scans.append((f"Nmax={Nmax}", sc))

    metrics = {"preset": "band" if par.c_eff == 1.2 else "generation", "t_rep": t, "eta_fix": eta, "cases": []}
    for label, sc in scans:
        ipeak = int(np.argmax(sc["chiD"]))
        metrics["cases"].append({
            "label": label,
            "Nmax": int(sc["Nmax"]),
            "D_peak_chiD": float(sc["D"][ipeak]),
            "chiD_max": float(sc["chiD"][ipeak]),
            "winner_Dmin": int(sc["winner"][0]),
            "winner_Dmax": int(sc["winner"][-1]),
            "Nmean_Dmin": float(sc["Nmean"][0]),
            "Nmean_Dmax": float(sc["Nmean"][-1]),
        })
    with open(f"nmax_convergence_{tag}_metrics.json","w",encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2))

    plt.figure(figsize=(7.0, 4.2))
    for label, sc in scans:
        plt.plot(sc["D"], sc["Nmean"], label=label)
    plt.xlabel("D")
    plt.ylabel(r"$\langle N \rangle(D)$")
    plt.title(rf"$N_{{\max}}$ convergence: $\langle N \rangle$ at $t={t:g}$, $\eta={eta:g}$")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"fig_Nmax_convergence_{tag}_Nmean.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    for label, sc in scans:
        plt.plot(sc["D"], sc["chiD"], label=label)
        ipeak = int(np.argmax(sc["chiD"]))
        plt.scatter([sc["D"][ipeak]], [sc["chiD"][ipeak]], s=18)
    plt.xlabel("D")
    plt.ylabel(r"$\chi_D = |\partial_D \langle N \rangle|$")
    plt.title(rf"$N_{{\max}}$ convergence: $\chi_D$ at $t={t:g}$, $\eta={eta:g}$")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"fig_Nmax_convergence_{tag}_chiD.png", dpi=220)
    plt.close()

    print(f"[nmax-convergence] wrote fig_Nmax_convergence_{tag}_Nmean.png, fig_Nmax_convergence_{tag}_chiD.png")
    print("[nmax-convergence] metrics:", metrics["cases"])
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--preset", default="band", choices=["band","generation"], help="Named parameter preset.")
        sp.add_argument("--use-N", action="store_true", help="Use exp(-a D N) instead of exp(-a D (N-1)).")

        # t_coh handling
        sp.add_argument("--tcoh-mode", default="fixed", choices=["fixed","geom"],
                        help="Coherence-time mode: fixed uses preset t; geom uses t * tcoh_geom(D) from a curve table.")
        sp.add_argument("--tcoh-curve", default="tcoh_geom_curve_axis1d.csv",
                        help="CSV providing geometry-closed tcoh_geom(D). Used when --tcoh-mode=geom.")
        sp.add_argument("--tcoh-extrap", default="linear-log", choices=["linear-log","clamp","error"],
                        help="How to handle D outside curve domain when --tcoh-mode=geom.")
        sp.add_argument("--tcoh-keep-domain", action="store_true",
                        help="Do not auto-restrict (Dmin,Dmax) to the curve domain when using geom mode.")

    sp1 = sub.add_parser("transition-band", help="Generate transition-band diagnostics figures.")
    add_common(sp1)

    sp2 = sub.add_parser("pnt-windows", help="Generate P_N bar plots at three coherence windows.")
    add_common(sp2)

    sp3 = sub.add_parser("kernels-scaling", help="Generate normalized g_N, Gamma_N and early-time kernel scaling.")
    add_common(sp3)

    sp4 = sub.add_parser("grid-convergence", help="Grid (D-mesh) convergence test for Nmean and chiD.")
    add_common(sp4)

    sp5 = sub.add_parser("nmax-convergence", help="N_max cutoff convergence test for Nmean and chiD.")
    add_common(sp5)

    sp6 = sub.add_parser("all", help="Generate all figures for a preset (including convergence appendices).")
    add_common(sp6)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    par = preset(args.preset)
    if args.use_N:
        par.use_Nminus1 = False

    # t_coh mode setup
    par._tcoh_mode = str(getattr(args, "tcoh_mode", "fixed"))
    if par._tcoh_mode == "geom":
        par._tcoh_curve = TcohCurve(getattr(args, "tcoh_curve", "tcoh_geom_curve_axis1d.csv"), extrap=getattr(args, "tcoh_extrap", "linear-log"))
        if not getattr(args, "tcoh_keep_domain", False):
            par.Dmin = max(float(par.Dmin), float(par._tcoh_curve.Dmin))
            par.Dmax = min(float(par.Dmax), float(par._tcoh_curve.Dmax))

    if args.cmd in ("transition-band", "all"):
        fig_transition_band(par)
    if args.cmd in ("pnt-windows", "all"):
        fig_pnt_three_windows(par)
    if args.cmd in ("kernels-scaling", "all"):
        fig_kernels_scaling(par)
    if args.cmd in ("grid-convergence", "all"):
        fig_grid_convergence(par, args.preset)
    if args.cmd in ("nmax-convergence", "all"):
        fig_nmax_convergence(par, args.preset)

if __name__ == "__main__":
    main()
