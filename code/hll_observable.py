#!/usr/bin/env python3
"""
Shared H->ll observable mapping utilities.

Two map-level modes are supported:
  - proxy_wratio:    mu = W_N / W_N(ref)
  - eft_wilson_diag: mu = |c_ll / c_ll(ref)|^2 with
                     c_ll = y_eff_raw_N(D) * P_N^(kin)(D,eta)
  - eft_wilson_matched:
                     C_{eH}^{ij} = sum_N Y_{iN} P_N^(kin) Y_{jN}
                     mu_ll = (|C_ii/C_ii(ref)|^2) / (Gamma_tot/Gamma_tot(ref))
  - eft_wilson_uv_tree:
                     C_{eH}^{ij} = sum_N g_{iN}(D) [P_N^(kin)/M_N^2(D)] g_{jN}(D)
                     mu_ll = (|C_ii/C_ii(ref)|^2) / (Gamma_tot/Gamma_tot(ref))
  - eft_wilson_uv_rge:
                     C_{eH}^{ij}(mu_low) from UV-tree matrix
                     + finite one-loop matching + leading-log running
                     mu_ll = (|C_ii/C_ii(ref)|^2) / (Gamma_tot/Gamma_tot(ref))
"""

from __future__ import annotations

from dataclasses import dataclass

from pslt_lib import PSLTKinetics


@dataclass(frozen=True)
class HLLObservableConfig:
    mode: str = "eft_wilson_uv_rge"
    t_coh: float = 1.0
    ref_D: float = 10.0
    ref_eta: float = 1.0
    n_max: int = 20

    def __post_init__(self) -> None:
        if self.mode not in {"proxy_wratio", "eft_wilson_diag", "eft_wilson_matched", "eft_wilson_uv_tree", "eft_wilson_uv_rge"}:
            raise ValueError(f"Unsupported HLL observable mode: {self.mode}")
        if self.n_max < 3:
            raise ValueError("n_max must be >= 3")
        if self.t_coh <= 0.0:
            raise ValueError("t_coh must be > 0")


class HLLChannelPredictor:
    def __init__(self, kinetics: PSLTKinetics, layer_n: int, cfg: HLLObservableConfig):
        self.kinetics = kinetics
        self.layer_n = int(layer_n)
        self.cfg = cfg
        self.ref_amp = self.channel_amplitude(cfg.ref_D, cfg.ref_eta)
        if self.ref_amp <= 0.0:
            raise RuntimeError(
                f"Non-positive reference amplitude for layer N={self.layer_n} "
                f"in mode={self.cfg.mode} at (D,eta)=({self.cfg.ref_D},{self.cfg.ref_eta})."
            )

    def channel_amplitude(self, d_val: float, eta_val: float) -> float:
        return float(
            self.kinetics.hll_channel_amplitude(
                self.layer_n,
                float(d_val),
                float(eta_val),
                float(self.cfg.t_coh),
                observable_mode=self.cfg.mode,
                N_max=self.cfg.n_max,
            )
        )

    def mu_pred(self, d_val: float, eta_val: float) -> float:
        if self.cfg.mode in {"eft_wilson_matched", "eft_wilson_uv_tree", "eft_wilson_uv_rge"}:
            return float(
                self.kinetics.hll_mu_pred(
                    layer_n=self.layer_n,
                    D=float(d_val),
                    eta=float(eta_val),
                    t_coh=float(self.cfg.t_coh),
                    ref_D=float(self.cfg.ref_D),
                    ref_eta=float(self.cfg.ref_eta),
                    observable_mode=self.cfg.mode,
                    N_max=int(self.cfg.n_max),
                )
            )
        amp = self.channel_amplitude(d_val, eta_val)
        ratio = float(amp / max(self.ref_amp, 1e-30))
        if self.cfg.mode == "proxy_wratio":
            return ratio
        return ratio * ratio
