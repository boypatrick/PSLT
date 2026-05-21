# PINN V3 Downstream Map Coupling Plan

V3 starts only after the V2.6 seed-stability freeze gate.  Its purpose is not
to turn the PINN sandbox into a proof engine.  Its purpose is to expose a
stable differentiable spectral table that downstream PSLT map code can consume
without silently changing the manuscript baseline.

## Mathematical Interface

The frozen V2 emulator represents a \(D\)-dependent Ritz subspace for the
self-adjoint cylindrical operator

```math
H_D u_k(D) = E_k(D) u_k(D),
```

with diagnostic frequencies

```math
\omega_k(D)=\sqrt{1+E_k(D)}.
```

V3.0 exports only the map

```math
D \longmapsto \{E_k(D),\omega_k(D)\}_{k=0}^{2}
```

together with the same Gram and strong-residual guards used in V2.  It does
not compute

```math
S_N(D),\qquad r_N(D,\eta)=\eta e^{-2S_N(D)},\qquad
\Gamma_N(D,\eta),\qquad P_N(D,\eta).
```

Those quantities require a separate energy-to-action adapter

```math
S_N(D)=\mathcal A[D,E_N(D)]
```

that is not assumed by the V2 self-adjoint emulator.  Until that adapter is
audited, the V3 table is an input-side coupling artifact only.

## V3.0 Spectral Export Status

Tracked outputs:

- `v3_downstream_spectral_table.csv`
- `v3_downstream_spectral_metrics.json`
- `v3_downstream_manifest.json`

Command:

```bash
D_VALUES=$(python3 - <<'PY'
vals=[6.0+0.25*i for i in range(49)]
print(','.join(f'{v:g}' for v in vals))
PY
)
pinn/.venv/bin/python pinn/evaluate_v2_parametric_d.py \
  --run v2_parametric_D6_18_K3_augmented_final_800 \
  --D-values "$D_VALUES" \
  --e2-turning-D 14.0 \
  --device auto \
  --run-name v3_downstream_spectral_export_d025_turn14
```

The denser \(D=6,\ldots,18\) step-0.25 export shows the third branch turning
near \(D\simeq14.0\), not at the earlier quarter-holdout guard value
\(D=13.5\).  With the corrected branch-turning metadata:

```text
max_dense_corr_offdiag = 4.998e-02 < 5e-2
max_dense_residual_L2  = 1.234e-01 < 3e-1
anchor_consistent_monotone_ok = true
suspicious_D = none
needs_finite_volume_check = false
```

Current V3.0 status:

```text
OPENED POSITIVE / EXPORT-ONLY DOWNSTREAM SPECTRAL ADAPTER
```

## Next V3 Gate

The next meaningful V3 task is not more neural training.  It is an audited
energy-to-action bridge:

```math
(D,E_k(D)) \mapsto S_k(D)
```

with deterministic finite-volume or single-track cross-checks.  Only after
that bridge passes should V3 compute \(r_N\), \(\Gamma_N\), or \(P_N\).

Recommended V3.1 gate:

```text
Given the V3.0 spectral table, construct an explicit action adapter
S_k^{PINN}(D) and compare it against canonical deterministic action artifacts.
If the map changes the baseline R3/winner topology, keep it diagnostic-only.
```

Submission policy:

```text
V3 remains outside the paper baseline.  It is a PINN-assisted differentiable
emulator interface, not a replacement for finite-volume certificates or the
existing PSLT release maps.
```

## V3.1 Energy-to-Action Bridge Status

Tracked outputs:

- `v3_energy_to_action_true_crosscheck.csv`
- `v3_energy_to_action_detail.csv`
- `v3_energy_to_action_summary.json`

Command:

```bash
pinn/.venv/bin/python pinn/bridge_v3_energy_to_action.py \
  --out-prefix v3_energy_to_action
```

The adapter uses the action-derived axial shifted potential

```math
U(0,z;D)
=m_0^2(\Omega^2-1)
+(1-6\xi)\Omega^{-1}\Delta\Omega
```

and computes

```math
S(D,E)=\int_{\mathcal B(D,E)}
\sqrt{(U(0,z;D)-E)_+}\,dz,
```

where \(\mathcal B(D,E)\) is the central forbidden component when one exists.
For negative-energy single-track bound states, the script uses the same inner
turning-crossing convention as `output/true_single_track/true_results.json`.

Canonical deterministic cross-check:

```text
true_crosscheck_max_abs_S_error = 3.55e-15
true_crosscheck_max_rel_S_error = 2.30e-16
true_crosscheck_pass = true
```

Applying the same adapter to the V3.0 PINN self-adjoint spectral export gives:

```text
n_bridge_rows = 147
status_counts = CENTRAL_BARRIER: 63, NEAREST_FORBIDDEN_ISLAND: 84
mode 0 central-barrier D range: 6.0..11.75
mode 1 central-barrier D range: 6.0..11.75
mode 2 central-barrier D range: 6.0..9.5
```

Current V3.1 status:

```text
DIAGNOSTIC BRIDGE ONLY / ADAPTER VALIDATED, OPERATOR IDENTIFICATION NOT CLOSED
```

Reading:

- the action adapter exactly reproduces the canonical deterministic
  single-track WKB artifact;
- the PINN V3.0 self-adjoint spectrum is not the same spectral object as the
  legacy negative-energy single-track bound artifact;
- therefore V3.1 must not be promoted to \(\Gamma_N\), \(P_N\), or a release
  map without a separate operator-identification gate.

Recommended next step:

```text
V3.2 operator-identification gate: decide whether the self-adjoint cylindrical
PINN spectrum can be mapped to the single-track WKB action object, or close the
V3 line as a diagnostic spectral emulator only.
```

## V3.2 Operator-Identification Gate Status

Tracked outputs:

- `v3_operator_identification_detail.csv`
- `v3_operator_identification_summary.json`

Command:

```bash
pinn/.venv/bin/python pinn/check_v3_operator_identification.py
```

The gate compares three objects:

```math
\{E_{\rm true}^{\rm single-track}(D)\},
\qquad
\{E_k^{\rm PINN}(D)\}_{k=0}^2,
\qquad
\mathcal A[D,E]\mapsto S(D,E).
```

Numerical evidence:

```text
true_energy_range          = [-0.9205263085, -0.0723988815]
selfadjoint_energy_range   = [ 0.7413694263,  1.1093039513]
selfadjoint_anchor_n_neg   = 0
central_barrier_fraction   = 63 / 147 = 0.4285714286
direct_operator_identification_pass = false
```

Current V3.2 status:

```text
CLOSED NEGATIVE / DIAGNOSTIC ONLY
```

Mathematical reading:

- V3.1 validates the action functional on the canonical deterministic
  single-track artifact;
- the V3 PINN export is a positive finite-volume self-adjoint spectral table,
  while the canonical single-track WKB artifact is a negative-energy bound
  chain;
- the action components induced by the PINN table are not uniformly central
  barrier components;
- therefore there is no direct operator identification and no permission to
  feed \(S_k^{\rm PINN}\) into \(r_N\), \(\Gamma_N\), \(P_N\), or release maps.

## V4/V5 Quarantine And Closure

Tracked outputs:

- `v4_downstream_quarantine_summary.json`
- `v5_pinn_closure_summary.json`

Commands:

```bash
pinn/.venv/bin/python pinn/check_v4_downstream_quarantine.py
pinn/.venv/bin/python pinn/check_v5_pinn_closure.py
```

V4 passes because V3.2 is negative and no PINN-derived release-map artifacts
exist:

```text
status = QUARANTINE_PASS_DIAGNOSTIC_ONLY
blocked = r_N, Gamma_N, P_N, release maps, paper baseline figures
```

V5 closes the full PINN line:

```text
status = PINN_SANDBOX_CLOSED_DIAGNOSTIC_ONLY
closure_pass = true
```

Final policy:

```text
The PINN line is frozen as an optional differentiable spectral
emulator/diagnostic package.  It can nominate deterministic finite-volume
follow-up points or diagnostic plots, but it does not alter paper/main.tex,
the release baseline, Gamma_N, P_N, or submission figures.
```
