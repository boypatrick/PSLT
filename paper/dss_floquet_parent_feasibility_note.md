# Large-d DSS/Floquet Parent Feasibility Note

Status: D0--D3 closed as non-baseline feasibility gates.  D3 closes negative for parent promotion.

Source under test: arXiv:2601.14358, "Analytic discrete self-similar solutions of Einstein-Klein-Gordon at large D".

## D0: Symbol And Claim Firewall

This line is explicitly outside the PSLT submission baseline.  It does not modify `paper/main.tex`, baseline maps, figure generation, or the A1--A5 wording.

Notation is frozen as follows:

- `d`: spacetime dimension in the large-d Einstein--Klein--Gordon parent.
- `D_sep`: PSLT two-center separation parameter.
- `Delta`: DSS echoing period.

The DSS/Floquet parent line is only a parent-side gravitational-scalar feasibility route.  It is not a proof of:

- exactly three SM generations,
- gauge charges,
- `g_N`,
- `B_N`,
- `H->mumu`,
- or the current release `P_N` map.

The scientific question is narrower: can a large-d discrete-self-similar Einstein--Klein--Gordon parent provide a controlled analytic toy parent for the PSLT spectral-layer hierarchy?

## D1: LO/NLO DSS Consistency Reproduction

The arXiv paper rewrites the spherically symmetric Einstein--massless--Klein--Gordon system using

```text
epsilon = 1/(d - 1),
tau = -log(-t),
x = -r/t.
```

At leading order, the relevant large-d DSS expressions are

```text
Omega_LO = Pi_LO^2,

Pi_LO(tau,x) = beta(tau) / sqrt(1 + beta(tau)^2 x^2),

f_LO(tau,x) = sqrt((1 + beta(tau)^2 x^2)/(1 + beta(tau)^2)).
```

The paper's example periodic profile is

```text
beta(tau) = cos(2 pi tau) + sin(6 pi tau)/A,
A ~= 15.9476,
Delta = 1.
```

The NLO consistency condition for the echoing period is

```text
Delta = |beta''| / (3 |beta'|) evaluated at beta = 0.
```

Executable gate:

```bash
python3 code/check_dss_lo_consistency.py
```

Current output:

```text
D1_LO_NLO_DSS_CONSISTENCY_REPRODUCED
A_ref=15.9476000000
tau0=0.2401842459302251
Delta=9.9999996405309244e-01
A_solved=1.5947599366191568e+01
Delta_solved=1.0000000000000153e+00
```

Registered artifacts:

- `output/dss_floquet/dss_lo_consistency_summary.csv`
- `output/dss_floquet/dss_lo_consistency_detail.json`

Numerical summary:

| Check | Value |
| --- | ---: |
| `abs(Delta(A_ref)-1)` | `3.5946907561701380e-08` |
| `beta''+3 beta'` at rounded `A_ref` zero | `-6.9974443661635632e-07` |
| solved `A` for `Delta=1` | `15.947599366191568` |
| `beta''+3 beta'` at solved `A` zero | `2.9842794901924208e-13` |
| max periodicity residual | `2.1094237467877974e-15` |
| max half-period antisymmetry residual | `1.7763568394002505e-15` |
| max `Omega_LO - Pi_LO^2` residual | `0.0` |
| max SSH boundary residual `f_LO(x=1)-1` | `0.0` |
| min `f_LO-x` on sampled grid | `0.0` |
| max `f_LO-1` on sampled grid | `0.0` |

Interpretation:

- D1 reproduces the quoted DSS example and the NLO echoing-period consistency condition.
- The rounded paper value `A=15.9476` is sufficient for the roadmap gate.
- The exact solved value is close enough that the residual collapses to machine precision.
- This validates the arithmetic starting point for a possible D2 monodromy/Floquet test.

## D2: Toy Floquet Monodromy Spectrum

D2 asks a deliberately narrower question than a parent promotion theorem:

```text
Does the D1 periodic DSS profile support a stable ordered toy Floquet spectrum?
```

The tested reduced proxy is a Hamiltonian Hill system,

```text
y_N'' + [omega_N^2 + kappa q(tau)] y_N = 0,
omega_N = pi (N - 1/2),
```

with the periodic modulation obtained from the LO DSS geometry,

```text
I_LO(tau) = int_0^1 Omega_LO(tau,x) dx = |beta(tau)| atan(|beta(tau)|),
q(tau) = I_LO(tau) - mean(I_LO).
```

The one-period monodromy matrix is computed by RK4 integration of the
fundamental matrix over `Delta=1`; the Floquet quasi-frequency is unwrapped
onto the branch closest to the unperturbed `omega_N`.  Because this is a
Hamiltonian toy problem, the expected positive outcome is unit-circle
stability and ordered phase bands.  It does not define `Gamma_N`, does not
claim growth/decay, and does not modify the PSLT baseline.

Executable gate:

```bash
python3 code/check_dss_floquet_monodromy.py
```

Current output:

```text
D2_TOY_FLOQUET_PHASE_ORDERING_STABLE_NO_GROWTH_CLAIM
kappa=0 stable_all=True ordering_pass=True min_gap=3.1415926535878071e+00
kappa=0.25 stable_all=True ordering_pass=True min_gap=3.1415818255576164e+00
kappa=0.5 stable_all=True ordering_pass=True min_gap=3.1415493419832021e+00
kappa=1 stable_all=True ordering_pass=True min_gap=3.1414194097883561e+00
```

Registered artifacts:

- `output/dss_floquet/dss_floquet_monodromy_summary.csv`
- `output/dss_floquet/dss_floquet_monodromy_detail.csv`
- `output/dss_floquet/dss_floquet_monodromy_detail.json`

Numerical summary:

| `kappa` | Stable on unit circle | Ordered | Minimum quasi-frequency gap | Max `|log |mu||` | Max `|det M - 1|` | Max quasi-frequency shift |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| `0` | yes | yes | `3.1415926535878071e+00` | `6.9944050551385107e-15` | `1.3766765505351941e-14` | `3.1334934647020418e-12` |
| `0.25` | yes | yes | `3.1415818255576164e+00` | `7.2164496600635443e-15` | `1.3766765505351941e-14` | `1.0372276388226354e-05` |
| `0.5` | yes | yes | `3.1415493419832021e+00` | `7.9936057773011586e-15` | `1.5987211554602254e-14` | `4.1488246317555877e-05` |
| `1` | yes | yes | `3.1414194097883561e+00` | `3.9968028886505557e-15` | `8.4376949871511897e-15` | `1.6594349880616477e-04` |

Interpretation:

- D2 is nonnegative: the D1 periodic DSS geometry does not destroy ordered low-mode Floquet bands in this reduced Hamiltonian test.
- The gaps remain close to the unperturbed spacing `pi`, with only small perturbative shifts up to `kappa=1`.
- The unit-circle multipliers are a consistency feature of the Hamiltonian proxy, not a growth-rate prediction.
- The result is strong enough to justify D3 as a projected-conformal bridge attempt, but not strong enough to alter `paper/main.tex` or the release `P_N` map.

## D3: Projected-Conformal Bridge Attempt

D3 tests the necessary condition behind any parent-side conformal promotion:

```text
Can a projected DSS profile approximate the PSLT two-center Omega with C^2 control?
```

The `C^2` requirement is not cosmetic.  The PSLT scalar operator uses

```text
V_eff = m0^2 Omega^2 + (1 - 6 xi) Omega^{-1} nabla^2 Omega,
```

so a perturbation `delta Omega` enters as

```text
delta V_eff =
  [2 m0^2 Omega - (1 - 6 xi) Omega^{-2} nabla^2 Omega] delta Omega
  + (1 - 6 xi) Omega^{-1} nabla^2 delta Omega
  + O(delta Omega^2).
```

Thus any bridge that cannot control at least `delta Omega`, `d_z delta Omega`,
and `d_z^2 delta Omega` on the axial restriction cannot control `V_eff` in the
full geometry.  The axial test is only a necessary condition: a full
two-dimensional/three-dimensional `C^2` bridge would imply the axial one.
Failure on the axial gate therefore blocks D4.

The tested candidate family is intentionally generous:

```text
Omega_DSS^R(z)
  = c0 + c1 [F(|z-D_sep/2|/R) + F(|z+D_sep/2|/R)],

F(x) = < beta(tau)^2 / (1 + beta(tau)^2 x^2) >_tau.
```

For each axial window, the script scans `R` and fits `c0,c1` against the
target `Omega`, `d_z Omega`, and `d_z^2 Omega` with scale-normalized least
squares.  It reports two policies:

- `domain_safe`: all lobe coordinates stay in the LO DSS chart `x <= 1`;
- `analytic_extension`: the same formula is analytically extended to `x > 1`
  as a diagnostic stress test.

The adoption criterion is deliberately hard:

```text
rel C0 <= 0.05, rel C1 <= 0.10, rel C2 <= 0.20
```

on all required windows under the `domain_safe` policy.

Executable gate:

```bash
python3 code/check_dss_projected_conformal_bridge.py
```

Current output:

```text
D3_PROJECTED_CONFORMAL_C2_BRIDGE_FAILED_DIAGNOSTIC_ONLY
```

Registered artifacts:

- `output/dss_floquet/dss_projected_conformal_bridge_summary.csv`
- `output/dss_floquet/dss_projected_conformal_bridge_detail.csv`
- `output/dss_floquet/dss_projected_conformal_bridge_detail.json`

Domain-safe numerical summary:

| Window | Required | Pass | Best `R` | rel `C0` | rel `C1` | rel `C2` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `full_axial` | yes | no | `18.60335195530726` | `0.3299171869876872` | `0.870656122415794` | `1.001809537624721` |
| `core_excluded_2p5eps` | yes | no | `9.5` | `0.09303930411793862` | `0.7475804885033633` | `0.9748112409937157` |
| `shoulder_excluded_5eps` | yes | no | `9.0` | `0.0604723978375798` | `0.43274591064664847` | `0.8798956871054806` |
| `central_barrier_only_10eps` | no | no | `8.0` | `0.005180460020711656` | `0.047183279892660225` | `0.2795242912561587` |

Analytic-extension diagnostic:

| Window | Required | Pass | Best `R` | Max `x` used | rel `C0` | rel `C1` | rel `C2` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full_axial` | yes | no | `0.2670391061452514` | `37.44769874476987` | `0.027915759288744173` | `0.12477935239767864` | `0.212569546994096` |
| `core_excluded_2p5eps` | yes | no | `0.2670391061452514` | `35.57531380753138` | `0.07822571072588154` | `0.05876992614806835` | `0.1489792782073488` |
| `shoulder_excluded_5eps` | yes | yes | `0.4340782122905028` | `20.733590733590734` | `0.02961851686781023` | `0.050532431043330765` | `0.0806051238918185` |
| `central_barrier_only_10eps` | no | yes | `0.7681564245810055` | `10.414545454545456` | `0.004553104687649229` | `0.031575959516648426` | `0.050523375880851866` |

Interpretation:

- D3 fails under the physically safer `domain_safe` policy by a wide margin,
  especially in `C1` and `C2`.
- The analytic extension can fit the central/shoulder region better, but it
  uses `x >> 1`, outside the conservative LO DSS chart, and still fails the
  required full/core windows.
- Therefore the large-`d` DSS/Floquet line remains useful as geometric
  intuition and toy Floquet evidence, but it is not a parent-side conformal
  bridge for the present PSLT `Omega -> V_eff -> omega_N -> S_N` chain.

## Stop Rule After D3

D3 closes the bridge attempt negative.  Do not proceed to D4--D6 unless a genuinely new projection mechanism is proposed.  The present DSS/Floquet line is diagnostic-only and does not modify `paper/main.tex`, baseline maps, or the release `P_N` chain.
