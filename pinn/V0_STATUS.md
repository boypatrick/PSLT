# V0 PINN Status: Two-Center Poisson Conformal Factor

Date: 2026-05-18

## Scope

V0 targets only the two-center Poisson geometry

```math
\Omega(\rho,z;D)=1+a(r_+^{-1}+r_-^{-1}),
\qquad
\nabla^2\Omega=-4\pi\sigma.
```

It does not touch the manuscript, the spectral eigenvalue solver, the Rank-2
kinetic map, or any Standard Model interpretation.

## Mathematical Setup

The trainable coordinate is

```math
s=\rho^2.
```

For a smooth axisymmetric field \(\Omega(\rho,z)=u(s,z)\),

```math
\partial_{\rho\rho}\Omega+\rho^{-1}\partial_\rho\Omega
=4s u_{ss}+4u_s.
```

The residual used by `train_two_center_poisson_pinn.py` is therefore

```math
\mathcal R_\theta
=4s\partial_s^2\Omega_\theta
+4\partial_s\Omega_\theta
+\partial_z^2\Omega_\theta
+4\pi\sigma(s,z).
```

This avoids the explicit \(1/\rho\) coordinate singularity at the axis.

## Analytic Identity Check

Command:

```bash
pinn/.venv/bin/python pinn/validate_poisson_identity.py
```

Result:

```text
max_abs_error=6.478826e-06
median_abs_error=2.023953e-08
max_rel_error=2.123396e-04
median_rel_error=4.169667e-06
```

This validates the finite-difference check of

```math
\nabla^2(r^2+\epsilon^2)^{-1/2}
=-3\epsilon^2(r^2+\epsilon^2)^{-5/2}.
```

## Smoke Runs

### `v0_smoke`

Pure PDE/boundary smoke, 100 steps.

```text
omega_rmse=1.4062627554
omega_median_rel=0.3789888620
residual_rmse=74.7152252197
residual_median_abs=0.1559474915
```

Reading: executable but not numerically useful.

### `v0_data_smoke`

Data-assisted smoke, 300 steps.

```text
omega_rmse=0.6094605327
omega_median_rel=0.0341817774
residual_rmse=52.7070732117
residual_median_abs=0.0640165284
```

Reading: value fit improves, residual still dominated by Plummer-core samples.

### `v0_warmstart_1k`

Data-assisted warm start, 1000 steps.

```text
omega_rmse=0.3762116730
omega_median_abs=0.0119377375
omega_median_rel=0.0098390589
residual_rmse=45.4038124084
residual_median_abs=0.0108385701
```

Reading: good median-value smoke result, but not a closed PINN solver.  The
maximum error and residual are still controlled by the sharp \(\epsilon=0.2\)
source cores.

## Current Verdict

V0 is active and executable:

- PyTorch is installed inside `pinn/.venv`.
- Training outputs are isolated under ignored `pinn/runs/`.
- The \(s=\rho^2\) residual works.
- The best short run reaches roughly one-percent median relative Omega error.

V0 is not yet production-grade:

- residual RMSE remains large near the Plummer cores;
- max relative error is not controlled;
- no adaptive/core-weighted curriculum has been closed;
- no comparison plot against the deterministic solver has been generated yet.

## Suggested Next Steps

1. Add core-split metrics: report errors inside and outside balls around
   \(p_\pm\).
2. Add a training curriculum: value warm start, then gradually increase
   `w_pde`.
3. Add diagnostic plots for \(\Omega_\theta-\Omega_{\rm exact}\) and residual
   heatmaps.
4. Only after V0 has stable holdout and residual gates, move to fixed-\(D\)
   Ritz/eigen-PINN.


## V0.1 Core-Split Metrics and Diagnostics

Implemented after the initial smoke runs:

- uniform holdout metrics;
- source-biased trainlike holdout metrics;
- core/bulk split using radius `3 eps` around each projected center;
- curriculum weights `w_pde_eff`, `w_data_eff` in `history.csv`;
- optional diagnostic heatmap `diagnostics.png` for exact Omega, PINN error, and Poisson residual.

### `v0_curriculum_1p5k`

Aggressive curriculum: data weight decreases while PDE weight ramps to `0.02`.

```text
uniform_omega_median_rel=0.0505870469
uniform_residual_median_abs=0.0048227035
trainlike_omega_median_rel=0.0540537499
trainlike_residual_median_abs=0.0052256063
trainlike_residual_core_abs_median_abs=21.4767417908
trainlike_residual_bulk_abs_median_abs=0.0048010349
```

Reading: excellent bulk/median residual, but value accuracy deteriorates.  The
PDE term is too strong relative to the value fit and still does not resolve the
Plummer cores.

### `v0_balanced_1p5k`

Conservative curriculum: data weight remains high and PDE ramps only to `0.002`.

```text
uniform_omega_median_rel=0.0235332381
uniform_residual_median_abs=0.0092392415
trainlike_omega_median_rel=0.0279835090
trainlike_residual_median_abs=0.0102759991
trainlike_residual_core_abs_median_abs=48.7723770142
trainlike_residual_bulk_abs_median_abs=0.0091400119
```

Reading: better value accuracy than the aggressive curriculum, but worse than
the pure warm-start value fit.  The core/bulk split shows the real bottleneck:
bulk residuals are already small, while core residuals around the sharp
Plummer sources remain uncontrolled.

## Updated V0 Verdict

V0.1 is a useful diagnostic PINN pipeline:

- it produces reproducible training artifacts;
- it separates bulk and core errors;
- it exposes the core residual bottleneck instead of hiding it in one scalar
  loss;
- it can generate error/residual heatmaps for visual inspection.

V0.1 is not yet a converged Poisson PINN solver:

- max errors remain large;
- core residual median remains order `10--50` in the current runs;
- uniform random holdout undersamples the small core region, so source-biased
  holdouts must be retained;
- the next improvement should be core-adaptive sampling or an analytic singular
  split rather than merely longer training.

## Recommended V0.2 Direction

The most promising next step is an analytic-background split:

```math
\Omega_\theta = \Omega_{\rm sing} + \delta\Omega_\theta,
```

where `Omega_sing` contains the two Plummer Green functions and the PINN learns
only a smooth correction or boundary/domain-deformation residual.  For the
current exact two-center Poisson test, this correction should be zero; that
makes it a sharp sanity check for whether the PINN architecture is inventing
neural artifacts.

If we keep the direct PINN route instead, V0.2 should add adaptive core sampling
and a separate core-weighted residual loss.

## V0.2 Analytic Singular Split

Implemented file:

- `train_two_center_poisson_split_pinn.py`

The split is

```math
\Omega_\theta(s,z)=\Omega_{\rm sing}(s,z)+\delta_\theta(s,z),
```

with

```math
\Omega_{\rm sing}
=1+a\left[
\bigl(s+(z-D/2)^2+\epsilon^2\bigr)^{-1/2}
+
\bigl(s+(z+D/2)^2+\epsilon^2\bigr)^{-1/2}
\right].
```

For one Plummer center,

```math
\left(4s\partial_s^2+4\partial_s+\partial_z^2\right)
\bigl(s+(z-z_0)^2+\epsilon^2\bigr)^{-1/2}
=-\frac{3\epsilon^2}
{\bigl(s+(z-z_0)^2+\epsilon^2\bigr)^{5/2}}.
```

Therefore

```math
\left(4s\partial_s^2+4\partial_s+\partial_z^2\right)\Omega_{\rm sing}
+4\pi\sigma=0.
```

The trainable correction satisfies the homogeneous equation

```math
\left(4s\partial_s^2+4\partial_s+\partial_z^2\right)\delta_\theta=0,
\qquad
\delta_\theta|_{\partial\mathcal D}=0.
```

By the maximum principle for the corresponding axisymmetric harmonic
correction, the exact V0 two-center Poisson test has the unique solution

```math
\delta_\theta\equiv0.
```

This is the main V0.2 point: the neural network no longer has to represent the
sharp Plummer source.  The source is handled analytically, and the network is
only a monitor for artificial homogeneous corrections.

### `v0p2_split_exact_zero_smoke`

Command:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_split_pinn.py \
  --steps 1 \
  --n-interior 512 \
  --n-boundary 128 \
  --n-eval 1024 \
  --hidden 32 \
  --layers 3 \
  --log-every 1 \
  --make-plots \
  --plot-grid 64 \
  --run-name v0p2_split_exact_zero_smoke
```

Result:

```text
uniform_delta_max_abs=0.0
uniform_omega_median_rel=0.0
uniform_laplace_residual_median_abs=0.0
trainlike_delta_max_abs=0.0
trainlike_omega_median_rel=0.0
trainlike_laplace_residual_core_abs_median_abs=0.0
```

Reading: with zero-initialized final correction layer, the exact split remains
exactly at the analytic solution.  This is expected and is the cleanest
sanity-gate for the singular split.

### `v0p2_split_random_stress_300`

Command:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_split_pinn.py \
  --random-final \
  --steps 300 \
  --n-interior 1024 \
  --n-boundary 256 \
  --n-eval 2048 \
  --hidden 64 \
  --layers 4 \
  --log-every 100 \
  --w-pde 1 \
  --w-boundary 20 \
  --w-data 10 \
  --make-plots \
  --plot-grid 72 \
  --run-name v0p2_split_random_stress_300
```

Result:

```text
uniform_delta_median_abs=1.5677837655e-04
uniform_omega_median_rel=1.2901661103e-04
uniform_laplace_residual_median_abs=1.1751771672e-04
trainlike_delta_median_abs=1.1520413682e-04
trainlike_omega_median_rel=5.6688102632e-05
trainlike_laplace_residual_core_abs_median_abs=1.9697010430e-05
trainlike_laplace_residual_bulk_abs_median_abs=1.1217853171e-04
```

Reading: even from a random final correction layer, the homogeneous correction
is rapidly driven near zero.  More importantly, the previous V0.1 core residual
bottleneck disappears: the core median homogeneous residual is about
`2e-5`, not order `10--50`.

### `v0p2_singular_split_d_sweep`

Implemented file:

- `validate_singular_split.py`

Command:

```bash
pinn/.venv/bin/python pinn/validate_singular_split.py \
  --D 6 12 18 \
  --n-uniform 4096 \
  --n-core 4096 \
  --json-out pinn/runs/v0p2_singular_split_d_sweep.json
```

Result:

```text
D=6  uniform_median_abs=1.176e-18  uniform_max_abs=1.421e-14  core_median_abs=5.329e-15  core_max_abs=1.705e-13
D=12 uniform_median_abs=1.046e-18  uniform_max_abs=5.773e-15  core_median_abs=5.329e-15  core_max_abs=2.274e-13
D=18 uniform_median_abs=9.029e-19  uniform_max_abs=9.992e-16  core_median_abs=5.329e-15  core_max_abs=1.705e-13
```

Reading: the analytic split identity is not a single-\(D\) artifact.  The same
autograd derivative structure used by the split PINN validates
\(L_s\Omega_{\rm sing}+4\pi\sigma=0\) to float64 precision on both uniform and
source-biased holdouts for representative small/mid/large separations.

## Updated V0.2 Verdict

V0.2 closes the exact two-center Poisson geometry test as an analytic split:

- the Plummer source cores are absorbed by \(\Omega_{\rm sing}\);
- the learned object is the source-free correction \(\delta_\theta\);
- the exact correction is zero by the maximum principle;
- the zero-initialized split gives machine-zero diagnostics;
- the random-final stress run relaxes to a small correction without a core
  residual explosion.
- the singular identity is validated across \(D=6,12,18\) with source-biased
  holdouts at float64 precision.

This should be treated as a geometry-side PINN sanity closure, not as a
spectral/eigenvalue solver and not as a manuscript theorem.  The next useful
PINN task is no longer longer training on the direct source problem.  It is to
use this split as the base for one controlled non-exact setting:

1. boundary/domain deformation correction;
2. parameterized \(D\)-dependent split monitor;
3. fixed-\(D\) Ritz/eigen-PINN with the analytic \(\Omega_{\rm sing}\) supplied
   as a frozen geometry input.

## V0.3 Nonzero Harmonic Boundary/Domain Deformation

Implemented file:

- `train_harmonic_correction_pinn.py`

The controlled nonzero correction is

```math
\delta_*(s,z)=A\frac{2z^2-s}{L^2}.
```

It is exactly harmonic in the same axisymmetric \(s=\rho^2\) operator:

```math
\left(4s\partial_s^2+4\partial_s+\partial_z^2\right)(2z^2-s)
=0+4(-1)+4=0.
```

Thus

```math
\Omega=\Omega_{\rm sing}+\delta_*
```

still solves the same source equation, but with nonzero boundary/domain
deformation data carried entirely by the homogeneous correction.  This tests a
real correction channel rather than the V0.2 zero-correction sanity case.

### `v0p3_harmonic_deformation_800`

Command:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py \
  --steps 800 \
  --n-interior 1536 \
  --n-boundary 384 \
  --n-eval 4096 \
  --hidden 64 \
  --layers 4 \
  --log-every 200 \
  --w-pde 1 \
  --w-boundary 80 \
  --w-data 40 \
  --deformation-amp 0.02 \
  --make-plots \
  --plot-grid 90 \
  --run-name v0p3_harmonic_deformation_800
```

Result:

```text
exact_delta_laplace_residual_max_abs=0.0
uniform_delta_median_abs=6.9692730904e-04
uniform_omega_median_rel=5.5776315276e-04
uniform_laplace_residual_median_abs=1.9053893629e-04
trainlike_delta_median_abs=4.9689644948e-04
trainlike_omega_median_rel=3.9904759615e-04
trainlike_laplace_residual_core_abs_median_abs=1.2626886019e-04
trainlike_laplace_residual_bulk_abs_median_abs=1.6992968449e-04
```

Reading: the split PINN can learn a nonzero homogeneous correction without
reintroducing a Plummer-core residual explosion.  As a smoke test, this is
successful: the full geometry error is below \(10^{-3}\) in median relative
terms and the homogeneous residual median is \(O(10^{-4})\).

V0.3 is not yet a hard closure theorem:

- the correction-only median relative error is still about \(7--9\%\), partly
  because \(\delta_*\) crosses zero and relative error becomes ill-conditioned;
- this has only been run at \(D=12\);
- a harder gate should add either a harmonic-basis coefficient head or a
  \(D=6,12,18\) deformation sweep.

## Updated PINN Recommendation

The best next heartbeat step is V0.3.1:

1. add a harmonic-basis head \(c(2z^2-s)/L^2\) plus a residual MLP;
2. verify that the fitted coefficient \(c\) recovers \(A=0.02\);
3. repeat the deformation smoke for \(D=6,12,18\).

Only after that should the sandbox move to fixed-\(D\) Ritz/eigen-PINN.

## V0.3.1 Harmonic-Basis Head and D-Sweep

Updated file:

- `train_harmonic_correction_pinn.py`

New modes:

- `--basis-head`: adds an explicit trainable coefficient \(c\) multiplying
  \((2z^2-s)/L^2\), plus the residual MLP.
- `--basis-only`: uses only the coefficient \(c\), giving an identifiable
  coefficient-recovery gate.

### Mathematical Gate

In basis-only mode,

```math
\delta_\theta(s,z)=c\frac{2z^2-s}{L^2}.
```

The training target is

```math
\delta_*(s,z)=A\frac{2z^2-s}{L^2},\qquad A=0.02.
```

Because the basis is exactly harmonic,

```math
L_s\delta_\theta=0
```

for every value of \(c\).  The PDE loss is therefore identically zero, and the
boundary/data losses reduce to a one-parameter least-squares recovery of
\(c=A\).  This isolates the coefficient-identifiability question from neural
approximation error.

### `v0p3p1_basis_only_D12_200`

Command:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py \
  --basis-only \
  --steps 200 \
  --n-interior 1024 \
  --n-boundary 256 \
  --n-eval 2048 \
  --hidden 32 \
  --layers 2 \
  --log-every 50 \
  --w-pde 1 \
  --w-boundary 80 \
  --w-data 40 \
  --deformation-amp 0.02 \
  --run-name v0p3p1_basis_only_D12_200
```

Result:

```text
learned_basis_coeff=0.019999489188194275
basis_coeff_error=5.103647708892822e-07
uniform_delta_median_abs=1.3690441847e-07
uniform_omega_median_rel=9.9213472993e-08
uniform_laplace_residual_median_abs=0.0
trainlike_omega_median_rel=9.4980634913e-08
trainlike_laplace_residual_median_abs=0.0
```

Reading: the harmonic coefficient gate recovers \(A=0.02\) to about
\(5.1\times10^{-7}\), with exactly zero homogeneous residual.

### `v0p3p1_basis_sweep_D6_D12_D18_200`

Commands:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 6  --basis-only --steps 200 --n-interior 1024 --n-boundary 256 --n-eval 2048 --hidden 32 --layers 2 --log-every 100 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --run-name v0p3p1_basis_sweep_D6_200
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 12 --basis-only --steps 200 --n-interior 1024 --n-boundary 256 --n-eval 2048 --hidden 32 --layers 2 --log-every 100 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --run-name v0p3p1_basis_sweep_D12_200
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 18 --basis-only --steps 200 --n-interior 1024 --n-boundary 256 --n-eval 2048 --hidden 32 --layers 2 --log-every 100 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --run-name v0p3p1_basis_sweep_D18_200
```

Result:

```text
D=6   learned_basis_coeff=0.019999489188194275  basis_coeff_error=5.1036477089e-07  median_residual=0.0
D=12  learned_basis_coeff=0.019999489188194275  basis_coeff_error=5.1036477089e-07  median_residual=0.0
D=18  learned_basis_coeff=0.019999487325549126  basis_coeff_error=5.1222741604e-07  median_residual=0.0
```

Reading: the basis recovery is stable across representative two-center
separations.  This closes the V0.3.1 coefficient gate.

### `v0p3p1_basis_head_residual_D12_800`

Command:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py \
  --D 12 \
  --basis-head \
  --steps 800 \
  --n-interior 1536 \
  --n-boundary 384 \
  --n-eval 4096 \
  --hidden 64 \
  --layers 4 \
  --log-every 200 \
  --w-pde 1 \
  --w-boundary 80 \
  --w-data 40 \
  --deformation-amp 0.02 \
  --make-plots \
  --plot-grid 90 \
  --run-name v0p3p1_basis_head_residual_D12_800
```

Result:

```text
learned_basis_coeff=0.01974114216864109
basis_coeff_error=2.5885738432e-04
uniform_delta_median_abs=1.1626165360e-05
uniform_omega_median_rel=9.4701963462e-06
uniform_laplace_residual_median_abs=7.0618407335e-06
trainlike_delta_median_abs=1.1231750250e-05
trainlike_omega_median_rel=5.9974790929e-06
trainlike_laplace_residual_core_abs_median_abs=1.2171862181e-05
```

Reading: with a residual MLP, the field approximation improves by roughly two
orders of magnitude compared with the pure-MLP V0.3 smoke.  The coefficient is
less identifiable because the residual network can absorb a small component of
the same harmonic direction.  Therefore:

- use `--basis-only` for coefficient recovery;
- use `--basis-head` plus residual MLP for general approximation quality;
- if coefficient interpretability matters in the mixed model, add an
  orthogonality penalty against the harmonic basis.

## Updated PINN Recommendation After V0.3.1

V0.3.1 closes the controlled harmonic correction gate.  The next useful step is
one of:

1. V0.3.2: add residual-basis orthogonality penalty so mixed
   basis-head+MLP models recover both field accuracy and coefficient
   interpretability;
2. V0.4: move to fixed-\(D\) Ritz/eigen-PINN using frozen
   \(\Omega_{\rm sing}\), with deterministic finite-difference eigenvalues as
   holdout references.

## V0.3.2 Residual-Basis Orthogonality

Updated file:

- `train_harmonic_correction_pinn.py`

New option:

- `--w-orth`: penalizes the residual MLP projection onto the explicit harmonic
  basis.

The mixed model is

```math
\delta_\theta(s,z)
=c\phi(s,z)+r_\theta(s,z),
\qquad
\phi(s,z)=\frac{2z^2-s}{L^2}.
```

Without a constraint, \(r_\theta\) can absorb part of \(\phi\), making \(c\)
less identifiable.  V0.3.2 adds

```math
\mathcal L_{\rm orth}
=
\frac{\langle r_\theta,\phi\rangle^2}
{\langle\phi,\phi\rangle}.
```

This is intentionally narrow: it does not suppress all residual corrections,
only the piece parallel to the explicitly interpretable harmonic basis.

### `v0p3p2_basis_head_orth_D12_800`

Command:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py \
  --D 12 \
  --basis-head \
  --w-orth 10000 \
  --steps 800 \
  --n-interior 1536 \
  --n-boundary 384 \
  --n-eval 4096 \
  --hidden 64 \
  --layers 4 \
  --log-every 200 \
  --w-pde 1 \
  --w-boundary 80 \
  --w-data 40 \
  --deformation-amp 0.02 \
  --make-plots \
  --plot-grid 90 \
  --run-name v0p3p2_basis_head_orth_D12_800
```

Result:

```text
learned_basis_coeff=0.02000238746404648
basis_coeff_error=2.3879110813e-06
uniform_omega_median_rel=5.5373402574e-06
uniform_laplace_residual_median_abs=3.1882664189e-06
uniform_residual_basis_coeff=1.3194514850e-06
trainlike_omega_median_rel=4.0716881813e-06
trainlike_laplace_residual_core_abs_median_abs=2.2565072868e-06
trainlike_residual_basis_coeff=6.7942528403e-07
```

Compared with the no-orthogonality mixed model:

```text
no_orth_basis_coeff_error=2.5885738432e-04
orth_basis_coeff_error=2.3879110813e-06
```

The coefficient interpretability improves by about two orders of magnitude,
while the full-field median relative error remains \(O(10^{-6})\).

### `v0p3p2_basis_head_orth_D6_D12_D18_800`

Commands:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 6  --basis-head --w-orth 10000 --steps 800 --n-interior 1536 --n-boundary 384 --n-eval 4096 --hidden 64 --layers 4 --log-every 400 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --run-name v0p3p2_basis_head_orth_D6_800
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 12 --basis-head --w-orth 10000 --steps 800 --n-interior 1536 --n-boundary 384 --n-eval 4096 --hidden 64 --layers 4 --log-every 200 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --make-plots --plot-grid 90 --run-name v0p3p2_basis_head_orth_D12_800
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py --D 18 --basis-head --w-orth 10000 --steps 800 --n-interior 1536 --n-boundary 384 --n-eval 4096 --hidden 64 --layers 4 --log-every 400 --w-pde 1 --w-boundary 80 --w-data 40 --deformation-amp 0.02 --run-name v0p3p2_basis_head_orth_D18_800
```

Result summary:

```text
D=6   learned_basis_coeff=0.020006833598017693  basis_coeff_error=6.8340450525e-06  trainlike_omega_median_rel=7.3324731602e-06  trainlike_residual_median_abs=7.1575341281e-06
D=12  learned_basis_coeff=0.02000238746404648   basis_coeff_error=2.3879110813e-06  trainlike_omega_median_rel=4.0716881813e-06  trainlike_residual_median_abs=3.1738600228e-06
D=18  learned_basis_coeff=0.02000129595398903   basis_coeff_error=1.2964010239e-06  trainlike_omega_median_rel=4.0016834646e-06  trainlike_residual_median_abs=5.6416320149e-06
```

Reading: V0.3.2 closes the controlled harmonic boundary/domain deformation
line.  The analytic singular split is retained, the nonzero harmonic correction
is coefficient-recoverable, and the residual MLP can be used without destroying
coefficient interpretability.

## Updated PINN Recommendation After V0.3.2

The PINN geometry sanity stack is now strong enough for the first spectral
experiment.  The next step should be:

1. V0.4 fixed-\(D\) Ritz/eigen-PINN at \(D=12\), using frozen
   \(\Omega_{\rm sing}\);
2. compare the first few eigenvalues against deterministic finite-difference
   references;
3. keep Sturm/finite-difference certificates as the source of truth, with PINN
   used only as a differentiable surrogate candidate.

## V0.4 Fixed-D Spectral PINN: Active Alignment Blocker

Implemented files:

- `train_ritz_1d_single_track_pinn.py`
- `validate_ritz_2d_localized_basis.py`
- `audit_v0p4_reference_alignment.py`

### V0.4.0 Convention Correction

The canonical single-track artifact reports eigenvalues of the shifted
operator

```math
H_U=-\nabla^2+U,
\qquad
U=V_{\rm eff}-m_0^2.
```

The first V0.4 scaffold mistakenly compared the Rayleigh quotient of
\(-\partial_z^2+V_{\rm eff}\) to the checked-in \(E\) artifact.  This was
corrected in `train_ritz_1d_single_track_pinn.py` so the script now optimizes
the shifted axial operator

```math
H_U^{1D}=-\partial_z^2+U(0,z;D).
```

### `v0p4_ritz_1d_shiftedU_D12_1200`

Command:

```bash
pinn/.venv/bin/python pinn/train_ritz_1d_single_track_pinn.py \
  --D 12 \
  --z-max 14 \
  --steps 1200 \
  --n-quad 2048 \
  --hidden 64 \
  --layers 4 \
  --log-every 200 \
  --make-plots \
  --plot-grid 900 \
  --run-name v0p4_ritz_1d_shiftedU_D12_1200
```

Result:

```text
E_pinn=0.9620608687400818
omega_pinn=1.400735831175915
E_ref=-0.9205263084737675
omega_ref=0.2819107864666276
E_abs_error_ref=1.8825871772138494
residual_l2_over_norm=0.15691807866096497
```

Reading: a plain axial MLP Ritz ansatz gets trapped in a continuum-like local
minimum and does not see the checked-in bound-sector reference.  This is an
expected spectral-PINN failure mode and is not a closure.

### `v0p4_ritz_2d_localized_D12`

Command:

```bash
pinn/.venv/bin/python pinn/validate_ritz_2d_localized_basis.py \
  --D 12 \
  --rho-max 4 \
  --z-max 20 \
  --n-s 120 \
  --n-z 600 \
  --run-name v0p4_ritz_2d_localized_D12
```

Result:

```text
E0_ritz=3.3401518487899224
omega0_ritz=2.0833031101570225
E_ref=-0.9205263084737675
E_abs_error_ref=4.26067815726369
basis_size=256
retained_basis_rank=236
```

Reading: even a localized Gaussian variational basis in the continuous
axisymmetric energy functional does not reproduce the checked-in artifact.
This suggests the next issue is not merely neural architecture; it is reference
operator alignment.

### `v0p4_reference_alignment_D12`

Command:

```bash
python3 pinn/audit_v0p4_reference_alignment.py \
  --D 12 \
  --run-name v0p4_reference_alignment_D12
```

Result:

```text
matrix_shape=(25000,25000)
matrix_nnz=123900
fro_norm=109237.40124998573
asym_fro=7998.333855397619
asym_rel=0.07321973759787391
eigsh_values_current_code=[-0.2840644552,-0.2098108375,-0.0333351515,0.1606768671,0.2231351386,0.5004381590]
artifact_E=-0.9205263084737675
current_first_negative=-0.2840644552283058
current_vs_artifact_E_abs_delta=0.6364618532454617
```

Reading: the current `code/true_single_track_solver.py` finite-difference
matrix is not symmetric in the ordinary matrix inner product used by `eigsh`,
and the current-code D12 output no longer reproduces the checked-in
`output/true_single_track/true_results.json` artifact.  This must be treated as
a V0.4 alignment blocker before any spectral PINN can use that artifact as a
holdout reference.

## Updated PINN Recommendation After V0.4 Alignment Audit

Do not keep training spectral PINNs against the current single-track artifact
yet.  The next heartbeat should do V0.4.1 reference-operator repair:

1. derive the correct self-adjoint weighted cylindrical discretization for
   \(H_U=-\nabla^2+U\);
2. compare its D12 low eigenvalues against both the legacy artifact and the
   current `true_single_track_solver.py` output;
3. decide whether PINN V0.4 should target the self-adjoint operator or only a
   legacy finite-difference proxy.

## V0.4.1 Self-Adjoint Weighted Cylindrical Reference

Implemented file:

- `audit_v0p4_self_adjoint_reference.py`

The self-adjoint continuum operator is

```math
H_U
=-\rho^{-1}\partial_\rho(\rho\partial_\rho)
-\partial_z^2+U,
\qquad
U=V_{\rm eff}-m_0^2,
```

with inner product

```math
\langle u,v\rangle_\rho
=\int \rho\,u v\,d\rho dz.
```

V0.4.1 discretizes the energy form as a generalized eigenproblem

```math
K u=E M u,
\qquad
M_{ij}\simeq \rho_i\Delta\rho\Delta z.
```

The stiffness \(K\) comes from

```math
\int \rho\left(|\partial_\rho u|^2+|\partial_z u|^2+U|u|^2\right)d\rho dz.
```

This removes the Euclidean nonsymmetry seen in the legacy finite-difference
matrix.

### `v0p4p1_selfadjoint_D12_n50x500`

Command:

```bash
python3 pinn/audit_v0p4_self_adjoint_reference.py \
  --D 12 \
  --n-rho 50 \
  --n-z 500 \
  --L-rho 4 \
  --L-z 20 \
  --run-name v0p4p1_selfadjoint_D12_n50x500
```

Result:

```text
K_asym_rel=0.0
E_selfadjoint=[0.7898376934,0.7898376948,1.0496834872,1.0496840837,1.2524902686,1.3670092242,...]
n_negative_selfadjoint=0
artifact_E=-0.9205263085
legacy_current_solver_E=[-0.1300114816,-0.1234545616,-0.0057197929,0.0939665333,...]
selfadjoint_first_vs_artifact_E_abs_delta=1.7103640019
```

### `v0p4p1_selfadjoint_D12_grid_sweep`

Commands:

```bash
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 12 --n-rho 30 --n-z 300 --L-rho 4 --L-z 20 --k 6 --run-name v0p4p1_selfadjoint_D12_n30x300
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 12 --n-rho 40 --n-z 400 --L-rho 4 --L-z 20 --k 6 --run-name v0p4p1_selfadjoint_D12_n40x400
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 12 --n-rho 50 --n-z 500 --L-rho 4 --L-z 20 --k 8 --run-name v0p4p1_selfadjoint_D12_n50x500
```

Result:

```text
30x300: E0_selfadjoint=0.7897461778  K_asym_rel=0.0  n_negative=0  legacy_first_E=-0.1937008402
40x400: E0_selfadjoint=0.7898087368  K_asym_rel=0.0  n_negative=0  legacy_first_E=-0.3546595156
50x500: E0_selfadjoint=0.7898376934  K_asym_rel=0.0  n_negative=0  legacy_first_E=-0.1300114816
```

Reading:

- the self-adjoint discretization is exactly symmetric in the generalized
  weighted formulation and appears grid-stable on this D12 sweep;
- it has no negative bound eigenvalue at D12;
- the legacy nonsymmetric solver gives grid-dependent negative values;
- the checked-in artifact is not reproduced by either the current legacy
  solver or the self-adjoint operator.

Therefore the V0.4 spectral PINN should not use
`output/true_single_track/true_results.json` as a hard holdout reference.  For
future PINN work, the clean target is the self-adjoint weighted cylindrical
operator.  The legacy artifact can only be used as a diagnostic/proxy unless a
separate legacy-reproduction route is explicitly requested.

## Updated PINN Recommendation After V0.4.1

The next heartbeat should either:

1. train a fixed-\(D=12\) spectral PINN against the self-adjoint weighted
   operator and target \(E_0\simeq0.78984\), clearly marked as a self-adjoint
   reference experiment; or
2. stop the spectral PINN line and commit the PINN sandbox package, because the
   geometry split and harmonic correction tests are closed while the legacy
   single-track artifact is not a clean self-adjoint holdout.

## V0.5 Weighted Self-Adjoint Ritz-PINN

Implemented file:

- `train_ritz_2d_selfadjoint_pinn.py`

V0.5 follows option 1 above.  It trains a Ritz-PINN against the repaired
self-adjoint weighted cylindrical operator rather than against the legacy
single-track artifact.  In \(s=\rho^2\) variables,

```math
H_Uu=-4s\,u_{ss}-4u_s-u_{zz}+U(s,z)u,
\qquad
U=V_{\rm eff}-m_0^2.
```

The variational target is

```math
E[u]=
\frac{\int (4s|u_s|^2+|u_z|^2+U|u|^2)\,ds\,dz}
     {\int |u|^2\,ds\,dz}.
```

The hard Dirichlet envelope imposes the finite-box boundary at \(s=s_{\max}\)
and \(z=\pm L_z\), while the \(s=0\) axis is left as the natural cylindrical
boundary.  This matches the weighted finite-volume reference used in V0.4.1.

### `v0p5_selfadjoint_smoke_200`

Command:

```bash
pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py \
  --D 12 \
  --n-s 36 \
  --n-z 96 \
  --steps 200 \
  --hidden 48 \
  --layers 3 \
  --log-every 50 \
  --n-residual 1024 \
  --run-name v0p5_selfadjoint_smoke_200
```

Result:

```text
E_pinn=0.7909942865
E0_selfadjoint_reference=0.7898376934
E_rel_error_selfadjoint_reference=1.4643428002e-03
strong_residual_l2_over_rms_u=7.1398891509e-02
```

### `v0p5_selfadjoint_D12_800`

Command:

```bash
pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py \
  --D 12 \
  --n-s 48 \
  --n-z 128 \
  --steps 800 \
  --hidden 64 \
  --layers 4 \
  --log-every 200 \
  --n-residual 2048 \
  --make-plots \
  --run-name v0p5_selfadjoint_D12_800
```

Result:

```text
E_pinn=0.7900940180
omega_pinn=1.3379439517
E0_selfadjoint_reference=0.7898376934
omega0_selfadjoint_reference=1.3378481578
E_abs_error_selfadjoint_reference=2.5632458493e-04
E_rel_error_selfadjoint_reference=3.2452817468e-04
strong_residual_l2_over_rms_u=2.8718549758e-02
strong_residual_median_abs_over_rms_u=6.5732561052e-03
```

Reading:

- V0.5 gives a positive spectral-PINN gate against the self-adjoint reference;
- the fixed-\(D=12\) Ritz-PINN reaches the finite-volume reference at
  \(3.3\times10^{-4}\) relative energy;
- the strong-form residual is not yet certificate-grade, but it is small enough
  for a sandbox PINN alignment test;
- the result does not rescue the legacy negative-energy artifact, which remains
  diagnostic-only unless a separate legacy-emulation route is explicitly opened.

## Updated PINN Recommendation After V0.5

The next useful step is one of:

1. run a V0.5.1 self-adjoint \(D\)-sweep by first generating weighted
   finite-volume references at \(D=6,12,18\), then training the same Ritz-PINN
   against those references; or
2. freeze V0 as a successful geometry-plus-self-adjoint-spectral sandbox and
   commit the package before opening any parametric-\(D\) PINN.

## V0.5.1 Self-Adjoint Finite-Volume Reference and PINN D-Sweep

Tracked summary:

- `v0p5p1_summary.csv`

V0.5.1 follows the first V0.5 recommendation.  It first generates weighted
self-adjoint finite-volume references at \(D=6,12,18\), then trains the same
weighted Ritz-PINN architecture against those references.

The reference operator is still

```math
K u=E M u,
\qquad
M_{ij}\simeq \rho_i\Delta\rho\Delta z,
```

with stiffness from

```math
\int\rho\left(|\partial_\rho u|^2+|\partial_z u|^2+U|u|^2\right)d\rho dz.
```

The PINN uses the equivalent \(s=\rho^2\) Rayleigh quotient

```math
E[u]=
\frac{\int(4s|u_s|^2+|u_z|^2+U|u|^2)\,ds\,dz}
     {\int |u|^2\,ds\,dz}.
```

### Finite-volume references

Commands:

```bash
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 6  --n-rho 50 --n-z 500 --L-rho 4 --L-z 20 --k 8 --run-name v0p5p1_selfadjoint_D6_n50x500
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 12 --n-rho 50 --n-z 500 --L-rho 4 --L-z 20 --k 8 --run-name v0p5p1_selfadjoint_D12_n50x500
python3 pinn/audit_v0p4_self_adjoint_reference.py --D 18 --n-rho 50 --n-z 500 --L-rho 4 --L-z 20 --k 8 --run-name v0p5p1_selfadjoint_D18_n50x500
```

Results:

```text
D=6:  E0_selfadjoint=0.7409156024, K_asym_rel=0.0, n_negative=0
D=12: E0_selfadjoint=0.7898376934, K_asym_rel=0.0, n_negative=0
D=18: E0_selfadjoint=0.8907781421, K_asym_rel=0.0, n_negative=0
```

### Ritz-PINN D-sweep

The final reported sweep uses the same architecture and a 3000-step protocol:

```bash
pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py \
  --D <D> \
  --n-s 48 \
  --n-z 128 \
  --steps 3000 \
  --hidden 64 \
  --layers 4 \
  --log-every 500 \
  --n-residual 2048 \
  --run-name v0p5p1_pinn_D<D>_3000
```

Results:

```text
D   E_ref         E_PINN        abs_error      rel_error      strong_L2
6   0.7409156024  0.7410645485  1.4895e-04    2.0103e-04    1.7344e-02
12  0.7898376934  0.7899821997  1.4451e-04    1.8296e-04    1.8361e-02
18  0.8907781421  0.8908920884  1.1395e-04    1.2792e-04    2.0674e-02
```

Reading:

- the self-adjoint reference sweep is exactly symmetric in the weighted
  generalized formulation on all three points;
- the Ritz-PINN D-sweep reaches all three references at \(1.3\)--\(2.1\)
  \(\times10^{-4}\) relative energy;
- the strong-form residual remains a PINN diagnostic, not a proof certificate;
- D18 required longer training because the 800-step broad ansatz initially
  lingered near \(E\simeq0.927\), close to the third self-adjoint eigenvalue,
  before descending to the \(E_0\simeq0.89078\) branch.

This closes V0.5.1 as a positive fixed-point D-sweep.  It does not change the
main paper baseline and still does not promote the legacy negative-energy
single-track artifact.

## Updated PINN Recommendation After V0.5.1

The next useful step is either:

1. commit/freeze the V0 sandbox package, because the geometry split,
   harmonic-correction, self-adjoint reference, and fixed-point spectral PINN
   gates are now all positive; or
2. open V1 as a parametric-\(D\) PINN, but with branch-tracking safeguards:
   warm-start continuation in \(D\), explicit reference checkpoints, and a
   rule that PINN branch changes must be verified by the self-adjoint
   finite-volume solver.

The detailed V1 branch-tracking plan is now separated into `V1_PLAN.md`.
