# PSLT PINN Sandbox

This directory is intentionally separate from the manuscript.  It is a sandbox
for Physics-Informed Neural Network (PINN) experiments on the PSLT geometry
chain, starting with the most controlled target: the two-center Poisson problem
for the conformal factor.

## First Target

The initial PINN target is the two-center conformal factor

```math
\Omega(\rho,z;D)=1+a\left(r_+^{-1}+r_-^{-1}\right),
\qquad
r_\pm^2=\rho^2+(z\mp D/2)^2+\epsilon^2.
```

It solves the projected Poisson equation

```math
\nabla^2\Omega=-4\pi a\left[\rho_\epsilon(x-x_+)+\rho_\epsilon(x-x_-)\right],
\qquad
\rho_\epsilon(r)=\frac{3\epsilon^2}{4\pi(r^2+\epsilon^2)^{5/2}}.
```

In axisymmetric coordinates, the PINN residual is

```math
\mathcal R_\theta(\rho,z)=
\partial_\rho^2\Omega_\theta+
\rho^{-1}\partial_\rho\Omega_\theta+
\partial_z^2\Omega_\theta
+4\pi\sigma(\rho,z).
```

The loss combines:

```math
\mathcal L=
w_{\rm pde}\langle |\mathcal R_\theta|^2\rangle_{\rm int}
+w_{\rm bdry}\langle |\Omega_\theta-\Omega_{\rm exact}|^2\rangle_{\partial\Omega}
+w_{\rm axis}\langle |\partial_\rho\Omega_\theta(0,z)|^2\rangle.
```

This tests whether a PINN can learn the parent geometry before touching the
harder spectral/eigenvalue problem.

## Files

- `two_center_geometry.py`: analytic two-center geometry utilities.
- `validate_poisson_identity.py`: pure NumPy finite-difference check of the
  Plummer source identity.
- `train_two_center_poisson_pinn.py`: PyTorch PINN training entrypoint.
- `train_two_center_poisson_split_pinn.py`: V0.2 analytic-singular-split
  PINN entrypoint, where the network learns only the homogeneous correction.
- `train_harmonic_correction_pinn.py`: V0.3 nonzero harmonic boundary/domain
  deformation test on top of the analytic singular split.
- `accelerator_utils.py`: conservative Torch accelerator selection helpers.
- `check_torch_accelerator.py`: reports CUDA/MPS availability and optional
  Torch matmul benchmark.
- `validate_singular_split.py`: autograd validation of
  \(L_s\Omega_{\rm sing}+4\pi\sigma=0\) across selected \(D\) values.
- `V1_PLAN.md`: branch-tracked parametric-\(D\) plan after V0.5.1.
- `v1_reference_summary.csv`: self-adjoint checkpoint references for
  \(D=6,9,12,15,18\).
- `v1_anchor_summary.csv`: fixed-\(D\) Ritz-PINN anchor results for the same
  checkpoint set.
- `train_v1_parametric_d_ritz_pinn.py`: first branch-tracked parametric-\(D\)
  lowest-branch Ritz-PINN scaffold.
- `v1_parametric_first_run_summary.csv`: first formal cold-start parametric
  run summary; diagnostic, not gate-passing.
- `v1_parametric_continuation_summary.csv`: gate-passing continuation
  parametric-\(D\) checkpoint summary.
- `evaluate_v1_parametric_d.py`: dense-\(D\) diagnostic evaluator for trained
  V1 parametric models.
- `v1_dense_eval_dstep1p5_summary.csv`: dense intermediate-\(D\) diagnostic
  summary for the gate-passing continuation model.
- `requirements.txt`: optional PINN dependencies.

## Quick Checks

Run the analytic source-identity check:

```bash
python3 pinn/validate_poisson_identity.py
```

Install PINN dependencies when ready:

```bash
python3 -m pip install -r pinn/requirements.txt
```

Then run a small training job:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_pinn.py --steps 2000 --n-interior 2048
```

Outputs go under `pinn/runs/`, which is ignored by git.

## Accelerator Check

On Apple Silicon, PyTorch may expose the Metal Performance Shaders backend as
`mps`.  The sandbox uses conservative device selection: `--device auto` uses
CUDA if available, then MPS only if a real tensor probe succeeds, otherwise CPU.
This avoids silently thinking a PINN is GPU-accelerated when the local runtime
cannot execute MPS kernels.

Check the current runtime:

```bash
pinn/.venv/bin/python pinn/check_torch_accelerator.py --device auto
```

Optional small benchmark:

```bash
pinn/.venv/bin/python pinn/check_torch_accelerator.py --device auto --benchmark --n 512 --repeat 10
```

The self-adjoint Ritz-PINN accepts the same selector:

```bash
pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py --device auto
```

Finite-volume references based on `scipy.sparse.linalg.eigsh` remain CPU-side;
the accelerator path applies to PyTorch PINN training.

## V0.2 Analytic Singular Split

The direct V0/V0.1 PINN asks a neural network to learn both the smooth field
and the sharp Plummer source cores.  V0.2 removes that avoidable burden by
splitting

```math
\Omega_\theta(s,z)=\Omega_{\rm sing}(s,z)+\delta_\theta(s,z),
```

where

```math
\Omega_{\rm sing}=1+a\left[
\bigl(s+(z-D/2)^2+\epsilon^2\bigr)^{-1/2}
+
\bigl(s+(z+D/2)^2+\epsilon^2\bigr)^{-1/2}
\right].
```

Since \(\Omega_{\rm sing}\) is the exact Plummer Green-function solution of the
V0 Poisson problem, the trainable correction satisfies

```math
4s\,\partial_s^2\delta_\theta
+4\,\partial_s\delta_\theta
+\partial_z^2\delta_\theta=0,
\qquad
\delta_\theta|_{\partial\mathcal D}=0.
```

For the exact two-center sanity test, the unique correction is
\(\delta_\theta\equiv0\).  This is intentional: the split checks whether the
PINN pipeline invents neural artifacts after the singular source has been
handled analytically.

Exact-zero split smoke:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_split_pinn.py \
  --steps 1 \
  --n-interior 512 \
  --n-boundary 128 \
  --n-eval 1024 \
  --hidden 32 \
  --layers 3 \
  --make-plots \
  --plot-grid 64 \
  --run-name v0p2_split_exact_zero_smoke
```

Random-final stress smoke:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_split_pinn.py \
  --random-final \
  --steps 300 \
  --n-interior 1024 \
  --n-boundary 256 \
  --n-eval 2048 \
  --hidden 64 \
  --layers 4 \
  --w-pde 1 \
  --w-boundary 20 \
  --w-data 10 \
  --make-plots \
  --plot-grid 72 \
  --run-name v0p2_split_random_stress_300
```

## Current V0 Reading

The direct V0.1 implementation is a smoke-tested sandbox, not a closed solver.
The best short direct run is a data-assisted warm start:

```bash
pinn/.venv/bin/python pinn/train_two_center_poisson_pinn.py \
  --steps 1000 \
  --n-interior 1024 \
  --n-boundary 256 \
  --n-data 2048 \
  --n-eval 4096 \
  --hidden 64 \
  --layers 4 \
  --w-pde 0.0001 \
  --w-boundary 2 \
  --w-axis 0.05 \
  --w-data 80 \
  --source-sample-frac 0.15 \
  --run-name v0_warmstart_1k
```

It gives a median relative Omega error near one percent, while the maximum
error and PDE residual remain controlled by the sharp Plummer cores.  Therefore
direct V0.1 is a working training/evaluation pipeline and a diagnostic for
adaptive sampling, not a production replacement for the deterministic
Poisson/finite-difference chain.

V0.2 changes the interpretation: for the exact two-center Poisson geometry, the
analytic split closes the source-core problem at the parent level.  The PINN is
then only a homogeneous-correction monitor.  This makes V0.2 a clean sanity
test and the preferred starting point for future non-exact backgrounds,
boundary deformations, or domain-mapping experiments.

The split identity can also be checked directly across \(D\):

```bash
pinn/.venv/bin/python pinn/validate_singular_split.py \
  --D 6 12 18 \
  --n-uniform 4096 \
  --n-core 4096 \
  --json-out pinn/runs/v0p2_singular_split_d_sweep.json
```

This validates the same autograd derivative structure used by the split PINN.

## V0.3 Nonzero Harmonic Boundary Deformation

V0.3 checks that the split PINN can learn a nontrivial homogeneous correction,
not only the zero solution.  The target is the harmonic polynomial

```math
\delta_*(s,z)=A\frac{2z^2-s}{L^2},
```

because

```math
\left(4s\partial_s^2+4\partial_s+\partial_z^2\right)(2z^2-s)
=-4+4=0.
```

Smoke command:

```bash
pinn/.venv/bin/python pinn/train_harmonic_correction_pinn.py \
  --steps 800 \
  --n-interior 1536 \
  --n-boundary 384 \
  --n-eval 4096 \
  --hidden 64 \
  --layers 4 \
  --w-pde 1 \
  --w-boundary 80 \
  --w-data 40 \
  --deformation-amp 0.02 \
  --make-plots \
  --plot-grid 90 \
  --run-name v0p3_harmonic_deformation_800
```

The current smoke reaches sub-\(10^{-3}\) median relative error in the full
\(\Omega_{\rm sing}+\delta_\theta\) field and homogeneous residual medians of
order \(10^{-4}\).  This is enough to validate the mechanism as a smoke test;
a harder V0.3 gate should add either a harmonic-basis head or a \(D\)-sweep.

## V0.3.1 Harmonic-Basis Head

The harder coefficient-recovery gate adds an explicit harmonic basis head

```math
\delta_\theta(s,z)=c\frac{2z^2-s}{L^2}
```

and asks whether training recovers \(c=A=0.02\).  In `--basis-only` mode the
coefficient is identifiable, and the \(D=6,12,18\) sweep recovers

```text
D=6   c=0.0199994892   |c-A|=5.10e-7
D=12  c=0.0199994892   |c-A|=5.10e-7
D=18  c=0.0199994873   |c-A|=5.12e-7
```

The mixed `--basis-head` plus residual-MLP mode gives a better field fit but a
less identifiable coefficient, because the residual network can absorb a small
component of the same harmonic direction.  The current D=12 mixed run gives
`c=0.0197411422`, with full-geometry median relative error around `1e-5`.

## V0.3.2 Residual-Basis Orthogonality

To make the mixed basis-head plus residual-MLP model coefficient-interpretable,
V0.3.2 adds

```math
\mathcal L_{\rm orth}
=
\frac{\langle r_\theta,\phi\rangle^2}
{\langle\phi,\phi\rangle},
\qquad
\phi(s,z)=\frac{2z^2-s}{L^2},
```

where \(r_\theta\) is the residual MLP component.  With `--w-orth 10000`, the
mixed model keeps the good field approximation while forcing the learned basis
coefficient to carry the harmonic component:

```text
D=6   c=0.0200068336   |c-A|=6.83e-6   trainlike_omega_median_rel=7.33e-6
D=12  c=0.0200023875   |c-A|=2.39e-6   trainlike_omega_median_rel=4.07e-6
D=18  c=0.0200012960   |c-A|=1.30e-6   trainlike_omega_median_rel=4.00e-6
```

The corresponding trainlike homogeneous residual medians are
`7.16e-6`, `3.17e-6`, and `5.64e-6`.  This closes the controlled harmonic
deformation line strongly enough to move to the first fixed-\(D\) Ritz/eigen
PINN experiment.

## V0.4 Spectral PINN Alignment Check

The first spectral attempt intentionally started with the simplest possible
surrogate: a fixed-\(D=12\) axial Ritz PINN using the shifted potential

```math
H_U=-\partial_z^2+U(0,z;D),
\qquad
U=V_{\rm eff}-m_0^2.
```

That plain MLP trial converged to a continuum-like local minimum, not the
checked-in bound reference:

```text
E_pinn=0.9620608687
E_ref=-0.9205263085
```

A localized 2D variational Gaussian basis also failed to reproduce the
checked-in reference.  The follow-up alignment audit found that the current
`code/true_single_track_solver.py` matrix is not symmetric in the ordinary
Euclidean matrix inner product used by `eigsh`, and that current-code output at
\(D=12\) differs from the checked-in artifact:

```text
matrix_asym_rel=7.3219737598e-02
current_first_negative=-0.2840644552
artifact_E=-0.9205263085
|delta_E|=0.6364618532
```

Therefore V0.4 should pause on neural training and first align the reference
operator: either construct the self-adjoint weighted cylindrical operator, or
explicitly reproduce the legacy finite-difference proxy before using it as a
PINN holdout.

## V0.4.1 Self-Adjoint Cylindrical Reference

V0.4.1 builds the weighted finite-volume generalized eigenproblem

```math
K u = E M u,
\qquad
M_{ij}\simeq \rho_i\,\Delta\rho\,\Delta z,
```

from the self-adjoint form of

```math
H_U=-\rho^{-1}\partial_\rho(\rho\partial_\rho)
-\partial_z^2+U.
```

The energy discretization gives a symmetric \(K\) in the weighted cylindrical
inner product.  On \(D=12\), the grid sweep gives:

```text
n_rho x n_z   E0_selfadjoint   K_asym_rel   n_negative
30 x 300      0.7897461778     0.0          0
40 x 400      0.7898087368     0.0          0
50 x 500      0.7898376934     0.0          0
```

The checked-in single-track artifact has `E=-0.9205263085`, while the current
legacy nonsymmetric solver gives grid-dependent negative values.  The
self-adjoint reference is stable and nonnegative on this D12 test, so future
PINN spectral work should target the self-adjoint operator unless the goal is
explicitly to emulate a legacy proxy.

## V0.5 Weighted Self-Adjoint Ritz-PINN

V0.5 is the first spectral PINN that targets the repaired self-adjoint
cylindrical operator rather than the legacy single-track artifact.  With
\(s=\rho^2\), the shifted operator is

```math
H_U u=-4s\,u_{ss}-4u_s-u_{zz}+U(s,z)u,
\qquad
U=V_{\rm eff}-m_0^2.
```

The Ritz functional uses the cylindrical measure
\(\rho\,d\rho\,dz=\tfrac12 ds\,dz\), so the common factor cancels:

```math
E[u]=
\frac{\int(4s|u_s|^2+|u_z|^2+U|u|^2)\,ds\,dz}
     {\int |u|^2\,ds\,dz}.
```

The script is:

- `train_ritz_2d_selfadjoint_pinn.py`

Current D12 gate:

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
E0_selfadjoint_reference=0.7898376934
E_rel_error_selfadjoint_reference=3.2452817468e-04
strong_residual_l2_over_rms_u=2.8718549758e-02
strong_residual_median_abs_over_rms_u=6.5732561052e-03
```

Reading: the fixed-\(D=12\) self-adjoint spectral PINN is now aligned with the
repaired weighted reference at the \(3.3\times10^{-4}\) relative-energy level.
This closes the first positive spectral-PINN gate.  It does not revive the
legacy negative-energy artifact; that artifact remains diagnostic-only unless
a separate legacy-emulation route is requested.

## V0.5.1 Self-Adjoint D-Sweep

V0.5.1 first generates deterministic finite-volume references at
\(D=6,12,18\) using the same self-adjoint generalized eigenproblem, then trains
the same weighted Ritz-PINN architecture against those references.

Reference command template:

```bash
python3 pinn/audit_v0p4_self_adjoint_reference.py \
  --D <D> \
  --n-rho 50 \
  --n-z 500 \
  --L-rho 4 \
  --L-z 20 \
  --k 8 \
  --run-name v0p5p1_selfadjoint_D<D>_n50x500
```

PINN command template:

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

Summary:

```text
D   E_ref         E_PINN        rel_error      strong_residual_L2
6   0.7409156024  0.7410645485  2.0103e-04    1.7344e-02
12  0.7898376934  0.7899821997  1.8296e-04    1.8361e-02
18  0.8907781421  0.8908920884  1.2792e-04    2.0674e-02
```

All three finite-volume references have `K_asym_rel=0.0` and
`n_negative_selfadjoint=0`.  The tracked summary is
`v0p5p1_summary.csv`.

Important D18 note: the 800-step broad ansatz initially paused near
`E≈0.927`, close to the third self-adjoint eigenvalue.  Extending the same
architecture to 3000 steps reached the lowest self-adjoint reference at
`1.28e-4` relative energy.  This is a useful warning for future parametric
\(D\) PINNs: large-\(D\) points need either longer training, continuation, or
an explicit eigen-branch tracking gate.
