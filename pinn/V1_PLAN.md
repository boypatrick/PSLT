# V1 Parametric-D Self-Adjoint Ritz-PINN Plan

V0 closed the fixed-geometry and fixed-point spectral sanity checks.  V1 should
not train a broad parametric network blindly.  The D18 V0.5.1 run showed that a
short run can stick to a higher eigen-branch before descending to the ground
branch.  Therefore V1 is a branch-tracked continuation problem.

## Mathematical Target

For \(D\in[D_{\min},D_{\max}]\), use

```math
s=\rho^2,\qquad
H_U(D)u=-4s\,u_{ss}-4u_s-u_{zz}+U(s,z;D)u,
```

where

```math
U(s,z;D)=V_{\rm eff}(\sqrt{s},z;D)-m_0^2.
```

The variational energy is

```math
E_D[u]=
\frac{\int(4s|u_s|^2+|u_z|^2+U(s,z;D)|u|^2)\,ds\,dz}
     {\int |u|^2\,ds\,dz}.
```

The parametric PINN represents

```math
u_\theta=u_\theta(s,z,D),
\qquad
E_\theta(D)=E_D[u_\theta(\cdot,\cdot,D)].
```

The first V1 target is only the lowest self-adjoint branch \(E_0(D)\), not a
multi-mode eigen-PINN.

## Why Continuation Is Required

V0.5.1 found:

```text
D=18, 800 steps:  E_PINN≈0.927
D=18, 3000 steps: E_PINN≈0.890892
reference:        E0≈0.890778
```

The 800-step value is close to a higher self-adjoint eigenvalue.  Thus a
parametric network can look numerically stable while riding the wrong branch.
V1 must include explicit branch checks.

## V1.0 Minimal Continuation Protocol

1. Generate finite-volume references on a sparse checkpoint set:

```text
D = 6, 9, 12, 15, 18
```

using the V0.4.1 self-adjoint generalized eigenproblem.

2. Train independent fixed-D PINNs at the checkpoints using the V0.5.1
   3000-step protocol.  These are the anchor states.

3. Train a parametric network \(u_\theta(s,z,D)\) with checkpoint loss:

```math
\mathcal L_{\rm Ritz}
=\frac1{|\mathcal D_{\rm train}|}\sum_{D_j} E_{D_j}[u_\theta(\cdot,D_j)].
```

4. Add branch-coherence loss against anchor energies:

```math
\mathcal L_{\rm chk}
=\sum_{D_j\in\mathcal D_{\rm chk}}
\left(
\frac{E_\theta(D_j)-E^{\rm FV}_0(D_j)}
     {\max(1,|E^{\rm FV}_0(D_j)|)}
\right)^2.
```

This does not replace the variational target; it prevents branch flips at
known checkpoints.

5. Add smoothness only as a weak regularizer:

```math
\mathcal L_D
=\lambda_D\int |\partial_D E_\theta(D)|^2\,dD.
```

Do not over-weight this term, because real avoided crossings or near-degenerate
doublets may create sharp local structure.

## Acceptance Gates

A V1 result is acceptable only if all of the following hold.

```text
max_D_checkpoint rel_error(E_PINN, E_FV) < 5e-4
median_D_checkpoint rel_error(E_PINN, E_FV) < 3e-4
max_D_checkpoint strong_residual_l2_over_rms_u < 5e-2
no checkpoint branch flip relative to FV eigenvalue ordering
```

For a dense \(D\)-grid, V1 may propose branch boundaries, but every suspicious
cell must be verified by the finite-volume self-adjoint solver before being
called physical.

## Deliverables

- `audit_v1_selfadjoint_reference_sweep.py` or a thin wrapper around
  `audit_v0p4_self_adjoint_reference.py`.
- `train_v1_parametric_d_ritz_pinn.py`.
- `v1_reference_summary.csv`.
- `v1_parametric_summary.csv`.
- One diagnostic plot under ignored `pinn/runs/`, not in the paper.

## Explicit Non-Goals

- Do not use `output/true_single_track/true_results.json` as a target.
- Do not claim a bound-state theorem from PINN convergence.
- Do not modify the manuscript.
- Do not open multi-mode orthogonality until the lowest branch is stable.

## Current Recommendation

Commit/freeze V0 before implementing V1.  If V1 is opened before a commit, keep
the first implementation narrow: generate \(D=9,15\) references and train only
checkpoint fixed-D anchors before writing the parametric network.
