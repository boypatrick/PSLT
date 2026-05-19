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

## V1.0 Reference Checkpoint Status

V0 has been frozen in commit `bcc2275` (`Freeze PINN V0 sandbox`).  The first
V1 reference step then generated the missing checkpoint references at
\(D=9,15\).  The full checkpoint table is now tracked in
`v1_reference_summary.csv`.

The \(D=9,15\) runs use the same self-adjoint finite-volume operator but skip
the legacy nonsymmetric comparison:

```bash
python3 pinn/audit_v0p4_self_adjoint_reference.py \
  --D 9 \
  --n-rho 50 \
  --n-z 500 \
  --L-rho 4 \
  --L-z 20 \
  --k 4 \
  --skip-legacy \
  --run-name v1_selfadjoint_D9_n50x500_k4

python3 pinn/audit_v0p4_self_adjoint_reference.py \
  --D 15 \
  --n-rho 50 \
  --n-z 500 \
  --L-rho 4 \
  --L-z 20 \
  --k 4 \
  --skip-legacy \
  --run-name v1_selfadjoint_D15_n50x500_k4
```

The narrower \(k=4\) extraction is enough for V1 because the target is the
lowest branch.  It avoids the slow high-mode/legacy diagnostic work that is not
part of V1.

Results:

```text
D   E0_selfadjoint   omega0          K_asym_rel   n_negative
6   0.7409156024     1.3194376084    0.0          0
9   0.7604497941     1.3268194278    0.0          0
12  0.7898376934     1.3378481578    0.0          0
15  0.8317674017     1.3534280187    0.0          0
18  0.8907781421     1.3750556869    0.0          0
```

Next V1 step: train fixed-\(D\) anchors at \(D=9,15\) with the V0.5.1
3000-step protocol before attempting a parametric-\(D\) network.

## V1.0 Fixed-D Anchor Status

Tracked summary:

- `v1_anchor_summary.csv`

The missing fixed-\(D\) anchors at \(D=9,15\) were trained with the same
3000-step V0.5.1 protocol and `--device auto`.  In this Codex runtime,
`--device auto` currently falls back to CPU because the MPS tensor probe fails;
the accelerator diagnostic is stored in each run's `metrics.json`.

Commands:

```bash
pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py \
  --D 9 \
  --n-s 48 \
  --n-z 128 \
  --steps 3000 \
  --hidden 64 \
  --layers 4 \
  --log-every 500 \
  --n-residual 2048 \
  --device auto \
  --run-name v1_anchor_D9_3000

pinn/.venv/bin/python pinn/train_ritz_2d_selfadjoint_pinn.py \
  --D 15 \
  --n-s 48 \
  --n-z 128 \
  --steps 3000 \
  --hidden 64 \
  --layers 4 \
  --log-every 500 \
  --n-residual 2048 \
  --device auto \
  --run-name v1_anchor_D15_3000
```

Full checkpoint anchor table:

```text
D   E_ref         E_PINN        rel_error      strong_L2
6   0.7409156024  0.7410645485  2.0103e-04    1.7344e-02
9   0.7604497941  0.7605930567  1.8839e-04    1.7382e-02
12  0.7898376934  0.7899821997  1.8296e-04    1.8361e-02
15  0.8317674017  0.8319042921  1.6458e-04    2.0537e-02
18  0.8907781421  0.8908920884  1.2792e-04    2.0674e-02
```

Gate status:

```text
max rel_error = 2.0103e-04 < 5e-4
median rel_error = 1.8296e-04 < 3e-4
max strong_L2 = 2.0674e-02 < 5e-2
```

This closes the fixed-checkpoint anchor stage for the lowest self-adjoint
branch.  The next V1 implementation step can be a narrow parametric-\(D\)
network with checkpoint loss against these anchors.
