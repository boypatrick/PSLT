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

## V1.1 Parametric-D Scaffold Status

Implemented file:

- `train_v1_parametric_d_ritz_pinn.py`

This is a scaffold for the lowest-branch parametric emulator

```math
u_\theta=u_\theta(s,z,D),
\qquad
E_\theta(D)=E_D[u_\theta(\cdot,\cdot,D)].
```

It reads `v1_anchor_summary.csv`, minimizes the checkpoint Ritz average, and
adds a branch-coherence penalty

```math
\mathcal L_{\rm chk}
=\frac1{|\mathcal D_{\rm chk}|}
\sum_{D_j}
\left(
\frac{E_\theta(D_j)-E^{\rm FV}_0(D_j)}
     {\max(1,|E^{\rm FV}_0(D_j)|)}
\right)^2.
```

It also records the accelerator diagnostic from `--device auto`, so MPS/CUDA
usage cannot be silently misreported.

Smoke command:

```bash
pinn/.venv/bin/python pinn/train_v1_parametric_d_ritz_pinn.py \
  --n-s 16 \
  --n-z 32 \
  --steps 3 \
  --hidden 24 \
  --layers 2 \
  --log-every 1 \
  --n-residual 128 \
  --device auto \
  --run-name v1_parametric_smoke_3
```

The three-step smoke intentionally does not pass the physical gate; it only
checks that the loss, checkpoint metrics, residual metrics, CSV/JSON output,
and accelerator fallback work.  The first real training run should keep the
same checkpoint set and use the acceptance gates above.

## V1.1 First Formal Parametric Run

Tracked summary:

- `v1_parametric_first_run_summary.csv`

Command:

```bash
pinn/.venv/bin/python pinn/train_v1_parametric_d_ritz_pinn.py \
  --n-s 32 \
  --n-z 80 \
  --steps 2000 \
  --hidden 80 \
  --layers 4 \
  --log-every 250 \
  --n-residual 1024 \
  --device auto \
  --run-name v1_parametric_D6_18_n32_z80_2000
```

Result:

```text
D   E_ref         E_parametric  rel_error    strong_L2
6   0.7409156024  0.7426179051  2.2976e-03  8.1964e-02
9   0.7604497941  0.7620103955  2.0522e-03  8.4729e-02
12  0.7898376934  0.7916654944  2.3141e-03  8.9925e-02
15  0.8317674017  0.8335520625  2.1456e-03  9.4871e-02
18  0.8907781421  0.8938794136  3.4815e-03  1.1577e-01
```

Gate status:

```text
max_rel_error = 3.4815e-03 > 5e-4
median_rel_error = 2.2976e-03 > 3e-4
max_strong_L2 = 1.1577e-01 > 5e-2
```

Reading: the cold-start parametric network is directionally useful but not
gate-safe.  It moves the checkpoint energies from order-one relative error to
the \(2\times10^{-3}\)--\(3.5\times10^{-3}\) band, but the residual remains
too high.  The next parametric attempt should not simply be longer cold-start
training.  It should add continuation/pretraining from the fixed-\(D\) anchors
or a checkpoint-conditioned basis head.

## V1.2 Continuation Parametric Run

Tracked summary:

- `v1_parametric_continuation_summary.csv`

The second formal run initializes from the cold-start parametric model and uses
a smaller learning rate with a slightly stronger checkpoint penalty.  This is
the minimal continuation/pretraining variant: same physics target, same
checkpoint set, but no random restart.

Command:

```bash
pinn/.venv/bin/python pinn/train_v1_parametric_d_ritz_pinn.py \
  --n-s 32 \
  --n-z 80 \
  --steps 4000 \
  --hidden 80 \
  --layers 4 \
  --lr 5e-4 \
  --w-checkpoint 300 \
  --log-every 500 \
  --n-residual 1024 \
  --device auto \
  --init-from-run v1_parametric_D6_18_n32_z80_2000 \
  --run-name v1_parametric_D6_18_n32_z80_continue_4000
```

Result:

```text
D   E_ref         E_parametric  rel_error    strong_L2
6   0.7409156024  0.7411381602  3.0038e-04  2.6193e-02
9   0.7604497941  0.7606557012  2.7077e-04  2.8688e-02
12  0.7898376934  0.7900170088  2.2703e-04  3.1394e-02
15  0.8317674017  0.8319070935  1.6795e-04  3.3578e-02
18  0.8907781421  0.8908556700  8.7034e-05  3.6608e-02
```

Gate status:

```text
max_rel_error = 3.0038e-04 < 5e-4
median_rel_error = 2.2703e-04 < 3e-4
max_strong_L2 = 3.6608e-02 < 5e-2
```

Reading: V1.2 is the first gate-passing parametric-\(D\) checkpoint emulator.
It should still be treated as a checkpoint result, not a dense-\(D\)
certificate.  The next step is to evaluate dense intermediate \(D\) points and
verify suspicious cells with the self-adjoint finite-volume solver.

## V1.3 Dense-D Diagnostic Evaluation

Implemented file:

- `evaluate_v1_parametric_d.py`

Tracked summary:

- `v1_dense_eval_dstep1p5_summary.csv`

Command:

```bash
pinn/.venv/bin/python pinn/evaluate_v1_parametric_d.py \
  --run v1_parametric_D6_18_n32_z80_continue_4000 \
  --D-min 6 \
  --D-max 18 \
  --D-step 1.5 \
  --device auto \
  --run-name v1_dense_eval_dstep1p5
```

Dense diagnostic result:

```text
D     E_parametric  strong_L2  monotone_ok  residual_ok
6.0   0.7411381602  2.3086e-02  yes         yes
7.5   0.7497829795  2.4004e-02  yes         yes
9.0   0.7606557012  2.5132e-02  yes         yes
10.5  0.7739714384  2.6241e-02  yes         yes
12.0  0.7900170088  2.7156e-02  yes         yes
13.5  0.8091673255  2.7809e-02  yes         yes
15.0  0.8319070935  2.8292e-02  yes         yes
16.5  0.8588642478  2.8866e-02  yes         yes
18.0  0.8908556700  3.0036e-02  yes         yes
```

Gate reading:

```text
monotone_ok = true
max_dense_strong_L2 = 3.0036e-02 < 5e-2
needs_finite_volume_check = false
```

This is not a continuum theorem, but it is a clean dense diagnostic for the
trained checkpoint model.  Since no monotonicity or residual anomaly appears on
the \(1.5\)-spaced dense grid, no immediate finite-volume suspicious-cell check
is triggered by V1.3.
