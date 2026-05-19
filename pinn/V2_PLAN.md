# V2 Multi-Mode Self-Adjoint Ritz-PINN Plan

V1 closed the lowest-branch parametric-\(D\) diagnostic.  V2 should not
immediately train a broad parametric multi-mode network.  The first target is a
fixed-\(D\), fixed-box, multi-mode Ritz-PINN that learns a low-dimensional
subspace and compares its Ritz values against the deterministic self-adjoint
finite-volume reference.

## Mathematical Target

In \(s=\rho^2\), the shifted self-adjoint operator is

\[
H_U u=-4s\,\partial_s^2u-4\,\partial_su-\partial_z^2u+U(s,z;D)u,
\]

where

\[
U=V_{\rm eff}-m_0^2
=m_0^2(\Omega^2-1)+(1-6\xi)\Omega^{-1}\Delta\Omega.
\]

The energy form is

\[
\mathfrak h_D[u,v]=
\int
\left(
4s\,u_s v_s+u_zv_z+Uuv
\right)\,ds\,dz,
\]

and the mass form is

\[
\mathfrak g[u,v]=\int uv\,ds\,dz.
\]

For a trial basis \(u_1,\ldots,u_K\), V2 assembles

\[
H_{ij}=\mathfrak h_D[u_i,u_j],
\qquad
G_{ij}=\mathfrak g[u_i,u_j],
\]

and reports the generalized Ritz spectrum

\[
Hc=EGc.
\]

This is the correct variational object for near-degenerate branches: individual
network outputs need not match unique eigenfunctions, but the learned subspace
must reproduce the reference Ritz values and have a well-conditioned Gram
matrix.

## V2.0 Fixed-D Gate

Initial fixed point:

- \(D=12\), because V0.5 and V1 already have stable references there.
- \(K=3\), because this is the first nontrivial multi-mode gate after the
  near-degenerate lowest pair.
- Self-adjoint reference source:
  `pinn/runs/v0p5p1_selfadjoint_D12_n50x500/metrics.json`.

Suggested command:

```bash
pinn/.venv/bin/python pinn/train_v2_fixed_d_multimode_ritz_pinn.py \
  --D 12 \
  --modes 3 \
  --n-s 32 \
  --n-z 96 \
  --steps 1200 \
  --hidden 80 \
  --layers 4 \
  --log-every 200 \
  --device auto \
  --run-name v2_fixed_D12_K3_smoke
```

Checkpoint targets:

\[
\max_k |E_k^{\rm Ritz}-E_k^{\rm FV}|/|E_k^{\rm FV}| < 5\times10^{-3},
\]

\[
\max_{i\ne j}
\left|
\frac{G_{ij}}{\sqrt{G_{ii}G_{jj}}}
\right|<0.15,
\]

and

\[
\max_k
\|H_Uu_k^{\rm Ritz}-E_ku_k^{\rm Ritz}\|_2/\|u_k^{\rm Ritz}\|_2<0.25.
\]

These are intentionally smoke-level thresholds.  V2.0 only asks whether the
multi-mode machinery is numerically sane; it is not a continuum theorem and not
a replacement for finite-volume eigensolves.

## Decision Rule

If V2.0 passes, continue to V2.1:

1. repeat at \(D=6,18\);
2. compare subspace errors and residuals across the branch-splitting endpoints;
3. only then consider a parametric-\(D\), \(K=3\) network.

If V2.0 fails, do not open parametric multi-mode training.  First tighten the
fixed-\(D\) ansatz by adding parity-separated heads or Gram-Schmidt
orthogonalization inside the network.

## V2.0 Smoke Status

Tracked summary:

- `v2_fixed_d12_smoke_summary.csv`

Runs completed:

```text
run                                max_rel_E   max_residual_L2  max_corr_offdiag
v2_fixed_D12_K3_smoke              9.02e-03    1.45e-01         4.18e-03
v2_fixed_D12_K3_tight_2400         6.26e-02    3.14e-01         3.72e-01
v2_fixed_D12_K3_lr8e4_2200         1.07e-02    1.54e-01         1.95e-03
v2_fixed_D12_K3_grid48x128_1600    1.86e-02    2.01e-01         4.25e-03
v2_fixed_D12_K3_parity_lobes_1600  4.22e-01    8.72e-01         2.93e-01
v2_fixed_D12_K3_best_2400          3.22e-03    9.31e-02         2.89e-03
```

Reading:

- The polynomial subspace scaffold is the correct V2 carrier.  With
  best-checkpoint retention, the D=12, K=3 smoke passes the initial gate:
  `max_rel_E=3.22e-03 < 5e-3`,
  `max_residual_L2=9.31e-02 < 2.5e-1`, and
  `max_corr_offdiag=2.89e-03 < 0.15`.
- The longer high-reference-weight run is not monotone stable; it improved near
  step 2000 and then lost orthogonality by the final checkpoint.  Best-checkpoint
  retention is therefore part of the V2 scaffold.
- The narrow two-lobe parity seed is a negative result.  It is too stiff for the
  first three self-adjoint modes at \(D=12\), and should not be used as the
  default V2 carrier.

Current V2 status:

```text
V2.0 CLOSED POSITIVE / FIXED-D D=12 SMOKE PASSED
```

Next recommended implementation step:

1. repeat the best-checkpoint polynomial scaffold at \(D=6\) and \(D=18\);
2. compare endpoint behavior, because these are where the low-mode branch
   spacing differs most from the \(D=12\) midpoint;
3. only after \(D=6,12,18\) pass should V2 consider a parametric-\(D\), K=3
   network.

## V2.1 Endpoint Status

Tracked endpoint summary:

- `v2_fixed_endpoint_summary.csv`

Endpoint runs:

```text
D   max_rel_E   max_residual_L2  max_corr_offdiag
6   2.69e-03    4.48e-02         4.34e-03
12  3.22e-03    9.31e-02         2.89e-03
18  2.52e-03    1.17e-01         2.81e-03
```

All three fixed-\(D\), \(K=3\) endpoint/midpoint smokes pass the V2 gate:

\[
\max_D\max_k
\frac{|E^{\rm Ritz}_k(D)-E^{\rm FV}_k(D)|}{|E^{\rm FV}_k(D)|}
=3.22\times 10^{-3}<5\times10^{-3},
\]

\[
\max_D \max_{i\ne j}
\left|
\frac{G_{ij}}{\sqrt{G_{ii}G_{jj}}}
\right|
=4.34\times10^{-3}<0.15,
\]

and

\[
\max_D\max_k
\|H_Uu_k^{\rm Ritz}-E_ku_k^{\rm Ritz}\|_2/\|u_k^{\rm Ritz}\|_2
=1.17\times10^{-1}<2.5\times10^{-1}.
\]

Current V2 status:

```text
V2.1 CLOSED POSITIVE / FIXED-D ENDPOINTS PASSED
```

Next recommended implementation step:

1. open a separate V2.2 parametric-\(D\), \(K=3\) scaffold;
2. use the fixed-\(D\) endpoint runs only as checkpoint anchors;
3. keep finite-volume references as source of truth and treat the neural model
   as a differentiable emulator, not a replacement eigensolver.
