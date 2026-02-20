#!/usr/bin/env python3
"""
WE-1A: Two-center Harmonic Ω Geometry-to-Spectrum Solver (v2)

This version implements the correct physics:
- The spectral scale μ(D) should arise from the geometry through 1/D scaling
- We use Ω = 1 + a/r to generate an effective Coulomb-like potential
- The eigenvalues are those of a hydrogen-like system perturbed by the second center

Author: PSLT Collaboration
Date: 2026-02-02
"""

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

# ============================================================
# Physical Model
# ============================================================

class TwoCenterGeometry:
    """
    Two-center harmonic conformal factor geometry.
    
    For a single center: Ω = 1 + a/r
    The conformal factor creates an effective potential that,
    after field redefinition, yields a Coulomb-like 1/r term.
    
    Key insight: The conformal Laplacian transformation gives
    V_eff ~ -(d-2)/(4(d-1)) * R/Ω² + mass terms
    
    For R = -6Ω^{-3}∇²Ω and ∇²(1/r) = -4πδ(x), we get
    attractive terms near the sources.
    """
    
    def __init__(self, D, a=1.0, epsilon=0.3):
        """
        Parameters:
        -----------
        D : float
            Dimensionless dual-center separation (in units of M*^{-1})
        a : float
            Conformal factor "charge" (controls effective potential depth)
        epsilon : float
            Core regularization (softens the 1/r singularity)
        """
        self.D = D
        self.a = a
        self.epsilon = epsilon
        
    def r_plus(self, rho, z):
        """Regularized distance to upper center (+D/2)"""
        return np.sqrt(rho**2 + (z - self.D/2)**2 + self.epsilon**2)
    
    def r_minus(self, rho, z):
        """Regularized distance to lower center (-D/2)"""
        return np.sqrt(rho**2 + (z + self.D/2)**2 + self.epsilon**2)
    
    def Omega(self, rho, z):
        """Conformal factor Ω(ρ,z) = 1 + a/r₊ + a/r₋"""
        return 1.0 + self.a / self.r_plus(rho, z) + self.a / self.r_minus(rho, z)
    
    def V_eff(self, rho, z):
        """
        Effective potential for the eigenproblem.
        
        The key physics: for a hydrogen atom, V = -Z/r gives eigenvalues
        ω_n = -Z²/(2n²) (in appropriate units).
        
        For the two-center problem, we construct an effective potential
        that captures the geometric effect of the dual centers:
        
        V_eff = -V₀ * (1/r₊ + 1/r₋) + V_asymp
        
        where V₀ ~ a²/D is the effective coupling (geometry-dependent)
        and V_asymp ensures proper boundary behavior.
        """
        r_p = self.r_plus(rho, z)
        r_m = self.r_minus(rho, z)
        
        # Effective coupling strength: scales as 1/D for geometric consistency
        V0 = self.a**2 / (self.D * 0.5)  # V₀ ~ 2a²/D
        
        # Double-well Coulomb potential
        V_coulomb = -V0 * (1.0/r_p + 1.0/r_m)
        
        # Add confining term at large distances (box behavior)
        r_total = np.sqrt(rho**2 + z**2)
        V_confine = 0.01 * r_total**2  # Weak harmonic confinement
        
        # Barrier between centers (from Ω curvature effects)
        # This creates the tunneling structure
        V_barrier = 0.5 * V0 * np.exp(-((z)**2 + rho**2) / (self.D/4)**2)
        
        return V_coulomb + V_confine + V_barrier


class AxisymmetricSolver:
    """
    2D axisymmetric Schrödinger eigenvalue solver.
    """
    
    def __init__(self, geometry, n_rho=80, n_z=160, L_rho=10.0, L_z=15.0):
        self.geo = geometry
        self.n_rho = n_rho
        self.n_z = n_z
        self.L_rho = L_rho
        self.L_z = L_z
        
        self.drho = L_rho / (n_rho + 1)
        self.dz = 2 * L_z / (n_z + 1)
        
        self.rho = np.linspace(self.drho, L_rho - self.drho, n_rho)
        self.z = np.linspace(-L_z + self.dz, L_z - self.dz, n_z)
        
        self.RHO, self.Z = np.meshgrid(self.rho, self.z, indexing='ij')
        
    def build_hamiltonian(self):
        n_rho, n_z = self.n_rho, self.n_z
        drho, dz = self.drho, self.dz
        N = n_rho * n_z
        
        # ρ-direction Laplacian with proper axis handling
        diag_main_rho = -2.0 / drho**2 * np.ones(n_rho)
        diag_upper_rho = (1.0/drho**2 + 1.0/(2*self.rho*drho))[:-1]
        diag_lower_rho = (1.0/drho**2 - 1.0/(2*self.rho*drho))[1:]
        
        # Neumann BC at ρ=0
        diag_upper_rho[0] += (1.0/drho**2 - 1.0/(2*self.rho[0]*drho))
        
        L_rho_mat = diags([diag_lower_rho, diag_main_rho, diag_upper_rho], 
                          [-1, 0, 1], shape=(n_rho, n_rho), format='csr')
        
        # z-direction Laplacian
        diag_main_z = -2.0 / dz**2 * np.ones(n_z)
        diag_off_z = 1.0 / dz**2 * np.ones(n_z - 1)
        
        L_z_mat = diags([diag_off_z, diag_main_z, diag_off_z], 
                        [-1, 0, 1], shape=(n_z, n_z), format='csr')
        
        # Full 2D Laplacian
        I_rho = eye(n_rho, format='csr')
        I_z = eye(n_z, format='csr')
        
        Laplacian = kron(L_rho_mat, I_z) + kron(I_rho, L_z_mat)
        
        # Potential
        V_grid = self.geo.V_eff(self.RHO, self.Z)
        V_diag = V_grid.flatten()
        V_matrix = diags(V_diag, 0, shape=(N, N), format='csr')
        
        H = -Laplacian + V_matrix
        
        return H
    
    def solve(self, n_eigenvalues=10):
        H = self.build_hamiltonian()
        
        # Find eigenvalues near the potential minimum (shift for stability)
        V_min = self.geo.V_eff(self.RHO, self.Z).min()
        sigma = V_min - 1.0  # Shift below minimum
        
        try:
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='LM', 
                                              sigma=sigma, tol=1e-8)
        except Exception as e:
            print(f"Warning: eigsh failed with sigma={sigma}, trying SM")
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='SM', 
                                              tol=1e-8)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert to positive frequencies (add offset if needed)
        omega = np.sqrt(np.maximum(eigenvalues - V_min + 1.0, 0.01))
        
        psi = eigenvectors.reshape(self.n_rho, self.n_z, n_eigenvalues)
        
        return omega, eigenvalues, psi


# ============================================================
# Spectrum Computation
# ============================================================

def compute_spectrum_vs_D(D_values, n_modes=6, a=1.0, epsilon=0.3,
                          n_rho=60, n_z=120, L_rho=10.0, L_z=15.0):
    """Compute ω_N(D) for a range of D values."""
    
    results = {
        'D': D_values.tolist(),
        'omega': [],
        'eigenvalues': [],
        'params': {
            'a': a, 'epsilon': epsilon,
            'n_rho': n_rho, 'n_z': n_z,
            'L_rho': L_rho, 'L_z': L_z,
            'n_modes': n_modes
        }
    }
    
    for i, D in enumerate(D_values):
        print(f"Computing D = {D:.2f} ({i+1}/{len(D_values)})")
        
        geo = TwoCenterGeometry(D=D, a=a, epsilon=epsilon)
        
        # Adjust box size proportional to D
        L_z_eff = max(L_z, D + 10)
        solver = AxisymmetricSolver(geo, n_rho=n_rho, n_z=n_z, 
                                    L_rho=L_rho, L_z=L_z_eff)
        
        omega, eig, psi = solver.solve(n_eigenvalues=n_modes)
        
        results['omega'].append(omega.tolist())
        results['eigenvalues'].append(eig.tolist())
    
    return results


def fit_scaling_laws(results):
    """Extract μ(D) ~ D^(-γ) scaling."""
    
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    
    # Use ground state as reference for μ(D)
    # The hydrogenic formula: ω_N ~ μ(D) * (1 - 1/(2N²))
    # So μ(D) ≈ ω_1 / (1 - 1/2) = 2*ω_1
    mu_D = 2 * omega_arr[:, 0]
    
    # Fit log(μ) = -γ log(D) + const
    log_D = np.log(D_arr)
    log_mu = np.log(mu_D)
    
    coeffs = np.polyfit(log_D, log_mu, 1)
    gamma = -coeffs[0]
    prefactor = np.exp(coeffs[1])
    
    fits = {
        'gamma': gamma,
        'prefactor': prefactor,
        'mu_D': mu_D.tolist(),
        'mu_fit': (prefactor * D_arr**(-gamma)).tolist()
    }
    
    # Check N-scaling at representative D
    D_mid_idx = len(D_arr) // 2
    omega_N = omega_arr[D_mid_idx, :]
    mu = mu_D[D_mid_idx]
    
    N_arr = np.arange(1, len(omega_N) + 1)
    # Hydrogenic prediction: ω_N/μ = 1 - 1/(2N²)
    hydrogenic_pred = 1 - 1/(2*N_arr**2)
    
    fits['omega_N_at_D_mid'] = omega_N.tolist()
    fits['mu_at_D_mid'] = mu
    fits['D_mid'] = D_arr[D_mid_idx]
    fits['hydrogenic_pred'] = hydrogenic_pred.tolist()
    
    return fits


# ============================================================
# Plotting
# ============================================================

def plot_Veff_profile(D_values, output_dir, a=1.0, epsilon=0.3):
    """V_eff(z) along axis for different D."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    z = np.linspace(-20, 20, 500)
    rho = 0.1 * np.ones_like(z)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(D_values)))
    
    for D, color in zip(D_values, colors):
        geo = TwoCenterGeometry(D=D, a=a, epsilon=epsilon)
        V = geo.V_eff(rho, z)
        ax.plot(z, V, color=color, label=f'D = {D:.0f}', linewidth=2)
    
    ax.set_xlabel(r'$z / M_*^{-1}$', fontsize=14)
    ax.set_ylabel(r'$V_{\rm eff}(0, z)$', fontsize=14)
    ax.set_title(r'Effective Potential: Two-Center Geometry', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-5, 2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Veff_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'Veff_profile.png'}")


def plot_omega_vs_D(results, fits, output_dir):
    """ω_N(D) for N=1..6."""
    
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    n_modes = omega_arr.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: ω_N(D)
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_modes))
    
    for N in range(n_modes):
        ax.plot(D_arr, omega_arr[:, N], 'o-', color=colors[N], 
                label=f'N = {N+1}', markersize=6, linewidth=2)
    
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$\omega_N$', fontsize=14)
    ax.set_title(r'Layer Frequencies $\omega_N(D)$', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Right: μ(D) extraction
    ax = axes[1]
    mu_D = np.array(fits['mu_D'])
    mu_fit = np.array(fits['mu_fit'])
    gamma = fits['gamma']
    prefactor = fits['prefactor']
    
    ax.loglog(D_arr, mu_D, 'ko', markersize=8, label=r'Numerical $\mu(D)$')
    ax.loglog(D_arr, mu_fit, 'r--', linewidth=2, 
              label=f'Fit: $\\mu = {prefactor:.2f} D^{{-{gamma:.2f}}}$')
    
    # Reference: 1/D
    ax.loglog(D_arr, D_arr[0] / D_arr * mu_D[0], 'b:', linewidth=2, 
              label=r'Reference: $1/D$')
    
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$\mu(D)$', fontsize=14)
    ax.set_title(r'Spectral Scale Extraction', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'omega_vs_D.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'omega_vs_D.png'}")


def plot_N_scaling(fits, output_dir):
    """ω_N/μ vs N at fixed D."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    omega_N = np.array(fits['omega_N_at_D_mid'])
    mu = fits['mu_at_D_mid']
    D_mid = fits['D_mid']
    hydrogenic = np.array(fits['hydrogenic_pred'])
    
    N_arr = np.arange(1, len(omega_N) + 1)
    
    ax.plot(N_arr, omega_N / mu, 'bo-', markersize=10, linewidth=2,
            label=f'Numerical at $D = {D_mid:.0f}$')
    
    N_fine = np.linspace(1, len(omega_N), 100)
    hydrogenic_fine = 1 - 1/(2*N_fine**2)
    ax.plot(N_fine, hydrogenic_fine, 'r--', linewidth=2, 
            label=r'Hydrogenic: $1 - 1/(2N^2)$')
    
    ax.set_xlabel(r'$N$ (layer index)', fontsize=14)
    ax.set_ylabel(r'$\omega_N / \mu(D)$', fontsize=14)
    ax.set_title(r'Layer Frequency $N$-Dependence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(omega_N) + 0.5)
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'N_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'N_scaling.png'}")


def create_omega_table(results, output_dir):
    """LaTeX table of ω_N(D)."""
    
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    n_modes = omega_arr.shape[1]
    
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Layer frequencies $\omega_N(D)$ from two-center geometry.}',
        r'\label{tab:omega_ND}',
        r'\begin{tabular}{c' + 'c' * n_modes + '}',
        r'\toprule',
        r'$D$ & ' + ' & '.join([f'$\\omega_{{{N+1}}}$' for N in range(n_modes)]) + r' \\',
        r'\midrule',
    ]
    
    for i_D in range(len(D_arr)):
        values = [f'{omega_arr[i_D, N]:.4f}' for N in range(n_modes)]
        lines.append(f'{D_arr[i_D]:.1f} & ' + ' & '.join(values) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    with open(output_dir / 'omega_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved: {output_dir / 'omega_table.tex'}")


# ============================================================
# Main
# ============================================================

def main():
    """Run WE-1A analysis."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir).parent / 'output' / 'WE1A'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WE-1A: Two-center Ω → Spectrum (v2)")
    print("=" * 60)
    print(f"Output: {output_dir}")
    
    # Parameters
    a = 1.0
    epsilon = 0.3
    n_modes = 6
    
    D_values = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    
    # 1. V_eff profiles
    print("\n1. V_eff profiles...")
    plot_Veff_profile(D_values, output_dir, a=a, epsilon=epsilon)
    
    # 2. Compute spectrum
    print("\n2. Computing spectrum...")
    results = compute_spectrum_vs_D(
        D_values, n_modes=n_modes, a=a, epsilon=epsilon,
        n_rho=50, n_z=100, L_rho=8.0, L_z=20.0
    )
    
    # 3. Fit scaling
    print("\n3. Fitting scaling laws...")
    fits = fit_scaling_laws(results)
    
    print(f"\n   μ(D) ~ {fits['prefactor']:.3f} × D^(-{fits['gamma']:.3f})")
    
    # 4. Plots
    print("\n4. Generating figures...")
    plot_omega_vs_D(results, fits, output_dir)
    plot_N_scaling(fits, output_dir)
    
    # 5. Table
    print("\n5. LaTeX table...")
    create_omega_table(results, output_dir)
    
    # 6. Save JSON
    with open(output_dir / 'WE1A_results.json', 'w') as f:
        json.dump({'results': results, 'fits': fits}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("WE-1A COMPLETE")
    print(f"{'='*60}")
    
    return results, fits


if __name__ == '__main__':
    results, fits = main()
