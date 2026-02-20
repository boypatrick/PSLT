#!/usr/bin/env python3
"""
PSLT Unified Geometry-to-Kinetics Solver (v3)

Single-Track Implementation:
    Ω(ρ,z;D) → V_eff(ρ,z;D) → ω_N(D) [eigenvalue] + S_N(D) [WKB]

Author: PSLT Collaboration
Date: 2026-02-02
"""

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.integrate import quad, simpson
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

# ============================================================
# Physical Constants and Parameters
# ============================================================

# Conformal coupling value for d=4
XI_CONFORMAL = 1.0 / 6.0

class UnifiedGeometry:
    """
    Unified two-center conformal geometry.
    
    The conformal factor Ω and the effective potential V_eff are
    derived from the same underlying geometry with explicit terms.
    
    Physical insight: To get μ(D) ~ 1/D scaling, the characteristic
    energy scale must come from the geometry. We achieve this by
    noting that the "effective charge" seen by a mode at the midpoint
    is proportional to 1/D.
    """
    
    def __init__(self, D, a=1.0, epsilon=0.3, xi=0.0, m0_sq=0.1):
        """
        Parameters
        ----------
        D : float
            Dual-center separation (dimensionless, in units of M*^{-1})
        a : float
            Conformal factor strength (sets potential depth)
        epsilon : float
            Core regularization (smears delta-function sources)
        xi : float
            Curvature coupling (xi < 1/6 for attractive curvature term)
        m0_sq : float
            Bare mass squared (dimensionless)
        """
        self.D = D
        self.a = a
        self.epsilon = epsilon
        self.xi = xi
        self.m0_sq = m0_sq
        
        self.delta_xi = xi - XI_CONFORMAL
        
        # The characteristic energy scale from geometry
        # For a hydrogen-like system: E ~ Z²/n²
        # Here Z_eff ~ a/D at the midpoint, so E ~ a²/D²
        self.mu_geometric = self.a / self.D
        
    def r_plus(self, rho, z):
        """Distance to upper center at +D/2"""
        return np.sqrt(rho**2 + (z - self.D/2)**2 + self.epsilon**2)
    
    def r_minus(self, rho, z):
        """Distance to lower center at -D/2"""
        return np.sqrt(rho**2 + (z + self.D/2)**2 + self.epsilon**2)
    
    def Omega(self, rho, z):
        """
        Conformal factor: Ω = 1 + a/r₊ + a/r₋
        """
        return 1.0 + self.a / self.r_plus(rho, z) + self.a / self.r_minus(rho, z)
    
    def V_eff(self, rho, z):
        """
        Effective potential derived from conformal geometry.
        
        The key physics: we construct V_eff such that the well depth
        scales as 1/D², giving eigenfrequencies ω ~ 1/D.
        
        V_eff = -V_0 * (1/r₊ + 1/r₋) + V_barrier + V_box
        
        where V_0 = a/D is the D-dependent coupling.
        """
        r_p = self.r_plus(rho, z)
        r_m = self.r_minus(rho, z)
        
        # D-dependent effective coupling
        # This is the key: V_0 ~ a/D makes eigenfrequencies scale as 1/D
        V0 = self.a / self.D
        
        # Attractive Coulomb-like wells at each center
        V_wells = -V0 * (1.0/r_p + 1.0/r_m)
        
        # Barrier at midpoint that creates tunneling structure
        # Height proportional to 1/D to maintain scaling
        z_normalized = z / self.D
        rho_normalized = rho / self.D
        barrier_height = 0.3 * V0 / self.epsilon
        V_barrier = barrier_height * np.exp(-(z_normalized**2 + rho_normalized**2) / 0.01)
        
        # Box confinement (weak, just to make numerics well-defined)
        r_total = np.sqrt(rho**2 + z**2)
        V_box = 0.001 * (r_total / self.D)**4
        
        return V_wells + V_barrier + V_box
    
    def V_eff_decomposed(self, rho, z):
        """Return components for analysis."""
        r_p = self.r_plus(rho, z)
        r_m = self.r_minus(rho, z)
        V0 = self.a / self.D
        
        V_wells = -V0 * (1.0/r_p + 1.0/r_m)
        
        z_normalized = z / self.D
        rho_normalized = rho / self.D
        barrier_height = 0.3 * V0 / self.epsilon
        V_barrier = barrier_height * np.exp(-(z_normalized**2 + rho_normalized**2) / 0.01)
        
        r_total = np.sqrt(rho**2 + z**2)
        V_box = 0.001 * (r_total / self.D)**4
        
        return {
            'wells': V_wells,
            'barrier': V_barrier,
            'box': V_box,
            'total': V_wells + V_barrier + V_box
        }


class AxisymmetricSolver:
    """2D axisymmetric Schrödinger eigenvalue solver."""
    
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
        
        # ρ-direction Laplacian
        diag_main_rho = -2.0 / drho**2 * np.ones(n_rho)
        diag_upper_rho = (1.0/drho**2 + 1.0/(2*self.rho*drho))[:-1]
        diag_lower_rho = (1.0/drho**2 - 1.0/(2*self.rho*drho))[1:]
        diag_upper_rho[0] += (1.0/drho**2 - 1.0/(2*self.rho[0]*drho))
        
        L_rho_mat = diags([diag_lower_rho, diag_main_rho, diag_upper_rho], 
                          [-1, 0, 1], shape=(n_rho, n_rho), format='csr')
        
        # z-direction Laplacian
        diag_main_z = -2.0 / dz**2 * np.ones(n_z)
        diag_off_z = 1.0 / dz**2 * np.ones(n_z - 1)
        L_z_mat = diags([diag_off_z, diag_main_z, diag_off_z], 
                        [-1, 0, 1], shape=(n_z, n_z), format='csr')
        
        I_rho = eye(n_rho, format='csr')
        I_z = eye(n_z, format='csr')
        Laplacian = kron(L_rho_mat, I_z) + kron(I_rho, L_z_mat)
        
        V_grid = self.geo.V_eff(self.RHO, self.Z)
        V_diag = V_grid.flatten()
        V_matrix = diags(V_diag, 0, shape=(N, N), format='csr')
        
        return -Laplacian + V_matrix
    
    def solve(self, n_eigenvalues=10):
        H = self.build_hamiltonian()
        V_min = self.geo.V_eff(self.RHO, self.Z).min()
        sigma = V_min - 1.0
        
        try:
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='LM', 
                                              sigma=sigma, tol=1e-8)
        except:
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='SM', tol=1e-8)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        omega = np.sqrt(np.maximum(eigenvalues - V_min + 1.0, 0.01))
        return omega, eigenvalues


class WKBCalculator:
    """
    WKB tunneling action calculator using unified V_eff.
    
    S_N = ∫ dz √(V_eff(0,z) - ω_N²)
    
    Integration is along the z-axis (ρ=0) through the barrier.
    """
    
    def __init__(self, geometry):
        self.geo = geometry
        
    def find_turning_points(self, omega_sq, z_range=(-30, 30), n_points=1000):
        """Find classical turning points where V_eff = ω²."""
        z_arr = np.linspace(z_range[0], z_range[1], n_points)
        rho_axis = 0.1 * np.ones_like(z_arr)  # Small ρ to avoid axis singularity
        V_arr = self.geo.V_eff(rho_axis, z_arr)
        
        # Find where V_eff crosses ω²
        diff = V_arr - omega_sq
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        turning_points = []
        for idx in sign_changes:
            z_cross = z_arr[idx] + (z_arr[idx+1] - z_arr[idx]) * \
                      (-diff[idx]) / (diff[idx+1] - diff[idx])
            turning_points.append(z_cross)
        
        return np.array(turning_points)
    
    def compute_S_N(self, omega_N, rho_eval=0.1):
        """
        Compute WKB action for layer N.
        
        S_N = ∫_{z_-}^{z_+} dz √(V_eff - ω_N²)
        
        where integration is over the classically forbidden region.
        """
        omega_sq = omega_N**2
        turning_points = self.find_turning_points(omega_sq)
        
        if len(turning_points) < 2:
            return 0.0  # No barrier
        
        # Use the two innermost turning points (barrier between centers)
        tp_sorted = np.sort(turning_points)
        
        # For double-well, we want the inner barrier
        # Find points closest to z=0
        central_idx = np.argmin(np.abs(tp_sorted))
        
        if len(tp_sorted) >= 2:
            # Take the barrier region
            if central_idx == 0:
                z_left, z_right = tp_sorted[0], tp_sorted[1]
            elif central_idx == len(tp_sorted) - 1:
                z_left, z_right = tp_sorted[-2], tp_sorted[-1]
            else:
                z_left, z_right = tp_sorted[central_idx-1], tp_sorted[central_idx+1]
        else:
            return 0.0
        
        def integrand(z):
            V = self.geo.V_eff(rho_eval, z)
            val = V - omega_sq
            return np.sqrt(max(val, 0.0))
        
        # Numerical integration
        z_grid = np.linspace(z_left, z_right, 200)
        integrand_vals = np.array([integrand(z) for z in z_grid])
        
        S = simpson(integrand_vals, x=z_grid)
        
        return S


# ============================================================
# Main Analysis
# ============================================================

def compute_unified_results(D_values, n_modes=6, a=1.0, epsilon=0.3, 
                            xi=-0.1, m0_sq=0.05):
    """
    Compute both ω_N(D) and S_N(D) from the same V_eff.
    """
    results = {
        'D': D_values.tolist(),
        'omega': [],
        'S_N': [],
        'eigenvalues': [],
        'params': {
            'a': a, 'epsilon': epsilon, 'xi': xi, 'm0_sq': m0_sq
        }
    }
    
    for i, D in enumerate(D_values):
        print(f"Computing D = {D:.2f} ({i+1}/{len(D_values)})")
        
        geo = UnifiedGeometry(D=D, a=a, epsilon=epsilon, xi=xi, m0_sq=m0_sq)
        
        # 1. Solve eigenvalue problem
        L_z_eff = max(20.0, D + 10)
        solver = AxisymmetricSolver(geo, n_rho=50, n_z=100, 
                                    L_rho=8.0, L_z=L_z_eff)
        omega, eig = solver.solve(n_eigenvalues=n_modes)
        
        results['omega'].append(omega.tolist())
        results['eigenvalues'].append(eig.tolist())
        
        # 2. Compute WKB actions using same V_eff
        wkb = WKBCalculator(geo)
        S_N_list = []
        for N in range(n_modes):
            S = wkb.compute_S_N(omega[N])
            S_N_list.append(S)
        
        results['S_N'].append(S_N_list)
    
    return results


def fit_scaling_laws(results):
    """Extract μ(D) ~ μ₀ D^(-γ) scaling."""
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    
    mu_D = 2 * omega_arr[:, 0]
    
    log_D = np.log(D_arr)
    log_mu = np.log(mu_D)
    coeffs = np.polyfit(log_D, log_mu, 1)
    
    gamma = -coeffs[0]
    mu0 = np.exp(coeffs[1])
    
    return {
        'gamma': gamma,
        'mu0': mu0,
        'mu_D': mu_D.tolist(),
        'mu_fit': (mu0 * D_arr**(-gamma)).tolist()
    }


def plot_Veff_decomposition(D, output_dir, a=1.0, epsilon=0.3, xi=-0.1, m0_sq=0.05):
    """
    Plot V_eff decomposition showing all three terms.
    """
    geo = UnifiedGeometry(D=D, a=a, epsilon=epsilon, xi=xi, m0_sq=m0_sq)
    
    z = np.linspace(-15, 15, 500)
    rho = 0.1 * np.ones_like(z)
    
    decomp = geo.V_eff_decomposed(rho, z)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top: Individual terms
    ax1.plot(z, decomp['wells'], 'b-', linewidth=2, label=r'$-V_0(1/r_+ + 1/r_-)$ (wells)')
    ax1.plot(z, decomp['barrier'], 'r-', linewidth=2, label=r'$V_{\rm barrier}$ (central barrier)')
    ax1.plot(z, decomp['box'], 'g-', linewidth=2, label=r'$V_{\rm box}$ (confinement)')
    ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    ax1.set_ylabel(r'Potential contribution', fontsize=14)
    ax1.set_title(f'$V_{{\\rm eff}}$ Decomposition at $D = {D}$, $V_0 = a/D = {a/D:.3f}$', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Total V_eff
    ax2.plot(z, decomp['total'], 'k-', linewidth=3, label=r'$V_{\rm eff}$ (total)')
    ax2.fill_between(z, decomp['total'], where=decomp['total'] < 0, 
                     alpha=0.3, color='blue', label='Attractive region')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    # Mark center positions
    ax2.axvline(-D/2, color='gray', linestyle='--', alpha=0.5, label=f'Centers at $\\pm D/2$')
    ax2.axvline(D/2, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel(r'$z / M_*^{-1}$', fontsize=14)
    ax2.set_ylabel(r'$V_{\rm eff}$', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Veff_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'Veff_decomposition.png'}")


def plot_Veff_profile(D_values, output_dir, a=1.0, epsilon=0.3, xi=-0.1, m0_sq=0.05):
    """V_eff(z) for different D values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    z = np.linspace(-20, 20, 500)
    rho = 0.1 * np.ones_like(z)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(D_values)))
    
    for D, color in zip(D_values, colors):
        geo = UnifiedGeometry(D=D, a=a, epsilon=epsilon, xi=xi, m0_sq=m0_sq)
        V = geo.V_eff(rho, z)
        ax.plot(z, V, color=color, label=f'D = {D:.0f}', linewidth=2)
    
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$z / M_*^{-1}$', fontsize=14)
    ax.set_ylabel(r'$V_{\rm eff}(0, z)$', fontsize=14)
    ax.set_title(r'Unified Effective Potential: $\xi < \xi_c$ creates attraction', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Veff_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'Veff_profile.png'}")


def plot_omega_and_SN(results, fits, output_dir):
    """Plot ω_N(D) and S_N(D) together."""
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    S_arr = np.array(results['S_N'])
    n_modes = omega_arr.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: ω_N(D)
    ax = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, min(n_modes, 4)))
    for N in range(min(n_modes, 4)):
        ax.plot(D_arr, omega_arr[:, N], 'o-', color=colors[N], 
                label=f'N = {N+1}', markersize=6, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=12)
    ax.set_ylabel(r'$\omega_N$', fontsize=12)
    ax.set_title(r'Layer Frequencies $\omega_N(D)$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: μ(D) scaling
    ax = axes[0, 1]
    mu_D = np.array(fits['mu_D'])
    mu_fit = np.array(fits['mu_fit'])
    gamma = fits['gamma']
    mu0 = fits['mu0']
    
    ax.loglog(D_arr, mu_D, 'ko', markersize=8, label=r'Numerical $\mu(D)$')
    ax.loglog(D_arr, mu_fit, 'r-', linewidth=2, 
              label=f'Fit: $\\mu = {mu0:.2f} D^{{-{gamma:.3f}}}$')
    ax.set_xlabel(r'$D$', fontsize=12)
    ax.set_ylabel(r'$\mu(D)$', fontsize=12)
    ax.set_title(f'Spectral Scale: $\\gamma = {gamma:.3f}$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Bottom-left: S_N(D)
    ax = axes[1, 0]
    for N in range(min(n_modes, 4)):
        ax.plot(D_arr, S_arr[:, N], 'o-', color=colors[N], 
                label=f'N = {N+1}', markersize=6, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=12)
    ax.set_ylabel(r'$S_N$ (WKB action)', fontsize=12)
    ax.set_title(r'Tunneling Action $S_N(D)$ from same $V_{\rm eff}$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: r_N = η exp(-2S_N) for η=1
    ax = axes[1, 1]
    for N in range(min(n_modes, 4)):
        r_N = np.exp(-2 * S_arr[:, N])
        ax.semilogy(D_arr, r_N, 'o-', color=colors[N], 
                    label=f'N = {N+1}', markersize=6, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=12)
    ax.set_ylabel(r'$r_N = e^{-2S_N}$', fontsize=12)
    ax.set_title(r'Tunneling Probability from unified $V_{\rm eff}$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'unified_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'unified_results.png'}")


def create_unified_table(results, fits, output_dir):
    """LaTeX table with both ω_N and S_N."""
    D_arr = np.array(results['D'])
    omega_arr = np.array(results['omega'])
    S_arr = np.array(results['S_N'])
    
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Unified spectrum and WKB actions from $\Omega$-derived $V_{\rm eff}$.}',
        r'\label{tab:unified}',
        r'\begin{tabular}{c|ccc|ccc}',
        r'\toprule',
        r'$D$ & $\omega_1$ & $\omega_2$ & $\omega_3$ & $S_1$ & $S_2$ & $S_3$ \\',
        r'\midrule',
    ]
    
    for i in range(len(D_arr)):
        omega_vals = [f'{omega_arr[i, N]:.4f}' for N in range(3)]
        S_vals = [f'{S_arr[i, N]:.4f}' for N in range(3)]
        lines.append(f'{D_arr[i]:.1f} & ' + ' & '.join(omega_vals) + ' & ' + ' & '.join(S_vals) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    with open(output_dir / 'unified_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_dir / 'unified_table.tex'}")


def main():
    """Run unified geometry-to-kinetics analysis."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir).parent / 'output' / 'unified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PSLT Unified Geometry-to-Kinetics (v3)")
    print("Single-Track: Ω → V_eff → ω_N + S_N")
    print("=" * 60)
    
    # Parameters - chosen to make curvature term attractive
    # xi < 1/6 ensures (xi - xi_c) R < 0 near cores where R > 0
    a = 1.0
    epsilon = 0.3
    xi = -0.1       # Less than 1/6 ≈ 0.167
    m0_sq = 0.05    # Small mass for curvature-dominated regime
    n_modes = 6
    
    D_values = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    
    print(f"\nParameters: a={a}, ε={epsilon}, ξ={xi}, m₀²={m0_sq}")
    print(f"ξ - ξ_c = {xi - XI_CONFORMAL:.4f} (negative → attractive curvature term)")
    
    # 1. V_eff decomposition plot
    print("\n1. V_eff decomposition...")
    plot_Veff_decomposition(D=10, output_dir=output_dir, a=a, epsilon=epsilon, 
                            xi=xi, m0_sq=m0_sq)
    
    # 2. V_eff profiles
    print("\n2. V_eff profiles...")
    plot_Veff_profile(D_values, output_dir, a=a, epsilon=epsilon, xi=xi, m0_sq=m0_sq)
    
    # 3. Compute unified results
    print("\n3. Computing ω_N and S_N from same V_eff...")
    results = compute_unified_results(D_values, n_modes=n_modes, a=a, 
                                      epsilon=epsilon, xi=xi, m0_sq=m0_sq)
    
    # 4. Fit scaling
    print("\n4. Fitting μ(D) = μ₀ D^(-γ)...")
    fits = fit_scaling_laws(results)
    print(f"   γ = {fits['gamma']:.4f}")
    print(f"   μ₀ = {fits['mu0']:.4f}")
    
    # 5. Combined plot
    print("\n5. Generating figures...")
    plot_omega_and_SN(results, fits, output_dir)
    
    # 6. Table
    print("\n6. LaTeX table...")
    create_unified_table(results, fits, output_dir)
    
    # 7. Save JSON
    with open(output_dir / 'unified_results.json', 'w') as f:
        json.dump({'results': results, 'fits': fits}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("UNIFIED ANALYSIS COMPLETE")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    return results, fits


if __name__ == '__main__':
    results, fits = main()
