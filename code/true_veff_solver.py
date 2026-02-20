#!/usr/bin/env python3
"""
TRUE V_eff Solver: Action-Derived Effective Potential

This implements the CORRECT single-track derivation:

From: (□_g - m₀² - ξR)Φ = 0
To:   [-∇² + V_eff]ψ = ω²ψ

where V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω  (DERIVED, not engineered)

Key insight: U ≡ V_eff - m₀² → 0 as r→∞, so bound states have E < 0.

Author: PSLT Collaboration (Corrected)
Date: 2026-02-03
"""

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

# ============================================================
# Physical Constants
# ============================================================

XI_CONFORMAL = 1.0 / 6.0  # 4D conformal coupling


class TrueTwoCenterGeometry:
    """
    Two-center harmonic conformal factor with CORRECT V_eff derivation.
    
    V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω
    
    This is the ONLY form allowed by the action - no engineered terms.
    """
    
    def __init__(self, D, a=1.0, epsilon=0.3, m0=1.0, xi=0.0):
        """
        Parameters
        ----------
        D : float
            Dual-center separation
        a : float
            Conformal factor strength: Ω = 1 + a(1/r₊ + 1/r₋)
        epsilon : float
            Core regularization
        m0 : float
            Bare scalar mass (sets asymptotic threshold)
        xi : float
            Curvature coupling (xi=1/6 is conformal)
        """
        self.D = D
        self.a = a
        self.epsilon = epsilon
        self.m0 = m0
        self.xi = xi
        
        # Check if conformal coupling (derivative term vanishes)
        self.is_conformal = abs(xi - XI_CONFORMAL) < 1e-10
        
    def r_plus(self, rho, z):
        """Distance to upper center at +D/2"""
        return np.sqrt(rho**2 + (z - self.D/2)**2 + self.epsilon**2)
    
    def r_minus(self, rho, z):
        """Distance to lower center at -D/2"""
        return np.sqrt(rho**2 + (z + self.D/2)**2 + self.epsilon**2)
    
    def Omega(self, rho, z):
        """
        Conformal factor: Ω = 1 + a/r₊ + a/r₋
        
        This is the harmonic superposition satisfying ∇²Ω = regularized sources.
        """
        return 1.0 + self.a / self.r_plus(rho, z) + self.a / self.r_minus(rho, z)
    
    def laplacian_Omega(self, rho, z):
        """
        ∇²Ω in 3D spherical for regularized 1/r:
        
        ∇²(1/√(r² + ε²)) = -3ε²/(r² + ε²)^(5/2)
        
        This is EXACT for the smeared source.
        """
        r_p = self.r_plus(rho, z)
        r_m = self.r_minus(rho, z)
        
        # Each center contributes
        lap_single = lambda r: -3.0 * self.epsilon**2 / r**5
        
        return self.a * (lap_single(r_p) + lap_single(r_m))
    
    def V_eff(self, rho, z):
        """
        THE CORRECT V_eff derived from action:
        
        V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω
        
        NO ENGINEERED TERMS (barrier, box, etc.)
        """
        Om = self.Omega(rho, z)
        lap_Om = self.laplacian_Omega(rho, z)
        
        # Mass term
        V_mass = self.m0**2 * Om**2
        
        # Derivative/curvature term (vanishes at xi=1/6)
        coeff = 1.0 - 6.0 * self.xi
        V_deriv = coeff * lap_Om / Om
        
        return V_mass + V_deriv
    
    def U_shifted(self, rho, z):
        """
        U ≡ V_eff - m₀² → 0 as r → ∞
        
        This is the correct potential for bound state analysis.
        Bound states satisfy E < 0.
        """
        return self.V_eff(rho, z) - self.m0**2
    
    def V_eff_components(self, rho, z):
        """Return components for analysis."""
        Om = self.Omega(rho, z)
        lap_Om = self.laplacian_Omega(rho, z)
        
        V_mass = self.m0**2 * Om**2
        coeff = 1.0 - 6.0 * self.xi
        V_deriv = coeff * lap_Om / Om
        
        return {
            'Omega': Om,
            'laplacian_Omega': lap_Om,
            'V_mass': V_mass,
            'V_deriv': V_deriv,
            'V_eff': V_mass + V_deriv,
            'U': V_mass + V_deriv - self.m0**2
        }


class AxisymmetricSolver:
    """
    2D axisymmetric solver for [-∇² + V_eff]ψ = ω²ψ
    
    Uses cylindrical coordinates (ρ, z) with axisymmetry.
    """
    
    def __init__(self, geometry, n_rho=60, n_z=120, L_rho=20.0, L_z=40.0):
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
        """Build [-∇² + V_eff] in cylindrical coordinates."""
        n_rho, n_z = self.n_rho, self.n_z
        drho, dz = self.drho, self.dz
        N = n_rho * n_z
        
        # ρ-direction: -d²/dρ² - (1/ρ)d/dρ
        diag_main_rho = -2.0 / drho**2 * np.ones(n_rho)
        diag_upper_rho = (1.0/drho**2 + 1.0/(2*self.rho*drho))[:-1]
        diag_lower_rho = (1.0/drho**2 - 1.0/(2*self.rho*drho))[1:]
        # Neumann BC at ρ=0
        diag_upper_rho[0] += (1.0/drho**2 - 1.0/(2*self.rho[0]*drho))
        
        L_rho_mat = diags([diag_lower_rho, diag_main_rho, diag_upper_rho], 
                          [-1, 0, 1], shape=(n_rho, n_rho), format='csr')
        
        # z-direction: -d²/dz²
        diag_main_z = -2.0 / dz**2 * np.ones(n_z)
        diag_off_z = 1.0 / dz**2 * np.ones(n_z - 1)
        L_z_mat = diags([diag_off_z, diag_main_z, diag_off_z], 
                        [-1, 0, 1], shape=(n_z, n_z), format='csr')
        
        I_rho = eye(n_rho, format='csr')
        I_z = eye(n_z, format='csr')
        Laplacian = kron(L_rho_mat, I_z) + kron(I_rho, L_z_mat)
        
        # Potential (use V_eff, not U, because we solve ω² eigenvalue)
        V_grid = self.geo.V_eff(self.RHO, self.Z)
        V_diag = V_grid.flatten()
        V_matrix = diags(V_diag, 0, shape=(N, N), format='csr')
        
        return -Laplacian + V_matrix
    
    def solve(self, n_eigenvalues=6):
        """
        Solve for eigenvalues ω².
        
        The continuum threshold is m₀², so bound states have ω² < m₀².
        """
        H = self.build_hamiltonian()
        
        # Shift-invert near the bottom of the potential
        V_min = self.geo.V_eff(self.RHO, self.Z).min()
        sigma = V_min - 0.1
        
        try:
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='LM', 
                                              sigma=sigma, tol=1e-8)
        except Exception as e:
            print(f"Warning: eigsh with sigma failed: {e}")
            eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='SM', tol=1e-8)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute E = ω² - m₀² (should be < 0 for bound states)
        E = eigenvalues - self.geo.m0**2
        
        # ω = sqrt(eigenvalue) for the actual frequency
        omega = np.sqrt(np.maximum(eigenvalues, 0.0))
        
        return {
            'omega_sq': eigenvalues,
            'omega': omega,
            'E': E,  # E < 0 means bound state
            'eigenvectors': eigenvectors
        }


class WKBCalculator:
    """
    WKB tunneling action using CORRECT threshold definition.
    
    S_N = ∫ dz √(U(z) - E_N)  where U = V_eff - m₀²
    
    Integration is over the classically forbidden region where U > E.
    """
    
    def __init__(self, geometry):
        self.geo = geometry
        
    def find_turning_points_1D(self, E, z_arr, rho_eval=0.1):
        """
        Find turning points where U(z) = E.
        
        E should be < 0 for bound states.
        """
        U_arr = self.geo.U_shifted(rho_eval * np.ones_like(z_arr), z_arr)
        
        diff = U_arr - E
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        turning_points = []
        for idx in sign_changes:
            # Linear interpolation
            z_cross = z_arr[idx] + (z_arr[idx+1] - z_arr[idx]) * \
                      (-diff[idx]) / (diff[idx+1] - diff[idx])
            turning_points.append(z_cross)
        
        return np.array(turning_points)
    
    def compute_S_N(self, E_N, z_range=(-60, 60), n_points=2000, rho_eval=0.1):
        """
        Compute WKB action for state with energy E_N.
        
        S_N = ∫_{forbidden} dz √(U - E_N)
        """
        z_arr = np.linspace(z_range[0], z_range[1], n_points)
        U_arr = self.geo.U_shifted(rho_eval * np.ones_like(z_arr), z_arr)
        
        turning_points = self.find_turning_points_1D(E_N, z_arr, rho_eval)
        
        if len(turning_points) < 2:
            return 0.0, turning_points
        
        # For double-well, identify the central barrier region
        # Sort turning points and find the barrier between wells
        tp_sorted = np.sort(turning_points)
        
        # Find middle two turning points (barrier)
        if len(tp_sorted) >= 4:
            # Two wells with central barrier: use middle two
            mid = len(tp_sorted) // 2
            z_left, z_right = tp_sorted[mid-1], tp_sorted[mid]
        elif len(tp_sorted) == 2:
            z_left, z_right = tp_sorted[0], tp_sorted[1]
        else:
            return 0.0, turning_points
        
        # Integrate √(U - E) over forbidden region
        mask = (z_arr >= z_left) & (z_arr <= z_right)
        z_forbidden = z_arr[mask]
        U_forbidden = U_arr[mask]
        
        integrand = np.sqrt(np.maximum(U_forbidden - E_N, 0.0))
        S = simpson(integrand, x=z_forbidden)
        
        return S, turning_points


def sanity_check_xi_conformal(geo):
    """
    Sanity check: at ξ = 1/6, derivative term must vanish.
    
    V_eff → m₀²Ω² only.
    """
    geo_conformal = TrueTwoCenterGeometry(
        D=geo.D, a=geo.a, epsilon=geo.epsilon, 
        m0=geo.m0, xi=XI_CONFORMAL
    )
    
    z_test = np.linspace(-20, 20, 100)
    rho_test = 0.1 * np.ones_like(z_test)
    
    comp = geo_conformal.V_eff_components(rho_test, z_test)
    
    # V_deriv should be zero
    max_deriv = np.max(np.abs(comp['V_deriv']))
    
    print("\nSanity Check: ξ = 1/6 (conformal coupling)")
    print(f"  Max |V_deriv| = {max_deriv:.2e} (should be ~0)")
    print(f"  Passed: {max_deriv < 1e-10}")
    
    return max_deriv < 1e-10


def sanity_check_asymptotic(geo, L=100):
    """
    Sanity check: U → 0 as r → ∞.
    """
    z_far = np.array([L])
    rho_far = np.array([0.1])
    
    U_far = geo.U_shifted(rho_far, z_far)[0]
    
    print(f"\nSanity Check: U(z={L}) = {U_far:.6e} (should be ~0)")
    print(f"  Passed: {abs(U_far) < 0.01}")
    
    return abs(U_far) < 0.01


def run_analysis(D, a=1.0, epsilon=0.3, m0=1.0, xi=0.0, output_dir=None):
    """Run complete analysis for a single D value."""
    
    print(f"\n{'='*60}")
    print(f"D = {D}, a = {a}, ε = {epsilon}, m₀ = {m0}, ξ = {xi}")
    print(f"{'='*60}")
    
    geo = TrueTwoCenterGeometry(D=D, a=a, epsilon=epsilon, m0=m0, xi=xi)
    
    # Sanity checks
    sanity_check_xi_conformal(geo)
    sanity_check_asymptotic(geo)
    
    # 1. Plot U(z) profile
    z = np.linspace(-40, 40, 500)
    rho = 0.1 * np.ones_like(z)
    
    comp = geo.V_eff_components(rho, z)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Components
    ax = axes[0, 0]
    ax.plot(z, comp['V_mass'] - m0**2, 'b-', linewidth=2, label=r'$m_0^2(\Omega^2 - 1)$')
    ax.plot(z, comp['V_deriv'], 'r-', linewidth=2, label=rf'$(1-6\xi)\Omega^{{-1}}\nabla^2\Omega$')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'Potential component', fontsize=12)
    ax.set_title(f'U(z) decomposition at D = {D}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Total U
    ax = axes[0, 1]
    ax.plot(z, comp['U'], 'k-', linewidth=2, label=r'$U = V_{\rm eff} - m_0^2$')
    ax.fill_between(z, comp['U'], where=comp['U'] < 0, alpha=0.3, color='blue', label='Bound region')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(-D/2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(D/2, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'$U(z)$', fontsize=12)
    ax.set_title(f'Total shifted potential (threshold = 0)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Solve eigenvalue problem
    print("\nSolving eigenvalue problem...")
    L_z = max(40, D + 20)
    solver = AxisymmetricSolver(geo, n_rho=50, n_z=100, L_rho=15, L_z=L_z)
    result = solver.solve(n_eigenvalues=6)
    
    print("\nResults:")
    print(f"  m₀² (threshold) = {m0**2:.4f}")
    print(f"  {'N':>3} {'ω²':>12} {'ω':>10} {'E':>12} {'Bound?':>8}")
    print(f"  {'-'*50}")
    for n in range(len(result['omega'])):
        omega_sq = result['omega_sq'][n]
        omega = result['omega'][n]
        E = result['E'][n]
        bound = "YES" if E < 0 else "NO"
        print(f"  {n+1:>3} {omega_sq:>12.6f} {omega:>10.6f} {E:>12.6f} {bound:>8}")
    
    # 3. WKB analysis
    print("\nWKB tunneling action...")
    wkb = WKBCalculator(geo)
    
    S_list = []
    for n, E in enumerate(result['E']):
        S, tp = wkb.compute_S_N(E)
        S_list.append(S)
        print(f"  N={n+1}: E = {E:.6f}, turning_points = {len(tp)}, S = {S:.6f}")
    
    # Bottom-left: Eigenvalue diagram
    ax = axes[1, 0]
    n_arr = np.arange(1, len(result['omega']) + 1)
    colors = ['green' if E < 0 else 'red' for E in result['E']]
    ax.bar(n_arr, result['E'], color=colors, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=2)
    ax.set_xlabel(r'$N$', fontsize=12)
    ax.set_ylabel(r'$E_N = \omega_N^2 - m_0^2$', fontsize=12)
    ax.set_title('Bound state energies (green = bound, red = continuum)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: S_N
    ax = axes[1, 1]
    ax.bar(n_arr, S_list, color='purple', alpha=0.7)
    ax.set_xlabel(r'$N$', fontsize=12)
    ax.set_ylabel(r'$S_N$ (WKB action)', fontsize=12)
    ax.set_title('Tunneling action', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f'true_veff_D{D:.0f}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {output_dir / f'true_veff_D{D:.0f}.png'}")
    else:
        plt.show()
    
    return {
        'D': D,
        'omega': result['omega'].tolist(),
        'omega_sq': result['omega_sq'].tolist(),
        'E': result['E'].tolist(),
        'S_N': S_list,
        'n_bound': sum(1 for E in result['E'] if E < 0)
    }


def main():
    """Run analysis for representative D value first."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir).parent / 'output' / 'true_veff'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRUE V_eff SOLVER")
    print("V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω")
    print("NO ENGINEERED TERMS")
    print("=" * 60)
    
    # Parameters
    a = 1.0
    epsilon = 0.3
    m0 = 1.0
    xi = 0.0  # Non-conformal for attractive core
    
    print(f"\nParameters: a={a}, ε={epsilon}, m₀={m0}, ξ={xi}")
    print(f"Conformal coupling: ξ_c = {XI_CONFORMAL:.4f}")
    print(f"Coefficient (1-6ξ) = {1-6*xi:.4f}")
    
    # Run for representative D
    D = 12.0
    result = run_analysis(D, a=a, epsilon=epsilon, m0=m0, xi=xi, output_dir=output_dir)
    
    # Save results
    with open(output_dir / 'true_veff_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nKEY FINDINGS:")
    print(f"  Number of bound states (E < 0): {result['n_bound']}")
    print(f"  Non-zero S_N values: {sum(1 for s in result['S_N'] if s > 0)}")
    
    if result['n_bound'] == 0:
        print("\n⚠️  NO BOUND STATES FOUND!")
        print("   This means the current parameters don't support bound layers.")
        print("   Possible fixes:")
        print("   1. Increase a (stronger conformal factor)")
        print("   2. Decrease ξ (more negative for stronger attraction)")
        print("   3. Check if the geometry supports binding at all")
    
    return result


if __name__ == '__main__':
    result = main()
