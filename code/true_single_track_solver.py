#!/usr/bin/env python3
"""
CORRECT Single-Track V_eff Solver

This is the FINAL CORRECT implementation:
    V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω
    
    NO engineered terms (barrier, box, etc.)
    U = V_eff - m₀² has threshold at 0
    Bound states have E < 0

Author: PSLT Collaboration
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
# True Action-Derived Potential
# ============================================================

def Omega_func(rho, z, D, a, eps):
    """Conformal factor: Ω = 1 + a/r₊ + a/r₋"""
    r1 = np.sqrt(rho**2 + (z - D/2)**2 + eps**2)
    r2 = np.sqrt(rho**2 + (z + D/2)**2 + eps**2)
    return 1.0 + a*(1.0/r1 + 1.0/r2)

def laplacian_Omega_func(rho, z, D, a, eps):
    """∇²Ω for smeared sources: ∇²(1/√(r²+ε²)) = -3ε²/(r²+ε²)^(5/2)"""
    r1 = np.sqrt(rho**2 + (z - D/2)**2 + eps**2)
    r2 = np.sqrt(rho**2 + (z + D/2)**2 + eps**2)
    lap1 = -3*eps**2 / (r1**5)
    lap2 = -3*eps**2 / (r2**5)
    return a*(lap1 + lap2)

def U_func(rho, z, D, a, eps, m0, xi):
    """
    Shifted potential: U = V_eff - m₀² → 0 as r → ∞
    
    V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω  (from action)
    U = m₀²(Ω² - 1) + (1-6ξ)Ω⁻¹∇²Ω
    
    Bound states have E < 0.
    """
    Om = Omega_func(rho, z, D, a, eps)
    lapOm = laplacian_Omega_func(rho, z, D, a, eps)
    return m0**2*(Om**2 - 1.0) + (1.0 - 6.0*xi)*(lapOm/Om)


class TrueSolver:
    """2D axisymmetric solver with correct V_eff."""
    
    def __init__(self, D, a=1.0, eps=0.2, m0=1.0, xi=0.0,
                 n_rho=50, n_z=500, L_rho=4.0, L_z=18.0):
        self.D = D
        self.a = a
        self.eps = eps
        self.m0 = m0
        self.xi = xi
        
        self.n_rho = n_rho
        self.n_z = n_z
        self.L_rho = L_rho
        self.L_z = L_z
        
        # Grid (start rho from small value to avoid axis singularity)
        self.rho_min = 0.02
        self.drho = (L_rho - self.rho_min) / (n_rho - 1)
        self.dz = 2 * L_z / (n_z + 1)
        
        self.rho = np.linspace(self.rho_min, L_rho, n_rho)
        self.z = np.linspace(-L_z + self.dz, L_z - self.dz, n_z)
        
        self.RHO, self.Z = np.meshgrid(self.rho, self.z, indexing='ij')
        self.U_grid = U_func(self.RHO, self.Z, D, a, eps, m0, xi)
        
    def build_hamiltonian(self):
        """Build H = -∇² + U in cylindrical coordinates."""
        n_rho, n_z = self.n_rho, self.n_z
        drho, dz = self.drho, self.dz
        N = n_rho * n_z
        
        # ρ-direction with proper axis handling
        diag_main_rho = -2.0 / drho**2 * np.ones(n_rho)
        diag_upper_rho = np.array([1.0/drho**2 + 1.0/(2*self.rho[i]*drho) 
                                   for i in range(n_rho-1)])
        diag_lower_rho = np.array([1.0/drho**2 - 1.0/(2*self.rho[i+1]*drho) 
                                   for i in range(n_rho-1)])
        diag_upper_rho[0] += 1.0/drho**2 - 1.0/(2*self.rho[0]*drho)  # Neumann
        
        L_rho_mat = diags([diag_lower_rho, diag_main_rho, diag_upper_rho], 
                          [-1, 0, 1], shape=(n_rho, n_rho), format='csr')
        
        # z-direction
        diag_main_z = -2.0 / dz**2 * np.ones(n_z)
        diag_off_z = 1.0 / dz**2 * np.ones(n_z - 1)
        L_z_mat = diags([diag_off_z, diag_main_z, diag_off_z], 
                        [-1, 0, 1], shape=(n_z, n_z), format='csr')
        
        I_rho = eye(n_rho, format='csr')
        I_z = eye(n_z, format='csr')
        Laplacian = kron(L_rho_mat, I_z) + kron(I_rho, L_z_mat)
        
        U_diag = self.U_grid.flatten()
        U_matrix = diags(U_diag, 0, shape=(N, N), format='csr')
        
        return -Laplacian + U_matrix
    
    def solve(self, n_eigenvalues=6):
        """Solve for eigenvalues. Use which='SM' for states near threshold."""
        H = self.build_hamiltonian()
        eigenvalues, eigenvectors = eigsh(H, k=n_eigenvalues, which='SM', tol=1e-10)
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx]
    
    def compute_wkb_action(self, E, z_range=None, n_points=3000):
        """Compute WKB tunneling action for central barrier."""
        if z_range is None:
            z_range = (-self.L_z, self.L_z)
        
        z_arr = np.linspace(z_range[0], z_range[1], n_points)
        rho_axis = self.rho_min
        U_arr = np.array([U_func(rho_axis, z, self.D, self.a, self.eps, 
                                 self.m0, self.xi) for z in z_arr])
        
        # Find turning points
        diff = U_arr - E
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        if len(sign_changes) < 4:
            return 0.0, len(sign_changes)
        
        # Double-well: barrier between inner two turning points
        i1, i2 = sign_changes[1], sign_changes[2]
        z_left, z_right = z_arr[i1], z_arr[i2]
        
        mask = (z_arr >= z_left) & (z_arr <= z_right)
        integrand = np.sqrt(np.maximum(U_arr[mask] - E, 0))
        S = simpson(integrand, x=z_arr[mask])
        
        return S, len(sign_changes)


def run_D_scan(D_values, a=1.0, eps=0.2, m0=1.0, xi=0.0, output_dir=None):
    """Run analysis for multiple D values."""
    
    results = {
        'D': [],
        'E_bound': [],
        'omega': [],
        'S_N': [],
        'n_turning': [],
        'n_bound': [],
        'params': {'a': a, 'eps': eps, 'm0': m0, 'xi': xi}
    }
    
    print("D-scan with TRUE V_eff")
    print("="*60)
    print(f"V_eff = m₀²Ω² + (1-6ξ)Ω⁻¹∇²Ω")
    print(f"Parameters: a={a}, ε={eps}, m₀={m0}, ξ={xi}")
    print("="*60)
    
    for D in D_values:
        print(f"\nD = {D:.1f}...")
        
        # Adjust grid for D
        L_z = max(18, D + 8)
        solver = TrueSolver(D, a=a, eps=eps, m0=m0, xi=xi,
                           n_rho=50, n_z=500, L_rho=4.0, L_z=L_z)
        
        # Solve eigenvalues
        eigenvalues = solver.solve(n_eigenvalues=6)
        
        # Count bound states
        bound_E = [E for E in eigenvalues if E < 0]
        n_bound = len(bound_E)
        
        results['D'].append(D)
        results['n_bound'].append(n_bound)
        
        if n_bound > 0:
            E1 = bound_E[0]
            omega1 = np.sqrt(E1 + m0**2)
            S, n_tp = solver.compute_wkb_action(E1)
            
            results['E_bound'].append(E1)
            results['omega'].append(omega1)
            results['S_N'].append(S)
            results['n_turning'].append(n_tp)
            
            print(f"  Bound: E₁={E1:.4f}, ω₁={omega1:.4f}, S₁={S:.4f}, tp={n_tp}")
        else:
            results['E_bound'].append(None)
            results['omega'].append(None)
            results['S_N'].append(None)
            results['n_turning'].append(None)
            print(f"  No bound states")
    
    return results


def plot_results(results, output_dir):
    """Generate publication-quality figures."""
    
    D_arr = np.array(results['D'])
    
    # Filter to only D values with bound states
    mask = np.array([E is not None for E in results['E_bound']])
    D_bound = D_arr[mask]
    E_bound = np.array([E for E in results['E_bound'] if E is not None])
    omega = np.array([w for w in results['omega'] if w is not None])
    S_N = np.array([S for S in results['S_N'] if S is not None])
    
    if len(D_bound) == 0:
        print("No bound states found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: E vs D
    ax = axes[0, 0]
    ax.plot(D_bound, E_bound, 'bo-', markersize=8, linewidth=2)
    ax.axhline(0, color='k', linestyle='--', label='Threshold')
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$E_1 = \omega_1^2 - m_0^2$', fontsize=14)
    ax.set_title('Bound State Energy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: ω vs D
    ax = axes[0, 1]
    ax.plot(D_bound, omega, 'ro-', markersize=8, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$\omega_1$', fontsize=14)
    ax.set_title('Layer Frequency', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: S_N vs D
    ax = axes[1, 0]
    ax.plot(D_bound, S_N, 'go-', markersize=8, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$S_1$ (WKB action)', fontsize=14)
    ax.set_title('Tunneling Action (from SAME V_eff)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: r_N = exp(-2S) vs D
    ax = axes[1, 1]
    r_N = np.exp(-2*S_N)
    ax.semilogy(D_bound, r_N, 'mo-', markersize=8, linewidth=2)
    ax.set_xlabel(r'$D$', fontsize=14)
    ax.set_ylabel(r'$r_1 = e^{-2S_1}$', fontsize=14)
    ax.set_title('Tunneling Probability', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(r'TRUE Single-Track Results: $V_{\rm eff} = m_0^2\Omega^2 + (1-6\xi)\Omega^{-1}\nabla^2\Omega$', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'true_single_track.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'true_single_track.png'}")


def plot_Veff_profile(D, output_dir, a=1.0, eps=0.2, m0=1.0, xi=0.0):
    """Plot U(z) profile showing the double-well structure."""
    
    z = np.linspace(-20, 20, 2000)
    rho = 0.02 * np.ones_like(z)
    
    U = np.array([U_func(0.02, zz, D, a, eps, m0, xi) for zz in z])
    V_mass = np.array([m0**2*(Omega_func(0.02, zz, D, a, eps)**2 - 1) for zz in z])
    V_deriv = U - V_mass
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top: Components
    ax1.plot(z, V_mass, 'b-', linewidth=2, label=r'$m_0^2(\Omega^2 - 1)$ (mass)')
    ax1.plot(z, V_deriv, 'r-', linewidth=2, label=r'$(1-6\xi)\Omega^{-1}\nabla^2\Omega$ (curvature)')
    ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax1.axvline(-D/2, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(D/2, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Potential', fontsize=12)
    ax1.set_title(f'Action-Derived $V_{{eff}}$ Decomposition at D = {D}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-30, 15)
    
    # Bottom: Total U
    ax2.plot(z, U, 'k-', linewidth=2, label=r'$U = V_{\rm eff} - m_0^2$')
    ax2.fill_between(z, U, where=U<0, alpha=0.3, color='blue', label='Bound region')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax2.axvline(-D/2, color='gray', linestyle='--', alpha=0.5, label='Centers')
    ax2.axvline(D/2, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$z$', fontsize=12)
    ax2.set_ylabel(r'$U(z)$ (threshold = 0)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-30, 15)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'Veff_true_D{D:.0f}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'Veff_true_D{D:.0f}.png'}")


def main():
    """Run complete single-track analysis."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir).parent / 'output' / 'true_single_track'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    a = 1.0
    eps = 0.2
    m0 = 1.0
    xi = 0.0
    
    # D values
    D_values = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    
    # Plot V_eff profile
    plot_Veff_profile(12.0, output_dir, a, eps, m0, xi)
    
    # Run D scan
    results = run_D_scan(D_values, a, eps, m0, xi, output_dir)
    
    # Plot results
    plot_results(results, output_dir)
    
    # Save JSON
    # Convert None to null for JSON
    results_json = {k: [None if v is None else v for v in vals] 
                    if isinstance(vals, list) else vals 
                    for k, vals in results.items()}
    with open(output_dir / 'true_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*60)
    print("SINGLE-TRACK ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")
    
    return results


if __name__ == '__main__':
    results = main()
