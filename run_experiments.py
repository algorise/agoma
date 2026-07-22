#!/usr/bin/env python3
"""
Reproduction script for ICML 2026 paper #29461:
"Accelerated and Stable Convergence with Anchored Generalized Optimistic Method" (AGOMA / GOMA)
OpenReview ID: G6WKIN1heG, arXiv: 2606.21528

This script conducts rigorous numerical audits and empirical simulations for all 4 claims:
- Claim 1: Theorem 1 O(1/k^2) last-iterate accelerated convergence with fixed step size eta* in (0, 1/(2*sqrt(3)*L))
- Claim 2: Theorem 2 O(1/k^2) accelerated last-iterate rate with exploration step size gamma* in (0, (1/3L)*sqrt(5/26))
- Claim 3: Theorem 4 O(1/sqrt(k)) last-iterate rate under state-dependent stochastic noise with gamma_k=0
- Claim 4: Stochastic O(1/sqrt(k)) guarantee using linearly increasing minibatch sizes without variance reduction
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Directory setup
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper: Monotone Operators & Problems
# ---------------------------------------------------------
class SkewSymmetricGame:
    """
    Minimax bilinear game / skew-symmetric operator:
    f(x, y) = x^T B y + c^T x - d^T y
    G(z) = [B y + c; -B^T x + d] where z = [x; y]
    Operator matrix M = [0, B; -B^T, 0], which is skew-symmetric: z^T M z = 0.
    Monotone with Lipschitz constant L = ||B||_2.
    """
    def __init__(self, dim=10, seed=42):
        rng = np.random.RandomState(seed)
        B_raw = rng.randn(dim, dim)
        U, S, Vt = np.linalg.svd(B_raw)
        S = np.linspace(0.5, 2.5, dim) # condition number 5
        self.B = U @ np.diag(S) @ Vt
        self.dim = dim
        self.c = rng.randn(dim)
        self.d = rng.randn(dim)
        
        # Solution z* = [x*; y*] where G(z*) = 0
        # B y* = -c => y* = -B^{-1} c
        # -B^T x* = -d => x* = B^{-T} d
        self.y_star = np.linalg.solve(self.B, -self.c)
        self.x_star = np.linalg.solve(self.B.T, self.d)
        self.z_star = np.concatenate([self.x_star, self.y_star])
        
        # Lipschitz constant L = ||B||_2
        self.L = np.linalg.norm(self.B, ord=2)
        
    def operator(self, z):
        x, y = z[:self.dim], z[self.dim:]
        Gx = self.B @ y + self.c
        Gy = -self.B.T @ x + self.d
        return np.concatenate([Gx, Gy])

class NonLinearMonotoneVI:
    """
    Nonlinear Monotone VI: G(z) = M z + phi(z)
    where M is skew-symmetric + positive semi-definite, and phi(z)_i = arctan(z_i).
    Since phi'(z_i) = 1/(1+z_i^2) > 0, phi is strongly monotone, making G monotone and L-Lipschitz.
    """
    def __init__(self, dim=10, seed=42):
        rng = np.random.RandomState(seed)
        A = rng.randn(dim, dim)
        M = A - A.T + 0.1 * np.eye(dim) # skew-symmetric + positive diagonal
        self.M = M
        self.dim = dim
        self.z_star = np.zeros(dim) # G(0) = 0 since phi(0) = 0 and M 0 = 0
        self.L = np.linalg.norm(M, ord=2) + 1.0 # 1.0 is max derivative of arctan
        
    def operator(self, z):
        return self.M @ z + np.arctan(z)

# ---------------------------------------------------------
# GOMA Algorithm Implementations
# ---------------------------------------------------------
def run_goma(problem, z0, K=1000, eta=None, gamma=None, stochastic=False, sigma=0.1, kappa=0.5, batch_mode='linear'):
    """
    Anchored Generalized Optimistic Method (GOMA)
    z_{k+1} = beta_k * z0 + (1 - beta_k) * (z_k - eta_k * G_hat(z_bar_k))
    z_bar_k = z_k - gamma_k * G_hat(z_k)
    beta_k = 2 / (k + 6)
    """
    dim = len(z0)
    L = problem.L
    
    if eta is None:
        eta = 1.0 / (2.0 * np.sqrt(3.0) * L) * 0.99
    if gamma is None:
        gamma = eta
        
    z = z0.copy()
    z_history = [z.copy()]
    grad_norms = []
    
    for k in range(K):
        beta_k = 2.0 / (k + 6.0)
        
        # Primary operator evaluation
        G_zk = problem.operator(z)
        if stochastic:
            # Add state-dependent noise: E||G_hat||^2 <= sigma^2 + kappa ||G||^2
            if batch_mode == 'fixed':
                b_k = 1
            elif batch_mode == 'linear':
                b_k = k + 1
            else:
                b_k = 1
            
            noise_std = np.sqrt((sigma**2 + kappa * np.sum(G_zk**2)) / b_k)
            G_zk_noisy = G_zk + np.random.normal(0, noise_std, size=dim)
        else:
            G_zk_noisy = G_zk
            
        grad_norms.append(np.linalg.norm(G_zk))
        
        # Extrapolation step
        if gamma > 0:
            z_bar = z - gamma * G_zk_noisy
            G_zbar = problem.operator(z_bar)
            if stochastic:
                b_k = (k + 1) if batch_mode == 'linear' else 1
                noise_std_bar = np.sqrt((sigma**2 + kappa * np.sum(G_zbar**2)) / b_k)
                G_zbar_noisy = G_zbar + np.random.normal(0, noise_std_bar, size=dim)
            else:
                G_zbar_noisy = G_zbar
        else:
            G_zbar_noisy = G_zk_noisy
            
        # Update step with anchoring
        z_next = beta_k * z0 + (1.0 - beta_k) * (z - eta * G_zbar_noisy)
        z = z_next
        z_history.append(z.copy())
        
    grad_norms.append(np.linalg.norm(problem.operator(z)))
    return np.array(grad_norms), np.array(z_history)

# ---------------------------------------------------------
# Claim 1 Verification: Theorem 1 O(1/k^2) Rate with eta*
# ---------------------------------------------------------
def verify_claim1():
    print("--- Verifying Claim 1: Theorem 1 O(1/k^2) Last-Iterate Convergence ---")
    problem = SkewSymmetricGame(dim=10, seed=42)
    L = problem.L
    eta_star = 0.99 / (2.0 * np.sqrt(3.0) * L)
    
    rng = np.random.RandomState(123)
    z0 = problem.z_star + rng.randn(2 * problem.dim) * 2.0
    dist0_sq = np.sum((z0 - problem.z_star)**2)
    
    K = 1000
    grad_norms, _ = run_goma(problem, z0, K=K, eta=eta_star, gamma=eta_star, stochastic=False)
    
    k_vals = np.arange(K + 1)
    sq_grad_norms = grad_norms**2
    
    # Theorem 1 theoretical bound: ||G(x_k)||^2 <= 264 * L^2 / (k+6)^2 * ||x0 - x*||^2
    theoretical_bound = (264.0 * (L**2) * dist0_sq) / ((k_vals + 6.0)**2)
    
    # Check if empirical <= theoretical bound for all k
    bound_satisfied = np.all(sq_grad_norms <= theoretical_bound + 1e-12)
    
    # Estimate convergence slope alpha: log(||G||^2) = -alpha * log(k) + C
    # Fit on last 50% iterations
    half_K = K // 2
    log_k = np.log(k_vals[half_K:])
    log_grad_sq = np.log(sq_grad_norms[half_K:])
    slope, _ = np.polyfit(log_k, log_grad_sq, 1)
    empirical_rate = -slope
    
    print(f"  Lipschitz constant L: {L:.4f}")
    print(f"  Step size eta*: {eta_star:.6f} (Limit: {1.0/(2.0*np.sqrt(3.0)*L):.6f})")
    print(f"  Empirical O(1/k^alpha) decay rate alpha: {empirical_rate:.4f} (Expected: ~2.0)")
    print(f"  Theoretical upper bound satisfied at all iterations: {bound_satisfied}")
    print(f"  Final ||G(z_K)||^2: {sq_grad_norms[-1]:.4e} vs Bound: {theoretical_bound[-1]:.4e}")
    
    res = {
        "claim": 1,
        "verified": bool(bound_satisfied and empirical_rate >= 1.8),
        "empirical_decay_rate": float(empirical_rate),
        "bound_satisfied": bool(bound_satisfied),
        "final_grad_norm_sq": float(sq_grad_norms[-1]),
        "final_theoretical_bound": float(theoretical_bound[-1]),
        "L": float(L),
        "eta_star": float(eta_star)
    }
    
    # Save plot
    plt.figure(figsize=(7, 5))
    plt.semilogy(k_vals, sq_grad_norms, label="GOMA Empirical $\|G(z_k)\|^2$", color="#1f77b4", lw=2)
    plt.semilogy(k_vals, theoretical_bound, '--', label="Theorem 1 Bound $\\frac{264 L^2}{(k+6)^2} \|z_0-z^*\|^2$", color="#d62728", lw=1.8)
    plt.xlabel("Iteration $k$")
    plt.ylabel("Squared Operator Norm $\|G(z_k)\|^2$")
    plt.title("Claim 1: Theorem 1 Accelerated $O(1/k^2)$ Last-Iterate Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "claim1_theorem1_convergence.png"), dpi=200)
    plt.close()
    
    return res

# ---------------------------------------------------------
# Claim 2 Verification: Theorem 2 Rate with gamma*
# ---------------------------------------------------------
def verify_claim2():
    print("\n--- Verifying Claim 2: Theorem 2 Rate under Exploration Step Size gamma* ---")
    problem = NonLinearMonotoneVI(dim=10, seed=42)
    L = problem.L
    
    # Bound for gamma*: gamma* in (0, (1/3L)*sqrt(5/26))
    gamma_max = (1.0 / (3.0 * L)) * np.sqrt(5.0 / 26.0)
    gamma_star = 0.95 * gamma_max
    eta_star = 0.99 / (2.0 * np.sqrt(3.0) * L)
    
    rng = np.random.RandomState(456)
    z0 = problem.z_star + rng.randn(problem.dim) * 1.5
    dist0_sq = np.sum((z0 - problem.z_star)**2)
    
    K = 1000
    grad_norms, _ = run_goma(problem, z0, K=K, eta=eta_star, gamma=gamma_star, stochastic=False)
    
    k_vals = np.arange(K + 1)
    sq_grad_norms = grad_norms**2
    
    half_K = K // 2
    slope, _ = np.polyfit(np.log(k_vals[half_K:]), np.log(sq_grad_norms[half_K:]), 1)
    empirical_rate = -slope
    
    print(f"  Lipschitz constant L: {L:.4f}")
    print(f"  Exploration step size gamma*: {gamma_star:.6f} (Limit: {gamma_max:.6f})")
    print(f"  Empirical O(1/k^alpha) decay rate alpha: {empirical_rate:.4f} (Expected: ~2.0)")
    
    res = {
        "claim": 2,
        "verified": bool(empirical_rate >= 1.8),
        "empirical_decay_rate": float(empirical_rate),
        "gamma_star": float(gamma_star),
        "gamma_max": float(gamma_max),
        "final_grad_norm_sq": float(sq_grad_norms[-1])
    }
    
    plt.figure(figsize=(7, 5))
    plt.semilogy(k_vals, sq_grad_norms, label="GOMA Case II Empirical $\|G(z_k)\|^2$", color="#2ca02c", lw=2)
    plt.xlabel("Iteration $k$")
    plt.ylabel("Squared Operator Norm $\|G(z_k)\|^2$")
    plt.title("Claim 2: Theorem 2 Accelerated $O(1/k^2)$ Rate with Exploration $\\gamma^*$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "claim2_theorem2_gamma_star.png"), dpi=200)
    plt.close()
    
    return res

# ---------------------------------------------------------
# Claim 3 Verification: Theorem 4 Stochastic O(1/sqrt(k)) Rate
# ---------------------------------------------------------
def verify_claim3():
    print("\n--- Verifying Claim 3: Theorem 4 Stochastic O(1/sqrt(k)) Last-Iterate Rate ---")
    problem = SkewSymmetricGame(dim=10, seed=42)
    L = problem.L
    sigma = 0.1
    kappa = 0.5
    eta_stoch = 0.1 / (L * np.sqrt(1.0 + kappa))
    
    rng = np.random.RandomState(789)
    z0 = problem.z_star + rng.randn(2 * problem.dim) * 1.5
    dist0_sq = np.sum((z0 - problem.z_star)**2)
    
    # Run across 30 seeds to estimate E||G(z_N)||^2
    num_seeds = 30
    K = 500
    all_runs = []
    
    for s in range(num_seeds):
        np.random.seed(1000 + s)
        grad_norms, _ = run_goma(problem, z0, K=K, eta=eta_stoch, gamma=0.0, stochastic=True, sigma=sigma, kappa=kappa, batch_mode='linear')
        all_runs.append(grad_norms**2)
        
    mean_sq_grad = np.mean(all_runs, axis=0)
    k_vals = np.arange(K + 1)
    
    # Theorem 4 bound: E||G(z_N)||^2 <= 1570 * L^2 * kappa * ||z0 - z*||^2 / sqrt(N+1) + 8 * sigma^2 / (kappa * sqrt(N+1))
    theoretical_stoch_bound = (1570.0 * (L**2) * kappa * dist0_sq) / np.sqrt(k_vals + 1.0) + (8.0 * (sigma**2)) / (kappa * np.sqrt(k_vals + 1.0))
    
    # Fit decay rate alpha: E||G||^2 ~ k^{-alpha}
    half_K = K // 2
    slope, _ = np.polyfit(np.log(k_vals[half_K:]), np.log(mean_sq_grad[half_K:]), 1)
    empirical_rate = -slope
    
    print(f"  Stochastic parameters: sigma = {sigma}, kappa = {kappa}, eta = {eta_stoch:.6f}")
    print(f"  Empirical O(1/k^alpha) decay rate alpha: {empirical_rate:.4f} (Theoretical upper bound rate: O(1/sqrt(k)))")
    print(f"  Final E||G(z_K)||^2: {mean_sq_grad[-1]:.4e} vs Theoretical Bound: {theoretical_stoch_bound[-1]:.4e}")
    
    res = {
        "claim": 3,
        "verified": bool(empirical_rate >= 0.35 and mean_sq_grad[-1] <= theoretical_stoch_bound[-1]),
        "empirical_decay_rate": float(empirical_rate),
        "final_mean_sq_grad": float(mean_sq_grad[-1]),
        "final_theoretical_bound": float(theoretical_stoch_bound[-1]),
        "sigma": float(sigma),
        "kappa": float(kappa)
    }
    
    plt.figure(figsize=(7, 5))
    plt.loglog(k_vals + 1, mean_sq_grad, label="Stochastic GOMA Empirical $\\mathbb{E}\|G(z_k)\|^2$", color="#ff7f0e", lw=2)
    plt.loglog(k_vals + 1, theoretical_stoch_bound, '--', label="Theorem 4 Bound $O(1/\\sqrt{k})$", color="#d62728", lw=1.8)
    plt.xlabel("Iteration $k+1$")
    plt.ylabel("Expected Squared Operator Norm $\\mathbb{E}\|G(z_k)\|^2$")
    plt.title("Claim 3: Theorem 4 Stochastic $O(1/\\sqrt{k})$ Last-Iterate Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "claim3_theorem4_stochastic.png"), dpi=200)
    plt.close()
    
    return res

# ---------------------------------------------------------
# Claim 4 Verification: Linearly Increasing Minibatches
# ---------------------------------------------------------
def verify_claim4():
    print("\n--- Verifying Claim 4: Linearly Increasing Minibatch Sizes vs Fixed Batch ---")
    problem = SkewSymmetricGame(dim=10, seed=42)
    L = problem.L
    sigma = 0.1
    kappa = 0.5
    eta_stoch = 0.1 / (L * np.sqrt(1.0 + kappa))
    
    rng = np.random.RandomState(789)
    z0 = problem.z_star + rng.randn(2 * problem.dim) * 1.5
    
    num_seeds = 25
    K = 400
    
    linear_runs = []
    fixed_runs = []
    
    for s in range(num_seeds):
        np.random.seed(2000 + s)
        g_lin, _ = run_goma(problem, z0, K=K, eta=eta_stoch, gamma=0.0, stochastic=True, sigma=sigma, kappa=kappa, batch_mode='linear')
        g_fix, _ = run_goma(problem, z0, K=K, eta=eta_stoch, gamma=0.0, stochastic=True, sigma=sigma, kappa=kappa, batch_mode='fixed')
        linear_runs.append(g_lin**2)
        fixed_runs.append(g_fix**2)
        
    mean_linear = np.mean(linear_runs, axis=0)
    mean_fixed = np.mean(fixed_runs, axis=0)
    k_vals = np.arange(K + 1)
    
    # Linearly increasing minibatches achieves O(1/sqrt(k)) decay without saturation,
    # whereas fixed minibatch size saturates at a variance floor O(sigma^2 / b_0).
    final_ratio = mean_fixed[-1] / mean_linear[-1]
    
    print(f"  Final E||G||^2 with Linearly Increasing Batch: {mean_linear[-1]:.6e}")
    print(f"  Final E||G||^2 with Fixed Batch: {mean_fixed[-1]:.6e}")
    print(f"  Performance Advantage Ratio (Fixed / Linear): {final_ratio:.2f}x lower noise floor")
    
    res = {
        "claim": 4,
        "verified": bool(final_ratio > 3.0),
        "final_linear_sq_grad": float(mean_linear[-1]),
        "final_fixed_sq_grad": float(mean_fixed[-1]),
        "advantage_ratio": float(final_ratio)
    }
    
    plt.figure(figsize=(7, 5))
    plt.loglog(k_vals + 1, mean_linear, label="Linearly Increasing Minibatch ($b_k = k$)", color="#1f77b4", lw=2)
    plt.loglog(k_vals + 1, mean_fixed, label="Fixed Minibatch ($b_k = 1$)", color="#9467bd", lw=2, linestyle="--")
    plt.xlabel("Iteration $k+1$")
    plt.ylabel("Expected Squared Operator Norm $\\mathbb{E}\|G(z_k)\|^2$")
    plt.title("Claim 4: Benefit of Linearly Increasing Minibatches without VR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "claim4_linearly_increasing_minibatches.png"), dpi=200)
    plt.close()
    
    return res

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("Running Reproduction Suite for ICML 2026 Paper #29461 (AGOMA)")
    print("=" * 60)
    
    start_time = time.time()
    
    res1 = verify_claim1()
    res2 = verify_claim2()
    res3 = verify_claim3()
    res4 = verify_claim4()
    
    elapsed = time.time() - start_time
    
    summary = {
        "paper_title": "Accelerated and Stable Convergence with Anchored Generalized Optimistic Method",
        "openreview_id": "G6WKIN1heG",
        "arxiv_id": "2606.21528",
        "execution_time_sec": elapsed,
        "claims_summary": {
            "claim_1": res1,
            "claim_2": res2,
            "claim_3": res3,
            "claim_4": res4
        },
        "all_claims_verified": res1["verified"] and res2["verified"] and res3["verified"] and res4["verified"]
    }
    
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"REPRODUCTION SUMMARY (Elapsed Time: {elapsed:.2f}s)")
    print("=" * 60)
    print(f"  Claim 1 Verified: {res1['verified']} (Decay rate: k^-{res1['empirical_decay_rate']:.2f})")
    print(f"  Claim 2 Verified: {res2['verified']} (Decay rate: k^-{res2['empirical_decay_rate']:.2f})")
    print(f"  Claim 3 Verified: {res3['verified']} (Stochastic decay rate: k^-{res3['empirical_decay_rate']:.2f})")
    print(f"  Claim 4 Verified: {res4['verified']} (Linearly increasing minibatch benefit: {res4['advantage_ratio']:.1f}x)")
    print(f"  ALL CLAIMS VERIFIED: {summary['all_claims_verified']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
