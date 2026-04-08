"""
Shared gene-level test implementation.

Extracted to avoid duplication between GeneTestGPU and RealLDGeneTest.
Uses PyTorch-native distributions to avoid scipy CPU round-trips in the hot path.
"""

import torch
import numpy as np


def pvalues_to_zscores(
    pvalues: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert GWAS p-values to z-scores on GPU.

    Uses the two-sided inverse normal transform: z = Phi^{-1}(1 - p/2).
    NaN p-values are replaced with 1.0 (no signal) before conversion.
    """
    p = pvalues.astype(np.float64, copy=True)

    # Guard against NaN — treat as no signal
    nan_mask = np.isnan(p)
    if nan_mask.any():
        p[nan_mask] = 1.0

    p = np.clip(p, 1e-300, 1.0 - 1e-10)

    # Use torch for the inverse normal CDF on GPU
    p_tensor = torch.tensor(p, dtype=torch.float64, device=device)
    # z = Phi^{-1}(1 - p/2) via erfinv: Phi^{-1}(x) = sqrt(2) * erfinv(2x - 1)
    z = torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device)) * torch.erfinv(
        2.0 * (1.0 - p_tensor / 2.0) - 1.0
    )
    z = z.clamp(-37.5, 37.5).float()
    return z


def gene_test_single(
    z_scores: torch.Tensor,
    ld_matrix: torch.Tensor,
    eigenvalue_threshold: float,
    device: torch.device,
) -> tuple[float, float, int]:
    """Gene-level test for a single gene (Brown's method / MAGMA mean model).

    Works with chi-squared statistics (z^2) directly, adjusting for LD via
    the Frobenius norm of the correlation matrix. This correctly handles
    two-sided (unsigned) z-scores from GWAS p-values.

    Algorithm:
      1. T = sum(z_i^2)             — sum of chi-squared(1) statistics
      2. E[T] = k                   — expected value under null
      3. Var[T] = 2 * trace(R^2)    — LD-adjusted variance (Brown 1975)
      4. Satterthwaite approximation: T ~ c * chi2(m) where
           c = trace(R^2) / k
           m = k^2 / trace(R^2)      — effective number of independent SNPs
      5. p = P(chi2(m) > T/c)

    Returns:
        (test_statistic, p_value, n_effective)
    """
    k = len(z_scores)

    if k == 0:
        return 0.0, 1.0, 0

    if k == 1:
        chi2_val = z_scores[0].item() ** 2
        dist = torch.distributions.Chi2(torch.tensor(1.0, device=device))
        p = 1.0 - dist.cdf(torch.tensor(chi2_val, device=device)).item()
        return chi2_val, p, 1

    # Test statistic: sum of squared z-scores
    test_stat = (z_scores ** 2).sum()

    # LD-adjusted variance via Frobenius norm: trace(R^2) = ||R||_F^2
    trace_R2 = (ld_matrix ** 2).sum()

    # Satterthwaite approximation parameters
    c = trace_R2 / k                     # scaling factor
    n_effective = int(round((k * k / trace_R2).item()))  # effective df
    n_effective = max(n_effective, 1)

    # P-value: P(chi2(m) > T/c)
    m = torch.tensor(float(n_effective), device=device)
    dist = torch.distributions.Chi2(m)
    p_value = (1.0 - dist.cdf(test_stat / c)).item()

    return float(test_stat.item()), p_value, n_effective


def pvalue_to_zscore(p_value: float, device: torch.device) -> float:
    """Convert a gene-level p-value to a z-score (probit). GPU-native."""
    if p_value <= 0:
        return 37.5
    if p_value >= 1.0:
        return 0.0
    # z = Phi^{-1}(1 - p) = sqrt(2) * erfinv(2*(1-p) - 1) = sqrt(2) * erfinv(1 - 2p)
    p_t = torch.tensor(p_value, dtype=torch.float64, device=device)
    z = torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device)) * torch.erfinv(
        1.0 - 2.0 * p_t
    )
    return float(z.clamp(-37.5, 37.5).item())
