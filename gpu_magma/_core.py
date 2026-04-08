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
    """Gene-level test for a single gene (whitened chi-squared).

    Algorithm:
      1. Regularize: R_reg = R + threshold * I
      2. Eigendecompose: R_reg = U Lambda U^T
      3. Threshold: keep eigenvalues > threshold
      4. Project: W = Lambda^{-1/2} U^T Z
      5. Test stat: T = sum(W^2)
      6. Under null: T ~ chi2(m) where m = n_retained_eigenvalues
      7. P-value from chi-squared survival function (GPU-native)

    Returns:
        (test_statistic, p_value, n_effective_eigenvalues)
    """
    k = len(z_scores)

    if k == 0:
        return 0.0, 1.0, 0

    if k == 1:
        chi2_val = z_scores[0].item() ** 2
        # chi2 SF for df=1 on GPU
        dist = torch.distributions.Chi2(torch.tensor(1.0, device=device))
        p = 1.0 - dist.cdf(torch.tensor(chi2_val, device=device)).item()
        return chi2_val, p, 1

    # Regularize LD matrix
    ld_reg = ld_matrix + eigenvalue_threshold * torch.eye(
        k, device=device, dtype=ld_matrix.dtype
    )

    # Eigendecomposition on GPU
    eigenvalues, eigenvectors = torch.linalg.eigh(ld_reg)

    # Threshold eigenvalues
    mask = eigenvalues > eigenvalue_threshold
    n_effective = int(mask.sum().item())

    if n_effective == 0:
        return 0.0, 1.0, 0

    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]

    # Project z-scores into eigenspace
    projected = eigenvectors.T @ z_scores
    weighted = projected / torch.sqrt(eigenvalues)

    # Test statistic
    test_stat = (weighted ** 2).sum()

    # P-value from chi-squared (GPU-native, no scipy)
    dist = torch.distributions.Chi2(torch.tensor(float(n_effective), device=device))
    p_value = (1.0 - dist.cdf(test_stat)).item()

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
