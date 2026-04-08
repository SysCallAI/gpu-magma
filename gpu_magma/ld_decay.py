"""
Distance-based LD decay approximation.

When a reference genotype panel isn't available, approximate the LD
correlation matrix using exponential distance decay:

  R(i,j) = exp(-|bp_i - bp_j| / decay_rate)

For EUR populations, LD decays to ~0 within 200-500 kb.
Default decay rate: 50 kb (r^2 ~ 0.135 at 100 kb, ~0.018 at 200 kb).

This is a common approximation used by:
- SuSiE (fine-mapping with approximate LD)
- PAINTOR (causal variant prioritization)
- Various post-GWAS summary stat methods

It gives much better calibration than identity LD (no correlation),
though not as accurate as a real reference panel.
"""

import torch
import numpy as np
from typing import Optional


class LDDecayApprox:
    """Approximate LD matrices using exponential distance decay."""

    def __init__(
        self,
        snp_positions: np.ndarray,
        snp_chromosomes: np.ndarray,
        decay_rate: float = 50_000,  # 50 kb default for EUR
        min_corr: float = 0.01,      # zero out correlations below this
        device: str = "cuda",
    ):
        self.positions = snp_positions
        self.chromosomes = snp_chromosomes
        self.decay_rate = decay_rate
        self.min_corr = min_corr
        self.device = torch.device(device)
        self.n_snps = len(snp_positions)

    def compute_ld_matrix(self, snp_indices: np.ndarray) -> torch.Tensor:
        """Compute approximate LD matrix for a set of SNPs.

        Returns a correlation matrix where R(i,j) = exp(-dist/decay_rate)
        for SNPs on the same chromosome, and R(i,j) = 0 across chromosomes.
        """
        k = len(snp_indices)

        if k == 0:
            return torch.empty(0, 0, device=self.device)
        if k == 1:
            return torch.ones(1, 1, device=self.device)

        # Get positions and chromosomes for these SNPs
        pos = torch.tensor(
            self.positions[snp_indices], dtype=torch.float32, device=self.device
        )
        chrs = self.chromosomes[snp_indices]

        # Compute pairwise distances on GPU
        dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))

        # Exponential decay
        R = torch.exp(-dist / self.decay_rate)

        # Zero out cross-chromosome correlations (should be rare within a gene, but just in case)
        same_chr = torch.tensor(
            np.equal.outer(chrs, chrs), dtype=torch.bool, device=self.device
        )
        R = R * same_chr.float()

        # Threshold small correlations
        R = R * (R >= self.min_corr).float()

        # Ensure diagonal is exactly 1.0
        R.fill_diagonal_(1.0)

        return R


if __name__ == "__main__":
    # Quick test
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Simulate 500 SNPs spanning 1 Mb on chr1
    n = 500
    positions = np.sort(np.random.randint(1_000_000, 2_000_000, n))
    chromosomes = np.ones(n, dtype=int)

    ld = LDDecayApprox(positions, chromosomes, decay_rate=50_000, device=device)

    t0 = time.time()
    R = ld.compute_ld_matrix(np.arange(n))
    elapsed = time.time() - t0

    print(f"LD matrix ({n}x{n}): {elapsed*1000:.1f}ms")
    print(f"Mean off-diagonal corr: {(R.sum() - R.trace()).item() / (n*(n-1)):.4f}")
    print(f"Sparsity (< 0.01): {(R < 0.01).float().mean().item():.1%}")
    print(f"Max off-diagonal: {(R - torch.eye(n, device=R.device)).max().item():.4f}")

    # Check eigenvalues are positive (PSD)
    eigenvalues = torch.linalg.eigvalsh(R)
    print(f"Min eigenvalue: {eigenvalues.min().item():.6f} (should be >= 0)")
    print(f"Effective rank: {(eigenvalues > 0.01).sum().item()}/{n}")
