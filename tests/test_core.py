"""Tests for shared gene test implementation (_core.py)."""

import torch
import numpy as np
import pytest
from gpu_magma._core import pvalues_to_zscores, gene_test_single, pvalue_to_zscore


class TestPvaluesToZscores:
    """Tests for p-value to z-score conversion."""

    def test_known_values(self, device):
        """Standard p-values produce expected z-scores."""
        p = np.array([0.05, 0.01, 0.001])
        z = pvalues_to_zscores(p, device)

        assert abs(z[0].item() - 1.96) < 0.01  # p=0.05 -> z~1.96
        assert abs(z[1].item() - 2.576) < 0.01  # p=0.01 -> z~2.576
        assert abs(z[2].item() - 3.291) < 0.01  # p=0.001 -> z~3.291

    def test_extreme_small_pvalue(self, device):
        """Very small p-values are clipped, not infinite."""
        p = np.array([1e-300, 1e-100])
        z = pvalues_to_zscores(p, device)

        assert torch.all(torch.isfinite(z))
        assert torch.all(z <= 37.5)
        assert torch.all(z > 0)

    def test_pvalue_one(self, device):
        """p=1.0 produces z near 0."""
        p = np.array([1.0])
        z = pvalues_to_zscores(p, device)
        assert abs(z[0].item()) < 0.01

    def test_nan_guard(self, device):
        """NaN p-values are treated as no signal (p=1.0 -> z~0)."""
        p = np.array([0.05, float("nan"), 0.001])
        z = pvalues_to_zscores(p, device)

        assert torch.all(torch.isfinite(z))
        assert abs(z[0].item() - 1.96) < 0.01  # unaffected
        assert abs(z[1].item()) < 0.01  # NaN -> ~0
        assert abs(z[2].item() - 3.291) < 0.01  # unaffected

    def test_output_dtype_is_float32(self, device):
        """Output tensor is float32 regardless of input precision."""
        p = np.array([0.05], dtype=np.float64)
        z = pvalues_to_zscores(p, device)
        assert z.dtype == torch.float32


class TestGeneTestSingle:
    """Tests for the single-gene test function."""

    def test_k_zero(self, device):
        """Empty gene returns stat=0, p=1, neff=0."""
        stat, p, neff = gene_test_single(
            torch.tensor([], device=device),
            torch.empty(0, 0, device=device),
            1e-5,
            device,
        )
        assert stat == 0.0
        assert p == 1.0
        assert neff == 0

    def test_k_one(self, device):
        """Single-SNP gene returns chi2(1) test."""
        z = torch.tensor([1.96], device=device)
        stat, p, neff = gene_test_single(
            z, torch.ones(1, 1, device=device), 1e-5, device
        )
        assert abs(stat - 1.96**2) < 0.01
        assert abs(p - 0.05) < 0.005
        assert neff == 1

    def test_identity_ld(self, device):
        """With identity LD, test stat equals sum of z-squared."""
        z = torch.tensor([3.0, 2.0, 1.0], device=device)
        ld = torch.eye(3, device=device)
        stat, p, neff = gene_test_single(z, ld, 1e-5, device)

        expected_stat = 3**2 + 2**2 + 1**2  # 14
        assert abs(stat - expected_stat) < 0.1
        assert neff == 3
        assert p < 0.01  # chi2(3) at 14 is highly significant

    def test_correlated_snps_reduce_effective_df(self, device):
        """Correlated SNPs produce fewer effective eigenvalues than k."""
        k = 20
        # Create a rank-2 LD matrix (highly correlated)
        v1 = torch.randn(k, device=device)
        v2 = torch.randn(k, device=device)
        ld = 0.4 * torch.outer(v1, v1) + 0.4 * torch.outer(v2, v2)
        # Normalize to correlation matrix
        d = torch.sqrt(torch.diag(ld))
        ld = ld / torch.outer(d, d)
        ld.fill_diagonal_(1.0)

        z = torch.randn(k, device=device)
        stat, p, neff = gene_test_single(z, ld, 1e-5, device)

        # Effective rank should be much less than k for a near-rank-2 matrix
        assert neff < k

    def test_pvalue_range(self, device):
        """P-value is always in [0, 1]."""
        for _ in range(20):
            k = np.random.randint(2, 50)
            z = torch.randn(k, device=device) * 2
            ld = torch.eye(k, device=device)
            stat, p, neff = gene_test_single(z, ld, 1e-5, device)
            assert 0.0 <= p <= 1.0

    def test_stronger_signal_smaller_pvalue(self, device):
        """Stronger z-scores produce smaller p-values."""
        ld = torch.eye(5, device=device)

        z_weak = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        z_strong = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0], device=device)

        _, p_weak, _ = gene_test_single(z_weak, ld, 1e-5, device)
        _, p_strong, _ = gene_test_single(z_strong, ld, 1e-5, device)

        assert p_strong < p_weak


class TestPvalueToZscore:
    """Tests for gene-level p-value to z-score conversion."""

    def test_known_value(self, device):
        """p=0.05 -> z~1.645 (one-sided)."""
        z = pvalue_to_zscore(0.05, device)
        assert abs(z - 1.645) < 0.01

    def test_zero_pvalue(self, device):
        """p=0 returns clamped maximum z=37.5."""
        z = pvalue_to_zscore(0.0, device)
        assert z == 37.5

    def test_one_pvalue(self, device):
        """p=1.0 returns z=0."""
        z = pvalue_to_zscore(1.0, device)
        assert z == 0.0

    def test_very_small_pvalue(self, device):
        """Very small p-values produce large positive z."""
        z = pvalue_to_zscore(1e-10, device)
        assert z > 6.0
        assert z <= 37.5
