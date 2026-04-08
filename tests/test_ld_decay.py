"""Tests for LD decay approximation."""

import torch
import numpy as np
import pytest
from gpu_magma.ld_decay import LDDecayApprox


@pytest.fixture
def decay_ld(device):
    """LDDecayApprox with 100 SNPs on chr1."""
    positions = np.arange(100) * 10_000  # 0 to 990kb, spaced 10kb
    chromosomes = np.ones(100, dtype=int)
    return LDDecayApprox(
        snp_positions=positions,
        snp_chromosomes=chromosomes,
        decay_rate=50_000,
        min_corr=0.01,
        device=str(device),
    )


class TestLDDecayApprox:
    """Tests for exponential decay LD approximation."""

    def test_diagonal_is_one(self, decay_ld):
        """Diagonal elements are exactly 1.0."""
        R = decay_ld.compute_ld_matrix(np.arange(20))
        diag = torch.diag(R)
        assert torch.allclose(diag, torch.ones_like(diag))

    def test_symmetry(self, decay_ld):
        """Matrix is symmetric."""
        R = decay_ld.compute_ld_matrix(np.arange(20))
        assert torch.allclose(R, R.T, atol=1e-6)

    def test_known_decay_at_rate(self, decay_ld):
        """At distance == decay_rate, correlation should be ~1/e = 0.368."""
        # SNP 0 is at 0bp, SNP 5 is at 50,000bp = decay_rate
        R = decay_ld.compute_ld_matrix(np.array([0, 5]))
        expected = np.exp(-1)  # ~0.368
        assert abs(R[0, 1].item() - expected) < 0.01

    def test_min_corr_threshold(self, device):
        """Correlations below min_corr are zeroed out."""
        positions = np.array([0, 1_000_000])  # 1Mb apart
        chromosomes = np.array([1, 1])
        ld = LDDecayApprox(positions, chromosomes, decay_rate=50_000, min_corr=0.01, device=str(device))

        R = ld.compute_ld_matrix(np.array([0, 1]))
        # exp(-1e6/5e4) = exp(-20) ~ 2e-9, well below min_corr
        assert R[0, 1].item() == 0.0

    def test_cross_chromosome_zero(self, device):
        """SNPs on different chromosomes have zero correlation."""
        positions = np.array([100_000, 100_000])  # same position
        chromosomes = np.array([1, 2])  # different chromosomes
        ld = LDDecayApprox(positions, chromosomes, decay_rate=50_000, device=str(device))

        R = ld.compute_ld_matrix(np.array([0, 1]))
        assert R[0, 1].item() == 0.0
        assert R[1, 0].item() == 0.0

    def test_positive_semi_definite(self, decay_ld):
        """Matrix eigenvalues are all non-negative (PSD)."""
        R = decay_ld.compute_ld_matrix(np.arange(50))
        eigenvalues = torch.linalg.eigvalsh(R)
        assert eigenvalues.min().item() >= -1e-6  # small tolerance for float32

    def test_single_snp(self, decay_ld):
        """Single SNP returns 1x1 identity."""
        R = decay_ld.compute_ld_matrix(np.array([0]))
        assert R.shape == (1, 1)
        assert R.item() == 1.0

    def test_empty_returns_empty(self, decay_ld):
        """Empty input returns 0x0 matrix."""
        R = decay_ld.compute_ld_matrix(np.array([], dtype=int))
        assert R.shape == (0, 0)
