"""Tests for LD matrix computation."""

import torch
import numpy as np
import pytest
from gpu_magma.ld import LDComputer


class TestLDComputerSynthetic:
    """Tests for LDComputer with synthetic data."""

    def test_load_synthetic_shape(self, synthetic_ld):
        """Synthetic genotypes have the correct shape."""
        assert synthetic_ld.genotypes is not None
        assert synthetic_ld.genotypes.shape == (100, 2000)

    def test_load_synthetic_standardized(self, synthetic_ld):
        """Synthetic genotypes are approximately zero-mean, unit-variance."""
        g = synthetic_ld.genotypes
        means = g.mean(dim=0)
        stds = g.std(dim=0)

        assert torch.allclose(means, torch.zeros_like(means), atol=0.1)
        assert torch.allclose(stds, torch.ones_like(stds), atol=0.2)

    def test_load_synthetic_metadata(self, synthetic_ld):
        """Synthetic data populates SNP metadata arrays."""
        assert synthetic_ld.snp_ids is not None
        assert len(synthetic_ld.snp_ids) == 2000
        assert synthetic_ld.snp_chr is not None
        assert synthetic_ld.snp_bp is not None


class TestComputeLDMatrix:
    """Tests for LD matrix computation."""

    def test_diagonal_is_one(self, synthetic_ld):
        """LD matrix diagonal is exactly 1.0."""
        idx = np.arange(10)
        R = synthetic_ld.compute_ld_matrix(idx)
        diag = torch.diag(R)
        assert torch.allclose(diag, torch.ones_like(diag))

    def test_symmetry(self, synthetic_ld):
        """LD matrix is symmetric."""
        idx = np.arange(20)
        R = synthetic_ld.compute_ld_matrix(idx)
        assert torch.allclose(R, R.T, atol=1e-6)

    def test_values_in_range(self, synthetic_ld):
        """LD correlation values are in [-1, 1]."""
        idx = np.arange(50)
        R = synthetic_ld.compute_ld_matrix(idx)
        assert R.min() >= -1.01  # small tolerance for float32
        assert R.max() <= 1.01

    def test_single_snp(self, synthetic_ld):
        """Single SNP produces 1x1 identity matrix."""
        R = synthetic_ld.compute_ld_matrix(np.array([0]))
        assert R.shape == (1, 1)
        assert R.item() == 1.0

    def test_no_data_raises(self, device):
        """Calling compute without loading data raises RuntimeError."""
        ld = LDComputer(device=str(device))
        with pytest.raises(RuntimeError, match="No reference data"):
            ld.compute_ld_matrix(np.array([0, 1, 2]))

    def test_block_ld_structure(self, synthetic_ld):
        """SNPs within the same block are more correlated than across blocks.

        Synthetic data uses block_size=50, so SNPs 0-49 share a latent factor
        and should have higher LD than SNPs 0 vs 50+.
        """
        within_block = np.arange(10)  # all in block 0
        across_blocks = np.array([0, 1, 50, 51])  # block 0 and block 1

        R_within = synthetic_ld.compute_ld_matrix(within_block)
        R_across = synthetic_ld.compute_ld_matrix(across_blocks)

        # Mean off-diagonal correlation should be higher within block
        mean_within = (R_within.sum() - R_within.trace()) / (10 * 9)
        mean_across = (R_across.sum() - R_across.trace()) / (4 * 3)

        assert abs(mean_within.item()) > abs(mean_across.item())


class TestComputeLDMatricesBatched:
    """Tests for batched LD matrix computation."""

    def test_returns_correct_count(self, synthetic_ld):
        """Returns one matrix per gene."""
        gene_indices = [np.arange(10), np.arange(20, 30), np.arange(40, 60)]
        results = synthetic_ld.compute_ld_matrices_batched(gene_indices)
        assert len(results) == 3

    def test_truncation(self, synthetic_ld):
        """Large genes are truncated to max_snps_per_gene."""
        gene_indices = [np.arange(200)]  # 200 SNPs
        results = synthetic_ld.compute_ld_matrices_batched(gene_indices, max_snps_per_gene=50)
        assert results[0].shape == (50, 50)
