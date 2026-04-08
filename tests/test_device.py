"""Device compatibility tests.

Ensures the full pipeline works on CPU (no CUDA required).
"""

import torch
import numpy as np
import pandas as pd
import pytest
from gpu_magma import (
    annotate_snps_to_genes,
    GeneTestGPU,
    LDComputer,
    LDDecayApprox,
)


class TestCPUDevice:
    """Full pipeline tests running on CPU."""

    def test_full_pipeline_cpu(self):
        """End-to-end pipeline works with device='cpu'."""
        ld = LDComputer(device="cpu")
        ld.load_synthetic(n_samples=50, n_snps=500)

        snp_chr = np.array(["1"] * 500)
        snp_bp = np.arange(500) * 1000

        genes = pd.DataFrame({
            "gene_id": ["G1", "G2"],
            "chr": ["1", "1"],
            "start": [50_000, 200_000],
            "end": [150_000, 350_000],
            "gene_name": ["G1", "G2"],
        })

        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=10_000)
        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 500)

        tester = GeneTestGPU(ld_computer=ld, device="cpu")
        result = tester.run(pvals, annots, verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result["p"].between(0, 1))

    def test_ld_decay_approx_cpu(self):
        """LDDecayApprox works on CPU."""
        positions = np.arange(100) * 10_000
        chromosomes = np.ones(100, dtype=int)
        ld = LDDecayApprox(
            snp_positions=positions,
            snp_chromosomes=chromosomes,
            decay_rate=50_000,
            device="cpu",
        )

        R = ld.compute_ld_matrix(np.arange(20))
        assert R.device == torch.device("cpu")
        assert R.shape == (20, 20)
        assert torch.allclose(torch.diag(R), torch.ones(20))

    def test_ld_decay_with_gene_test_cpu(self):
        """LDDecayApprox plugs into GeneTestGPU via duck typing on CPU."""
        n = 200
        positions = np.arange(n) * 5_000
        chromosomes = np.ones(n, dtype=int)
        ld = LDDecayApprox(positions, chromosomes, decay_rate=50_000, device="cpu")

        snp_chr = np.array(["1"] * n)
        snp_bp = positions
        genes = pd.DataFrame({
            "gene_id": ["G1"],
            "chr": ["1"],
            "start": [100_000],
            "end": [500_000],
            "gene_name": ["G1"],
        })

        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=10_000)
        pvals = np.random.RandomState(42).uniform(0.01, 1.0, n)

        tester = GeneTestGPU(ld_computer=ld, device="cpu")
        result = tester.run(pvals, annots, verbose=False)

        assert len(result) == 1
        assert 0 <= result.iloc[0]["p"] <= 1
