"""Shared test fixtures for GPU-MAGMA test suite."""

import pytest
import torch
import numpy as np
import pandas as pd
from gpu_magma import LDComputer, annotate_snps_to_genes, GeneAnnotation


@pytest.fixture
def device():
    """Use CPU for tests (no CUDA dependency in CI)."""
    return torch.device("cpu")


@pytest.fixture
def synthetic_ld(device):
    """LDComputer with small synthetic reference panel."""
    ld = LDComputer(device=str(device))
    ld.load_synthetic(n_samples=100, n_snps=2000)
    return ld


@pytest.fixture
def small_gwas():
    """Small GWAS dataset: 2000 SNPs on chr1, mostly null."""
    np.random.seed(42)
    n = 2000
    return {
        "chr": np.array(["1"] * n),
        "bp": np.arange(n) * 1000,
        "pvalues": np.random.uniform(0.01, 1.0, n),
    }


@pytest.fixture
def gene_locations():
    """Gene location DataFrame with 5 genes on chr1."""
    return pd.DataFrame({
        "gene_id": [f"GENE{i}" for i in range(1, 6)],
        "chr": ["1"] * 5,
        "start": [50_000, 200_000, 500_000, 800_000, 1_200_000],
        "end": [150_000, 350_000, 600_000, 950_000, 1_400_000],
        "gene_name": [f"GENE{i}" for i in range(1, 6)],
    })


@pytest.fixture
def annotations(small_gwas, gene_locations):
    """Gene annotations from small GWAS and gene locations."""
    return annotate_snps_to_genes(
        small_gwas["chr"],
        small_gwas["bp"],
        gene_locations,
        window=10_000,
    )
