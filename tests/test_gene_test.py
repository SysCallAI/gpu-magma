"""Tests for gene-level test pipeline (GeneTestGPU)."""

import torch
import numpy as np
import pandas as pd
import pytest
from gpu_magma import GeneTestGPU, LDComputer, annotate_snps_to_genes


class TestGeneTestGPUPipeline:
    """Integration tests for the full gene test pipeline."""

    def test_run_returns_dataframe(self, synthetic_ld, annotations):
        """run() returns a DataFrame with expected columns."""
        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 2000)
        tester = GeneTestGPU(ld_computer=synthetic_ld, device="cpu")
        result = tester.run(pvals, annotations, verbose=False)

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"gene_id", "gene_name", "chr", "start", "end",
                         "n_snps", "n_effective", "stat", "p", "z"}
        assert set(result.columns) == expected_cols

    def test_results_sorted_by_pvalue(self, synthetic_ld, annotations):
        """Results are sorted by p-value ascending."""
        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 2000)
        tester = GeneTestGPU(ld_computer=synthetic_ld, device="cpu")
        result = tester.run(pvals, annotations, verbose=False)

        if len(result) > 1:
            assert (result["p"].diff().dropna() >= 0).all()

    def test_signal_gene_has_small_pvalue(self, synthetic_ld, small_gwas, gene_locations):
        """Injecting strong signal in one gene makes it highly significant."""
        pvals = small_gwas["pvalues"].copy()
        # GENE3 spans [500k, 600k] = SNP indices 500-600
        pvals[500:610] = np.random.RandomState(99).uniform(1e-8, 1e-5, 110)

        annots = annotate_snps_to_genes(
            small_gwas["chr"], small_gwas["bp"], gene_locations, window=10_000
        )
        tester = GeneTestGPU(ld_computer=synthetic_ld, device="cpu")
        result = tester.run(pvals, annots, verbose=False)

        assert len(result) > 0
        # GENE3 should be highly significant (synthetic LD may cause other
        # genes to also be significant due to shared latent factors, so we
        # just check GENE3 has a very small p-value)
        gene3 = result[result["gene_name"] == "GENE3"]
        assert len(gene3) == 1
        assert gene3.iloc[0]["p"] < 0.001

    def test_verbose_false_suppresses_output(self, synthetic_ld, annotations, capsys):
        """verbose=False produces no stdout output."""
        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 2000)
        tester = GeneTestGPU(ld_computer=synthetic_ld, device="cpu")
        tester.run(pvals, annotations, verbose=False)

        captured = capsys.readouterr()
        # Only the synthetic load message should appear (from fixture),
        # not any gene test progress
        assert "Progress:" not in captured.out
        assert "Complete:" not in captured.out

    def test_skipped_genes_not_in_output(self, device):
        """Genes with zero SNPs produce no annotations, so empty results."""
        ld = LDComputer(device=str(device))
        ld.load_synthetic(n_samples=50, n_snps=100)

        # All SNPs on chr1, gene on chr2 -> no overlap at annotation stage
        snp_chr = np.array(["1"] * 100)
        snp_bp = np.arange(100) * 1000
        genes = pd.DataFrame({
            "gene_id": ["ORPHAN"],
            "chr": ["2"],
            "start": [0],
            "end": [100000],
            "gene_name": ["ORPHAN"],
        })
        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=10_000)

        # No genes annotated -> empty annotation list
        assert len(annots) == 0

        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 100)
        tester = GeneTestGPU(ld_computer=ld, device=str(device))
        result = tester.run(pvals, annots, verbose=False)

        # Empty DataFrame (no 'p' column to sort on) — handle gracefully
        assert len(result) == 0

    def test_max_snps_per_gene_truncation(self, device):
        """Genes with >max_snps_per_gene SNPs are truncated."""
        ld = LDComputer(device=str(device))
        ld.load_synthetic(n_samples=50, n_snps=500)

        snp_chr = np.array(["1"] * 500)
        snp_bp = np.arange(500) * 100
        genes = pd.DataFrame({
            "gene_id": ["BIG_GENE"],
            "chr": ["1"],
            "start": [0],
            "end": [50000],
            "gene_name": ["BIG_GENE"],
        })
        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=0)

        pvals = np.random.RandomState(42).uniform(0.01, 1.0, 500)
        tester = GeneTestGPU(ld_computer=ld, device=str(device), max_snps_per_gene=50)
        result = tester.run(pvals, annots, verbose=False)

        assert len(result) == 1
        assert result.iloc[0]["n_snps"] == 50


class TestNullDistribution:
    """Statistical tests for calibration under the null hypothesis."""

    def test_null_pvalues_not_inflated(self, device):
        """Under null, gene p-values should NOT be systematically small.

        Brown's method with high synthetic LD (n_eff ~4) produces slightly
        conservative p-values (median ~0.45), which is acceptable. The key
        invariant is: no false positive inflation under the null.
        """
        np.random.seed(42)
        ld = LDComputer(device=str(device))
        ld.load_synthetic(n_samples=200, n_snps=5000)

        snp_chr = np.array(["1"] * 5000)
        snp_bp = np.arange(5000) * 1000
        genes = pd.DataFrame({
            "gene_id": [f"G{i}" for i in range(50)],
            "chr": ["1"] * 50,
            "start": [i * 100_000 for i in range(50)],
            "end": [i * 100_000 + 50_000 for i in range(50)],
            "gene_name": [f"G{i}" for i in range(50)],
        })

        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=0)
        null_pvals = np.random.uniform(0.01, 1.0, 5000)

        tester = GeneTestGPU(ld_computer=ld, device=str(device))
        result = tester.run(null_pvals, annots, verbose=False)

        if len(result) >= 10:
            # Under null: false positive rate at p<0.05 should be <= 20%
            # (allowing conservative behavior but catching inflation)
            fpr = (result["p"] < 0.05).mean()
            assert fpr <= 0.20, (
                f"False positive inflation: {fpr:.0%} of null genes at p<0.05"
            )

    def test_truncation_does_not_inflate_null(self, device):
        """Evenly-spaced truncation should not inflate p-values under the null.

        Regression test: p-value-based truncation caused 98% of null genes
        to appear significant because it cherry-picked extreme values.
        """
        from scipy import stats

        np.random.seed(7)
        n_snps = 10_000
        ld = LDComputer(device=str(device))
        ld.load_synthetic(n_samples=100, n_snps=n_snps)

        snp_chr = np.array(["1"] * n_snps)
        snp_bp = np.arange(n_snps) * 100

        # 20 large genes with ~500 SNPs each (will hit max_snps_per_gene=100)
        genes = pd.DataFrame({
            "gene_id": [f"G{i}" for i in range(20)],
            "chr": ["1"] * 20,
            "start": [i * 50_000 for i in range(20)],
            "end": [i * 50_000 + 49_900 for i in range(20)],
            "gene_name": [f"G{i}" for i in range(20)],
        })

        annots = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=0)
        null_pvals = np.random.uniform(0.01, 1.0, n_snps)

        tester = GeneTestGPU(
            ld_computer=ld, device=str(device), max_snps_per_gene=100
        )
        result = tester.run(null_pvals, annots, verbose=False)

        # Under the null, fewer than 50% of genes should be "significant"
        # at p < 0.05. With the old p-value selection bug, ~100% were.
        sig_rate = (result["p"] < 0.05).mean()
        assert sig_rate < 0.5, (
            f"Truncation inflates null: {sig_rate:.0%} significant at p<0.05 "
            f"(expected <50%)"
        )
