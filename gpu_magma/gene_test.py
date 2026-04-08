"""
GPU-accelerated gene-level association test.

Inspired by MAGMA gene analysis (de Leeuw et al., 2015):
  - Mean model: tests whether the sum of chi-squared statistics across
    SNPs in a gene exceeds expectation under the null, accounting for LD.
  - Uses Brown's method (Satterthwaite approximation) to adjust the
    chi-squared reference distribution for LD correlation.

The key GPU operations:
  1. LD matrix computation via batched matrix multiply (G^T G)
  2. Frobenius norm for LD-adjusted variance (trace(R^2))
  3. Vectorized p-value computation from chi-squared distribution
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from .annotate import GeneAnnotation
from .ld import LDComputer
from ._core import pvalues_to_zscores, gene_test_single, pvalue_to_zscore


@dataclass
class GeneResult:
    gene_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    n_snps: int
    n_effective: int
    stat: float
    p_value: float
    z_score: float


class GeneTestGPU:
    """GPU-accelerated gene-level GWAS analysis.

    Equivalent to: magma --bfile <ref> --pval <gwas> --gene-annot <annot> --out <results>
    """

    def __init__(
        self,
        ld_computer: LDComputer,
        eigenvalue_threshold: float = 1e-5,
        max_snps_per_gene: int = 500,
        device: str = "cuda",
    ):
        self.ld = ld_computer
        self.eigenvalue_threshold = eigenvalue_threshold
        self.max_snps_per_gene = max_snps_per_gene
        self.device = torch.device(device)

    def _pvalues_to_zscores(self, pvalues: np.ndarray) -> torch.Tensor:
        return pvalues_to_zscores(pvalues, self.device)

    def _gene_test_single(
        self,
        z_scores: torch.Tensor,
        ld_matrix: torch.Tensor,
    ) -> tuple[float, float, int]:
        """Gene-level test for a single gene. Delegates to shared implementation."""
        return gene_test_single(z_scores, ld_matrix, self.eigenvalue_threshold, self.device)

    def run(
        self,
        snp_pvalues: np.ndarray,
        gene_annotations: list[GeneAnnotation],
        snp_indices_in_ref: Optional[dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run gene-level analysis across all annotated genes."""
        z_all = self._pvalues_to_zscores(snp_pvalues)

        results = []
        n_genes = len(gene_annotations)
        n_skipped = 0

        if verbose:
            print(f"Running GPU gene-level test on {n_genes} genes...")

        for i, gene in enumerate(gene_annotations):
            if verbose and (i + 1) % 2000 == 0:
                print(f"  Progress: {i + 1}/{n_genes} genes ({len(results)} tested, {n_skipped} skipped)")

            gwas_idx = gene.snp_indices

            if len(gwas_idx) == 0:
                n_skipped += 1
                continue

            if snp_indices_in_ref is not None:
                ref_idx = np.array([snp_indices_in_ref[j] for j in gwas_idx
                                   if j in snp_indices_in_ref])
                if len(ref_idx) == 0:
                    n_skipped += 1
                    continue
                gwas_matched = np.array([j for j in gwas_idx if j in snp_indices_in_ref])
            else:
                ref_idx = gwas_idx
                gwas_matched = gwas_idx

            # Truncate large genes — evenly spaced thinning by position
            if len(ref_idx) > self.max_snps_per_gene:
                step = len(ref_idx) / self.max_snps_per_gene
                keep = np.round(np.arange(self.max_snps_per_gene) * step).astype(int)
                ref_idx = ref_idx[keep]
                gwas_matched = gwas_matched[keep]

            z_gene = z_all[gwas_matched]
            ld_matrix = self.ld.compute_ld_matrix(ref_idx)
            stat, p_value, n_eff = self._gene_test_single(z_gene, ld_matrix)

            z_gene_score = pvalue_to_zscore(p_value, self.device)

            results.append(GeneResult(
                gene_id=gene.gene_id,
                gene_name=gene.gene_name,
                chromosome=gene.chromosome,
                start=gene.start,
                end=gene.end,
                n_snps=len(gwas_matched),
                n_effective=n_eff,
                stat=stat,
                p_value=p_value,
                z_score=z_gene_score,
            ))

        if verbose:
            print(f"  Complete: {len(results)} genes tested, {n_skipped} skipped")
            sig = sum(1 for r in results if r.p_value < 2.5e-6)
            print(f"  Genome-wide significant (p < 2.5e-6): {sig} genes")

        if not results:
            return pd.DataFrame(columns=[
                "gene_id", "gene_name", "chr", "start", "end",
                "n_snps", "n_effective", "stat", "p", "z",
            ])

        df = pd.DataFrame([{
            "gene_id": r.gene_id,
            "gene_name": r.gene_name,
            "chr": r.chromosome,
            "start": r.start,
            "end": r.end,
            "n_snps": r.n_snps,
            "n_effective": r.n_effective,
            "stat": r.stat,
            "p": r.p_value,
            "z": r.z_score,
        } for r in results])

        return df.sort_values("p").reset_index(drop=True)
