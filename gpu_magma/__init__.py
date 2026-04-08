"""
GPU-MAGMA: GPU-accelerated gene-level analysis of GWAS summary statistics.

Reimplements MAGMA (de Leeuw et al., 2015 PLOS Comp Bio) using PyTorch
for massively parallel gene-level association testing on GPU.

Core algorithm per gene:
  1. Map SNPs to gene regions (positional window)
  2. Extract z-scores from GWAS p-values
  3. Compute LD correlation matrix from reference genotypes
  4. Eigendecompose LD matrix, threshold near-zero eigenvalues
  5. Project z-scores into eigenspace
  6. Compute F-statistic and gene-level p-value

GPU advantage: Steps 3-6 are batched across all ~20K genes simultaneously.
On A100 80GB, the entire human genome completes in seconds vs hours on CPU.
"""

__version__ = "0.1.0"

from .annotate import annotate_snps_to_genes
from .gene_test import GeneTestGPU
from .ld import LDComputer
