"""
GPU-MAGMA: GPU-accelerated gene-level analysis of GWAS summary statistics.

Inspired by MAGMA (de Leeuw et al., 2015 PLOS Comp Bio), reimplemented
in PyTorch for gene-level association testing on GPU.

Core algorithm per gene:
  1. Map SNPs to gene regions (positional window)
  2. Extract z-scores from GWAS p-values
  3. Compute LD correlation matrix from reference genotypes
  4. Eigendecompose LD matrix, threshold near-zero eigenvalues
  5. Project z-scores into eigenspace (whitened decorrelation)
  6. Compute chi-squared test statistic and gene-level p-value

GPU advantage: Steps 3-6 leverage GPU-accelerated linear algebra
(eigendecomposition, matrix multiply) for each gene.

Note: GPU-MAGMA uses a whitened chi-squared test (decorrelation + unweighted
sum), which differs from MAGMA v1.08+'s weighted chi-squared with Imhof's
method. Both are valid approaches; see README for details.
"""

__version__ = "0.1.0"

from .annotate import annotate_snps_to_genes, GeneAnnotation, load_gene_locations
from .gene_test import GeneTestGPU, GeneResult
from .ld import LDComputer
from .ld_real import PerChromLDComputer, RealLDGeneTest
from .ld_decay import LDDecayApprox
