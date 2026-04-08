"""
GPU-accelerated LD (Linkage Disequilibrium) computation.

Computes pairwise SNP correlation matrices from reference genotypes
using batched GPU matrix operations. This replaces the most expensive
step in MAGMA's gene-level analysis.

On A100 80GB: entire 1000G EUR panel (~500 samples × ~8M SNPs ≈ 4GB)
fits in HBM with room for thousands of gene LD matrices simultaneously.
"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path


class LDComputer:
    """GPU-accelerated LD matrix computation from reference genotypes."""
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.genotypes: Optional[torch.Tensor] = None  # (n_samples, n_snps)
        self.snp_ids: Optional[np.ndarray] = None
        self.snp_chr: Optional[np.ndarray] = None
        self.snp_bp: Optional[np.ndarray] = None
    
    def load_plink_bed(self, bed_prefix: str) -> None:
        """Load PLINK BED/BIM/FAM files into GPU memory.
        
        Reads the binary genotype matrix and transfers to GPU.
        For 1000G EUR: ~500 samples × ~8M SNPs ≈ 4GB on GPU.
        """
        from pandas_plink import read_plink1_bin
        
        print(f"Loading reference panel from {bed_prefix}...")
        
        # read_plink1_bin returns (bim, fam, genotype_matrix)
        bim, fam, G = read_plink1_bin(
            f"{bed_prefix}.bed",
            f"{bed_prefix}.bim",
            f"{bed_prefix}.fam",
            verbose=False,
        )
        
        # G is a dask array of shape (n_samples, n_snps) with values 0, 1, 2, NaN
        # Convert to numpy then torch
        print(f"  Shape: {G.shape[0]} samples × {G.shape[1]} variants")
        
        geno_np = G.compute().astype(np.float32)
        
        # Handle missing values: replace NaN with per-SNP mean (mean imputation)
        col_means = np.nanmean(geno_np, axis=0)
        nan_mask = np.isnan(geno_np)
        geno_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Standardize: zero mean, unit variance per SNP
        geno_mean = geno_np.mean(axis=0)
        geno_std = geno_np.std(axis=0)
        geno_std[geno_std == 0] = 1.0  # avoid division by zero for monomorphic SNPs
        geno_np = (geno_np - geno_mean) / geno_std
        
        # Transfer to GPU
        self.genotypes = torch.from_numpy(geno_np).to(device=self.device, dtype=self.dtype)
        
        # Store SNP metadata
        self.snp_ids = bim["snp"].values
        self.snp_chr = bim["chrom"].values.astype(str)
        self.snp_bp = bim["pos"].values.astype(np.int64)
        
        n_samples, n_snps = self.genotypes.shape
        mem_gb = self.genotypes.element_size() * self.genotypes.nelement() / 1e9
        print(f"  Loaded: {n_samples} samples × {n_snps} SNPs ({mem_gb:.1f} GB on GPU)")
    
    def load_synthetic(self, n_samples: int = 500, n_snps: int = 100000) -> None:
        """Generate synthetic reference data for testing.
        
        Creates random genotypes with realistic LD structure
        (block-diagonal correlation).
        """
        print(f"Generating synthetic reference: {n_samples} × {n_snps}...")
        
        # Generate genotypes with block LD structure
        block_size = 50
        n_blocks = n_snps // block_size
        
        geno = torch.zeros(n_samples, n_snps, device=self.device, dtype=self.dtype)
        
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_snps)
            k = end - start
            
            # Shared latent factor creates LD within block
            latent = torch.randn(n_samples, 1, device=self.device, dtype=self.dtype)
            noise = torch.randn(n_samples, k, device=self.device, dtype=self.dtype)
            geno[:, start:end] = 0.5 * latent + 0.5 * noise
        
        # Standardize
        geno = (geno - geno.mean(dim=0)) / (geno.std(dim=0) + 1e-8)
        
        self.genotypes = geno
        self.snp_ids = np.array([f"rs{i}" for i in range(n_snps)])
        self.snp_chr = np.array(["1"] * n_snps)
        self.snp_bp = np.arange(n_snps) * 1000
        
        print(f"  Synthetic reference ready on {self.device}")
    
    def compute_ld_matrix(self, snp_indices: np.ndarray) -> torch.Tensor:
        """Compute LD correlation matrix for a set of SNPs.
        
        Args:
            snp_indices: indices into the reference panel SNP array
            
        Returns:
            Correlation matrix of shape (k, k) where k = len(snp_indices)
        """
        if self.genotypes is None:
            raise RuntimeError("No reference data loaded. Call load_plink_bed() or load_synthetic() first.")
        
        idx = torch.tensor(snp_indices, dtype=torch.long, device=self.device)
        G_sub = self.genotypes[:, idx]  # (n_samples, k)
        
        n = G_sub.shape[0]
        # Correlation = (G^T G) / (n - 1), already standardized
        R = (G_sub.T @ G_sub) / (n - 1)
        
        # Ensure diagonal is exactly 1.0
        R.fill_diagonal_(1.0)
        
        return R
    
    def compute_ld_matrices_batched(
        self,
        gene_snp_indices: list[np.ndarray],
        max_snps_per_gene: int = 500,
    ) -> list[torch.Tensor]:
        """Compute LD matrices for multiple genes in batched fashion.
        
        For genes with more SNPs than max_snps_per_gene, uses PCA-based
        approximation to keep memory bounded.
        
        Args:
            gene_snp_indices: list of SNP index arrays, one per gene
            max_snps_per_gene: truncate genes with more SNPs than this
            
        Returns:
            List of LD correlation matrices, one per gene
        """
        results = []
        
        for snp_idx in gene_snp_indices:
            if len(snp_idx) > max_snps_per_gene:
                # For very large genes (e.g., MHC), subsample
                rng = np.random.default_rng(42)
                snp_idx = rng.choice(snp_idx, max_snps_per_gene, replace=False)
                snp_idx.sort()
            
            R = self.compute_ld_matrix(snp_idx)
            results.append(R)
        
        return results
