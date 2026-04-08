"""
Per-chromosome real LD computation from 1000G EUR reference panel.

Loads BED/BIM/FAM files per chromosome on-demand. Uses position-based
matching (chr:bp) to map GWAS SNPs to reference panel SNPs, since SNP
IDs may differ between datasets.

Memory strategy: only one chromosome's genotypes in GPU memory at a time.
For 1000G EUR: ~6M SNPs (chr1, largest) x 503 samples x 4 bytes = ~12 GB.
Fits in A100 80GB with room for gene LD matrices.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


class PerChromLDComputer:
    """Real LD computation from per-chromosome PLINK BED files."""

    def __init__(
        self,
        bed_dir: str,
        bed_prefix: str = "1kg_eur_chr",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.bed_dir = Path(bed_dir)
        self.bed_prefix = bed_prefix
        self.device = torch.device(device)
        self.dtype = dtype

        # Per-chromosome reference data (lazy loaded)
        self._loaded_chr: Optional[int] = None
        self._geno: Optional[torch.Tensor] = None

        # Position index: chr -> {bp -> ref_index_within_chr}
        self._chr_bp_to_idx: dict[int, dict[int, int]] = {}
        self._chr_n_snps: dict[int, int] = {}

        self._build_index()

    def _build_index(self):
        """Read all BIM files to build position-based lookup."""
        total_snps = 0
        for chrom in range(1, 23):
            bim_path = self.bed_dir / f"{self.bed_prefix}{chrom}.bim"
            if not bim_path.exists():
                continue

            bp_to_idx = {}
            n_lines = 0
            with open(bim_path) as f:
                for i, line in enumerate(f):
                    fields = line.strip().split("\t")
                    bp = int(fields[3])
                    if bp not in bp_to_idx:
                        bp_to_idx[bp] = i
                    n_lines = i + 1

            self._chr_bp_to_idx[chrom] = bp_to_idx
            self._chr_n_snps[chrom] = len(bp_to_idx)
            total_snps += n_lines

        n_chr = len(self._chr_bp_to_idx)
        print(f"PerChromLDComputer: indexed {n_chr} chromosomes, ~{total_snps:,} total SNPs")
        for c in sorted(self._chr_bp_to_idx.keys()):
            print(f"  chr{c}: {self._chr_n_snps[c]:,} indexed positions")

    def _load_chromosome(self, chrom: int):
        """Load a chromosome's genotypes into GPU memory."""
        if self._loaded_chr == chrom:
            return

        bed_path = self.bed_dir / f"{self.bed_prefix}{chrom}"

        # Free previous chromosome
        if self._geno is not None:
            del self._geno
            torch.cuda.empty_cache()

        from pandas_plink import read_plink1_bin

        # pandas_plink 2.x returns a single xarray DataArray
        G = read_plink1_bin(
            f"{bed_path}.bed",
            f"{bed_path}.bim",
            f"{bed_path}.fam",
            verbose=False,
        )

        n_samples = G.shape[0]
        n_snps = G.shape[1]

        geno_np = G.values.astype(np.float32)

        # Mean imputation for missing values
        col_means = np.nanmean(geno_np, axis=0)
        nan_mask = np.isnan(geno_np)
        if nan_mask.any():
            geno_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Standardize per SNP
        geno_mean = geno_np.mean(axis=0)
        geno_std = geno_np.std(axis=0)
        geno_std[geno_std == 0] = 1.0
        geno_np = (geno_np - geno_mean) / geno_std

        self._geno = torch.from_numpy(geno_np).to(device=self.device, dtype=self.dtype)
        self._loaded_chr = chrom

        mem_gb = self._geno.element_size() * self._geno.nelement() / 1e9
        print(f"  Loaded chr{chrom}: {n_samples} samples x {n_snps:,} SNPs ({mem_gb:.1f} GB GPU)")

    def match_gwas_to_ref(
        self,
        gwas_chr: np.ndarray,
        gwas_bp: np.ndarray,
    ) -> dict[int, int]:
        """Match GWAS SNPs to reference panel by position.

        Returns mapping: gwas_index -> ref_index_within_chromosome.
        """
        mapping = {}
        n_matched = 0
        n_total = len(gwas_chr)

        for i in range(n_total):
            c = int(gwas_chr[i])
            bp = int(gwas_bp[i])

            if c in self._chr_bp_to_idx:
                ref_idx = self._chr_bp_to_idx[c].get(bp)
                if ref_idx is not None:
                    mapping[i] = ref_idx
                    n_matched += 1

        match_rate = n_matched / n_total * 100 if n_total > 0 else 0
        print(f"SNP matching: {n_matched:,}/{n_total:,} ({match_rate:.1f}%) GWAS SNPs in reference")
        return mapping

    def compute_ld_matrix(self, ref_indices: np.ndarray, chrom: int = None) -> torch.Tensor:
        """Compute real LD correlation matrix for reference panel SNPs."""
        k = len(ref_indices)

        if k == 0:
            return torch.empty(0, 0, device=self.device)
        if k == 1:
            return torch.ones(1, 1, device=self.device)

        if chrom is not None:
            self._load_chromosome(chrom)

        if self._geno is None:
            raise RuntimeError("No chromosome loaded. Provide chrom parameter.")

        idx = torch.tensor(ref_indices, dtype=torch.long, device=self.device)
        G_sub = self._geno[:, idx]

        n = G_sub.shape[0]
        R = (G_sub.T @ G_sub) / (n - 1)
        R.fill_diagonal_(1.0)

        return R


class RealLDGeneTest:
    """Run gene-level test with per-chromosome real LD.

    Handles chromosome-aware loading and SNP matching.
    """

    def __init__(
        self,
        ld_computer: PerChromLDComputer,
        device: str = "cuda",
        max_snps_per_gene: int = 500,
        eigenvalue_threshold: float = 0.01,
    ):
        self.ld = ld_computer
        self.device = torch.device(device)
        self.max_snps = max_snps_per_gene
        self.eig_threshold = eigenvalue_threshold

    def _gene_test_single(self, z_scores, ld_matrix):
        """Same algorithm as GeneTestGPU._gene_test_single."""
        from scipy import stats as sp_stats

        k = len(z_scores)
        if k == 0:
            return 0.0, 1.0, 0
        if k == 1:
            chi2 = z_scores[0].item() ** 2
            p = float(sp_stats.chi2.sf(chi2, df=1))
            return chi2, p, 1

        ld_reg = ld_matrix + self.eig_threshold * torch.eye(
            k, device=self.device, dtype=ld_matrix.dtype
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(ld_reg)
        mask = eigenvalues > self.eig_threshold
        n_effective = int(mask.sum().item())

        if n_effective == 0:
            return 0.0, 1.0, 0

        eigenvalues = eigenvalues[mask]
        eigenvectors = eigenvectors[:, mask]

        projected = eigenvectors.T @ z_scores
        weighted = projected / torch.sqrt(eigenvalues)
        test_stat = float((weighted ** 2).sum().item())

        p_value = float(sp_stats.chi2.sf(test_stat, df=n_effective))
        return test_stat, p_value, n_effective

    def run(
        self,
        snp_pvalues: np.ndarray,
        gene_annotations: list,
        gwas_chr: np.ndarray,
        gwas_bp: np.ndarray,
        verbose: bool = True,
    ):
        """Run gene-level test with real LD from per-chromosome reference."""
        import pandas as pd
        from scipy import stats as sp_stats

        # Step 1: Match GWAS SNPs to reference
        gwas_to_ref = self.ld.match_gwas_to_ref(gwas_chr, gwas_bp)

        # Step 2: Convert p-values to z-scores
        p = np.clip(snp_pvalues, 1e-300, 1.0 - 1e-10)
        z_np = sp_stats.norm.ppf(1 - p / 2)
        z_np = np.clip(z_np, -37.5, 37.5)
        z_all = torch.tensor(z_np, dtype=torch.float32, device=self.device)

        # Step 3: Group genes by chromosome for efficient loading
        chr_genes = {}
        for gene in gene_annotations:
            c = int(gene.chromosome)
            if c not in chr_genes:
                chr_genes[c] = []
            chr_genes[c].append(gene)

        results = []
        n_skipped = 0

        for chrom in sorted(chr_genes.keys()):
            genes_on_chr = chr_genes[chrom]

            if verbose:
                print(f"\nProcessing chr{chrom}: {len(genes_on_chr)} genes")

            self.ld._load_chromosome(chrom)

            for gene in genes_on_chr:
                gwas_idx = gene.snp_indices

                if len(gwas_idx) == 0:
                    n_skipped += 1
                    continue

                ref_idx = []
                gwas_matched = []
                for gi in gwas_idx:
                    if gi in gwas_to_ref:
                        ref_idx.append(gwas_to_ref[gi])
                        gwas_matched.append(gi)

                if len(ref_idx) < 2:
                    n_skipped += 1
                    continue

                ref_idx = np.array(ref_idx)
                gwas_matched = np.array(gwas_matched)

                # Truncate large genes
                if len(ref_idx) > self.max_snps:
                    p_subset = snp_pvalues[gwas_matched]
                    top_k = np.argsort(p_subset)[:self.max_snps]
                    ref_idx = ref_idx[top_k]
                    gwas_matched = gwas_matched[top_k]

                z_gene = z_all[gwas_matched]
                R = self.ld.compute_ld_matrix(ref_idx)
                stat, p_val, n_eff = self._gene_test_single(z_gene, R)

                z_score = float(sp_stats.norm.isf(p_val)) if p_val > 0 else 37.5

                results.append({
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "chr": gene.chromosome,
                    "start": gene.start,
                    "end": gene.end,
                    "n_snps": len(gwas_matched),
                    "n_effective": n_eff,
                    "stat": stat,
                    "p": p_val,
                    "z": z_score,
                })

            if verbose:
                tested = sum(1 for r in results if r["chr"] == str(chrom))
                print(f"  chr{chrom}: {tested} genes tested")

        if verbose:
            print(f"\nComplete: {len(results)} genes tested, {n_skipped} skipped")
            sig = sum(1 for r in results if r["p"] < 2.5e-6)
            print(f"Significant (p < 2.5e-6): {sig} genes")

        df = pd.DataFrame(results)
        return df.sort_values("p").reset_index(drop=True)
