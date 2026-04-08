"""
SNP-to-gene annotation module.

Maps SNPs to genes based on genomic position with configurable window.
Equivalent to MAGMA --annotate step.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class GeneAnnotation:
    """Result of SNP-to-gene annotation."""
    gene_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    snp_indices: np.ndarray  # indices into the GWAS SNP array
    n_snps: int


def load_gene_locations(path: str) -> pd.DataFrame:
    """Load gene location file (MAGMA format or BED).
    
    MAGMA format: gene_id  chr  start  end  strand  gene_name
    BED format: chr  start  end  gene_name
    """
    try:
        # Try MAGMA format first (tab-separated, 6 columns)
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["gene_id", "chr", "start", "end", "strand", "gene_name"],
                         comment="#")
        if len(df.columns) >= 4:
            df["chr"] = df["chr"].astype(str).str.replace("chr", "")
            return df[["gene_id", "chr", "start", "end", "gene_name"]].copy()
    except Exception:
        pass
    
    # Try BED format
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["chr", "start", "end", "gene_name"][:len(df.columns)]
    df["gene_id"] = df["gene_name"]
    df["chr"] = df["chr"].astype(str).str.replace("chr", "")
    return df


def annotate_snps_to_genes(
    snp_chr: np.ndarray,
    snp_bp: np.ndarray,
    gene_locations: pd.DataFrame,
    window: int = 10000,
) -> list[GeneAnnotation]:
    """Map SNPs to genes based on position.

    Args:
        snp_chr: array of chromosome labels for each SNP
        snp_bp: array of base pair positions for each SNP
        gene_locations: DataFrame with gene_id, chr, start, end, gene_name
        window: upstream/downstream window in base pairs (default 10kb)

    Returns:
        List of GeneAnnotation objects, one per gene with mapped SNPs
    """
    annotations = []

    # Convert chromosomes to string for matching
    snp_chr_str = np.asarray(snp_chr, dtype=str)
    snp_bp = np.asarray(snp_bp, dtype=np.int64)

    # Pre-group SNPs by chromosome for O(n_chr) scans instead of O(n_genes)
    chr_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unique_chrs = np.unique(snp_chr_str)
    for c in unique_chrs:
        mask = snp_chr_str == c
        indices = np.where(mask)[0]
        positions = snp_bp[indices]
        # Sort by position for searchsorted
        sort_order = np.argsort(positions)
        chr_groups[c] = (indices[sort_order], positions[sort_order])

    # Iterate genes using pre-grouped chromosome data
    gene_chr_col = gene_locations["chr"].astype(str).values
    gene_start_col = gene_locations["start"].values.astype(np.int64)
    gene_end_col = gene_locations["end"].values.astype(np.int64)
    gene_id_col = gene_locations["gene_id"].astype(str).values
    gene_name_col = (gene_locations["gene_name"].astype(str).values
                     if "gene_name" in gene_locations.columns
                     else gene_id_col)

    for i in range(len(gene_locations)):
        gene_chr = gene_chr_col[i]

        if gene_chr not in chr_groups:
            continue

        chr_indices, chr_positions = chr_groups[gene_chr]
        gene_start = int(gene_start_col[i]) - window
        gene_end = int(gene_end_col[i]) + window

        # Binary search for SNPs in [gene_start, gene_end]
        left = np.searchsorted(chr_positions, gene_start, side="left")
        right = np.searchsorted(chr_positions, gene_end, side="right")

        if left >= right:
            continue

        snp_idx = chr_indices[left:right]

        annotations.append(GeneAnnotation(
            gene_id=gene_id_col[i],
            gene_name=gene_name_col[i],
            chromosome=gene_chr,
            start=int(gene_start_col[i]),
            end=int(gene_end_col[i]),
            snp_indices=snp_idx,
            n_snps=len(snp_idx),
        ))

    return annotations


def generate_gene_locations_from_ensembl(
    build: str = "GRCh37",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate gene location file from Ensembl REST API.
    
    Fallback when MAGMA gene location files are unavailable.
    """
    import requests
    
    server = "https://grch37.rest.ensembl.org" if build == "GRCh37" else "https://rest.ensembl.org"
    
    genes = []
    for chrom in list(range(1, 23)) + ["X", "Y"]:
        url = f"{server}/overlap/region/human/{chrom}:1-300000000?feature=gene;biotype=protein_coding"
        headers = {"Content-Type": "application/json"}
        
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                for g in r.json():
                    genes.append({
                        "gene_id": g.get("id", ""),
                        "chr": str(chrom),
                        "start": g["start"],
                        "end": g["end"],
                        "gene_name": g.get("external_name", g.get("id", "")),
                    })
        except Exception as e:
            print(f"Warning: failed to fetch chr{chrom}: {e}")
            continue
    
    df = pd.DataFrame(genes)
    
    if output_path:
        df.to_csv(output_path, sep="\t", index=False, header=False)
    
    return df
