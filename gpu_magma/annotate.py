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
    snp_chr_str = np.array([str(c) for c in snp_chr])
    
    for _, gene in gene_locations.iterrows():
        gene_chr = str(gene["chr"])
        gene_start = int(gene["start"]) - window
        gene_end = int(gene["end"]) + window
        
        # Find SNPs in this gene region
        chr_mask = snp_chr_str == gene_chr
        pos_mask = (snp_bp >= gene_start) & (snp_bp <= gene_end)
        snp_mask = chr_mask & pos_mask
        
        snp_idx = np.where(snp_mask)[0]
        
        if len(snp_idx) == 0:
            continue
        
        annotations.append(GeneAnnotation(
            gene_id=str(gene["gene_id"]),
            gene_name=str(gene.get("gene_name", gene["gene_id"])),
            chromosome=gene_chr,
            start=int(gene["start"]),
            end=int(gene["end"]),
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
