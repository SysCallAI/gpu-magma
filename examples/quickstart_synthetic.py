"""
GPU-MAGMA Quickstart: Synthetic GWAS Signal Recovery
=====================================================

Demonstrates the full GPU-MAGMA pipeline on fully synthetic data:
  1. Build a synthetic reference panel (LDComputer.load_synthetic)
  2. Generate fake GWAS summary stats with numpy (20K SNPs, chr1-3)
  3. Inject a strong signal in one gene region
  4. Annotate SNPs → genes
  5. Run the gene-level test (GeneTestGPU)
  6. Confirm the signal gene is the top hit

No external files required — runs entirely in-memory.

Usage:
    python examples/quickstart_synthetic.py
"""

import time
import numpy as np
import pandas as pd
import torch

from gpu_magma import annotate_snps_to_genes, GeneTestGPU, LDComputer


# ---------------------------------------------------------------------------
# 0. Device selection
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Build a synthetic reference panel
#
#    LDComputer.load_synthetic() generates block-diagonal LD structure
#    with realistic within-block correlations (~r=0.5 between neighbours).
#    We match the reference panel's SNP count to our GWAS SNP count so that
#    indices align one-to-one (no SNP-matching step needed).
# ---------------------------------------------------------------------------

N_SNPS = 20_000         # total GWAS SNPs
N_SAMPLES_REF = 500     # reference panel sample size
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)

t_start = time.time()

print("Step 1: Loading synthetic reference panel...")
t0 = time.time()

ld = LDComputer(device=device)
ld.load_synthetic(n_samples=N_SAMPLES_REF, n_snps=N_SNPS)

print(f"  Done in {time.time() - t0:.2f}s\n")


# ---------------------------------------------------------------------------
# 2. Generate fake GWAS summary statistics
#
#    - 20K SNPs spread across chr1, chr2, chr3
#    - Chromosome boundaries: chr1 = SNPs 0-7999, chr2 = 8000-15999,
#      chr3 = 16000-19999
#    - Base pair positions: 1 kb spacing per chromosome
#    - Under the null: p-values ~ Uniform(0, 1)
# ---------------------------------------------------------------------------

print("Step 2: Generating synthetic GWAS summary statistics...")
t0 = time.time()

# Assign chromosomes (roughly 8K / 8K / 4K split across chr1-3)
snp_chr = np.empty(N_SNPS, dtype=object)
snp_chr[:8000]  = "1"
snp_chr[8000:16000] = "2"
snp_chr[16000:]     = "3"

# Base-pair positions: reset to 1 at each chromosome start, 1 kb spacing
snp_bp = np.zeros(N_SNPS, dtype=np.int64)
snp_bp[:8000]       = np.arange(8000)   * 1_000 + 1_000_000
snp_bp[8000:16000]  = np.arange(8000)   * 1_000 + 1_000_000
snp_bp[16000:]      = np.arange(4000)   * 1_000 + 1_000_000

# Null p-values drawn from Uniform(0, 1)
snp_pvalues = rng.uniform(0, 1, N_SNPS).astype(np.float64)

print(f"  {N_SNPS:,} SNPs generated across chr1-3")
print(f"  Done in {time.time() - t0:.4f}s\n")


# ---------------------------------------------------------------------------
# 3. Inject a strong signal in the "SIGNAL_GENE" region
#
#    The signal gene sits on chr2 around position 5.0 Mb (SNP indices ~12000).
#    We pick 30 consecutive SNPs and replace their p-values with values drawn
#    from Uniform(1e-8, 1e-4) — well into genome-wide significance territory.
#    This mimics a dense association peak expected from a causal variant in LD.
# ---------------------------------------------------------------------------

SIGNAL_GENE_NAME = "SIGNAL_GENE"

# Signal window: chr2, ~4.5 Mb – 5.5 Mb
SIGNAL_CHR       = "2"
SIGNAL_START_BP  = 4_500_000
SIGNAL_END_BP    = 5_500_000

# Find the SNP indices that fall in this window
signal_mask = (snp_chr == SIGNAL_CHR) & (snp_bp >= SIGNAL_START_BP) & (snp_bp <= SIGNAL_END_BP)
signal_indices = np.where(signal_mask)[0]

# Inject strong p-values for 30 of those SNPs
n_signal_snps = min(30, len(signal_indices))
chosen = rng.choice(signal_indices, size=n_signal_snps, replace=False)
snp_pvalues[chosen] = rng.uniform(1e-8, 1e-4, n_signal_snps)

print(f"Step 3: Signal injected")
print(f"  Gene: {SIGNAL_GENE_NAME} (chr{SIGNAL_CHR}: {SIGNAL_START_BP:,}–{SIGNAL_END_BP:,})")
print(f"  {n_signal_snps} SNPs given p-values in [1e-8, 1e-4]")
print(f"  Minimum injected p-value: {snp_pvalues[chosen].min():.3e}\n")


# ---------------------------------------------------------------------------
# 4. Build a gene location table
#
#    In real usage you would load this from a MAGMA .loc file or BED file via
#    load_gene_locations(). Here we construct a minimal DataFrame with:
#      - 100 background genes spread across chr1-3
#      - 1 signal gene matching the injected window
# ---------------------------------------------------------------------------

print("Step 4: Building gene location table...")
t0 = time.time()

gene_rows = []

# Background genes: 30 on chr1, 30 on chr2, 20 on chr3
# Each gene spans 100 kb, spaced ~300 kb apart so they don't overlap.
for g_idx, (chrom, n_genes_on_chr, chr_offset) in enumerate([
    ("1",  30, 1_000_000),
    ("2",  30, 1_000_000),
    ("3",  20, 1_000_000),
]):
    for k in range(n_genes_on_chr):
        gene_start = chr_offset + k * 300_000
        gene_end   = gene_start + 100_000
        gene_rows.append({
            "gene_id":   f"GENE_{chrom}_{k:03d}",
            "chr":       chrom,
            "start":     gene_start,
            "end":       gene_end,
            "gene_name": f"GENE_{chrom}_{k:03d}",
        })

# The signal gene: overlaps the injected window exactly
gene_rows.append({
    "gene_id":   SIGNAL_GENE_NAME,
    "chr":       SIGNAL_CHR,
    "start":     SIGNAL_START_BP,
    "end":       SIGNAL_END_BP,
    "gene_name": SIGNAL_GENE_NAME,
})

gene_locations = pd.DataFrame(gene_rows)
print(f"  {len(gene_locations)} genes in reference")
print(f"  Done in {time.time() - t0:.4f}s\n")


# ---------------------------------------------------------------------------
# 5. Annotate SNPs to genes
#
#    annotate_snps_to_genes() maps each SNP to every gene whose window
#    (gene body ± `window` bp) contains that SNP's position.
#    Returns a list of GeneAnnotation objects.
# ---------------------------------------------------------------------------

print("Step 5: Annotating SNPs to genes...")
t0 = time.time()

annotations = annotate_snps_to_genes(
    snp_chr=snp_chr,
    snp_bp=snp_bp,
    gene_locations=gene_locations,
    window=10_000,   # 10 kb upstream/downstream of each gene body
)

print(f"  {len(annotations)} genes with >=1 SNP annotated")
print(f"  Done in {time.time() - t0:.2f}s\n")


# ---------------------------------------------------------------------------
# 6. Run the gene-level test
#
#    GeneTestGPU.run() performs, for each gene:
#      a) Convert SNP p-values → z-scores
#      b) Compute LD matrix from the reference panel
#      c) Eigendecompose LD, threshold near-zero eigenvalues
#      d) Project z-scores into eigenspace (whitened decorrelation)
#      e) Compute F-statistic → gene p-value
#
#    Results are returned as a DataFrame sorted by ascending p-value.
# ---------------------------------------------------------------------------

print("Step 6: Running gene-level test (GPU-MAGMA)...")
t0 = time.time()

tester = GeneTestGPU(
    ld_computer=ld,
    eigenvalue_threshold=1e-5,
    max_snps_per_gene=500,
    device=device,
)

results_df = tester.run(
    snp_pvalues=snp_pvalues,
    gene_annotations=annotations,
    verbose=True,
)

t_gene_test = time.time() - t0
print(f"  Gene test done in {t_gene_test:.2f}s\n")


# ---------------------------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------------------------

t_total = time.time() - t_start

print("=" * 60)
print("TOP 10 GENE HITS")
print("=" * 60)
top10 = results_df.head(10)[["gene_name", "chr", "n_snps", "n_effective", "stat", "p", "z"]]
print(top10.to_string(index=False))

print()
print("=" * 60)
print("SIGNAL GENE RESULT")
print("=" * 60)
signal_row = results_df[results_df["gene_name"] == SIGNAL_GENE_NAME]
if not signal_row.empty:
    rank = results_df.index[results_df["gene_name"] == SIGNAL_GENE_NAME][0] + 1
    row = signal_row.iloc[0]
    print(f"  Gene:        {row['gene_name']}")
    print(f"  Rank:        #{rank} of {len(results_df)} genes tested")
    print(f"  p-value:     {row['p']:.3e}")
    print(f"  z-score:     {row['z']:.2f}")
    print(f"  F-stat:      {row['stat']:.2f}")
    print(f"  SNPs used:   {row['n_snps']} ({row['n_effective']} effective)")
    if rank == 1:
        print("  ✓ Signal gene is the #1 hit — recovery CONFIRMED")
    else:
        print(f"  Signal gene is rank #{rank} — check signal injection or gene window")
else:
    print(f"  WARNING: {SIGNAL_GENE_NAME} not found in results")

print()
print("=" * 60)
print(f"TIMING SUMMARY")
print("=" * 60)
print(f"  Total elapsed:    {t_total:.2f}s")
print(f"  Gene test alone:  {t_gene_test:.2f}s")
print(f"  Genes tested:     {len(results_df)}")
if len(results_df) > 0:
    print(f"  Throughput:       {len(results_df) / t_gene_test:.0f} genes/s")


if __name__ == "__main__":
    pass  # All code runs at module level; guard is here for import safety.
