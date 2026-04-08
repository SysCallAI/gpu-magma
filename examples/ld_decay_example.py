"""
GPU-MAGMA: LDDecayApprox — No Reference Panel Required
=======================================================

Demonstrates the distance-based LD approximation path.  This approach
requires only GWAS summary statistics and SNP positions — no PLINK BED
files or synthetic reference genotypes needed.

LD structure is approximated as:
    R(i,j) = exp(-|bp_i - bp_j| / decay_rate)

with correlations below `min_corr` zeroed out for efficiency.

This matches the approximation used by SuSiE, PAINTOR, and related
fine-mapping methods.  It is less accurate than a real reference panel
but is useful when:
  - No reference panel is available for the population
  - Quick exploratory analysis is needed
  - The study population is non-standard (e.g., founder cohort)

Usage:
    python examples/ld_decay_example.py
"""

import time
import numpy as np
import pandas as pd
import torch

from gpu_magma import LDDecayApprox, GeneTestGPU, annotate_snps_to_genes


# ---------------------------------------------------------------------------
# 0. Device selection — auto-detect GPU, fall back to CPU
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Generate synthetic GWAS summary statistics
#
#    - 15K SNPs across chr1 and chr2
#    - Null p-values ~ Uniform(0, 1)
#    - Two injected signals: one strong, one moderate
# ---------------------------------------------------------------------------

N_SNPS    = 15_000
RNG_SEED  = 7

rng = np.random.default_rng(RNG_SEED)

print("Step 1: Generating GWAS summary statistics...")
t0 = time.time()

# Chromosomes: first 10K SNPs on chr1, last 5K on chr2
snp_chr = np.empty(N_SNPS, dtype=object)
snp_chr[:10_000] = "1"
snp_chr[10_000:] = "2"

# Positions: 1 kb spacing, chromosome coordinates reset at chr boundary
snp_bp = np.zeros(N_SNPS, dtype=np.int64)
snp_bp[:10_000] = np.arange(10_000) * 1_000 + 1_000_000   # chr1: 1 Mb – 11 Mb
snp_bp[10_000:] = np.arange(5_000)  * 1_000 + 1_000_000   # chr2: 1 Mb –  6 Mb

# Null background
snp_pvalues = rng.uniform(0, 1, N_SNPS).astype(np.float64)

print(f"  {N_SNPS:,} SNPs across chr1 ({10_000:,}) + chr2 ({5_000:,})")
print(f"  Done in {time.time() - t0:.4f}s\n")


# ---------------------------------------------------------------------------
# 2. Inject two signals
#
#    STRONG_GENE:   chr1, 4.0–5.0 Mb   — p-values down to 1e-10
#    MODERATE_GENE: chr2, 2.0–3.0 Mb   — p-values down to 1e-5
# ---------------------------------------------------------------------------

# Strong signal — chr1
STRONG_GENE_NAME  = "STRONG_GENE"
STRONG_CHR        = "1"
STRONG_START      = 4_000_000
STRONG_END        = 5_000_000

strong_mask    = (snp_chr == STRONG_CHR) & (snp_bp >= STRONG_START) & (snp_bp <= STRONG_END)
strong_indices = np.where(strong_mask)[0]
n_strong       = min(25, len(strong_indices))
chosen_strong  = rng.choice(strong_indices, size=n_strong, replace=False)
snp_pvalues[chosen_strong] = rng.uniform(1e-10, 1e-6, n_strong)

# Moderate signal — chr2
MODERATE_GENE_NAME = "MODERATE_GENE"
MODERATE_CHR       = "2"
MODERATE_START     = 2_000_000
MODERATE_END       = 3_000_000

mod_mask    = (snp_chr == MODERATE_CHR) & (snp_bp >= MODERATE_START) & (snp_bp <= MODERATE_END)
mod_indices = np.where(mod_mask)[0]
n_moderate  = min(20, len(mod_indices))
chosen_mod  = rng.choice(mod_indices, size=n_moderate, replace=False)
snp_pvalues[chosen_mod] = rng.uniform(1e-5, 1e-3, n_moderate)

print("Step 2: Signals injected")
print(f"  {STRONG_GENE_NAME}:   chr{STRONG_CHR} {STRONG_START:,}–{STRONG_END:,}, "
      f"min p={snp_pvalues[chosen_strong].min():.2e}")
print(f"  {MODERATE_GENE_NAME}: chr{MODERATE_CHR} {MODERATE_START:,}–{MODERATE_END:,}, "
      f"min p={snp_pvalues[chosen_mod].min():.2e}\n")


# ---------------------------------------------------------------------------
# 3. Build the LDDecayApprox object
#
#    Pass all SNP positions and chromosomes.  The decay_rate (50 kb default)
#    is appropriate for EUR populations; increase to 100–200 kb for
#    admixed or non-EUR populations with longer haplotype blocks.
# ---------------------------------------------------------------------------

print("Step 3: Initialising LDDecayApprox (no reference panel needed)...")
t0 = time.time()

ld_approx = LDDecayApprox(
    snp_positions=snp_bp,
    snp_chromosomes=snp_chr,
    decay_rate=50_000,   # 50 kb — EUR default
    min_corr=0.01,       # zero out r < 0.01 for sparsity
    device=device,
)

print(f"  Covering {ld_approx.n_snps:,} SNPs")
print(f"  decay_rate = {ld_approx.decay_rate / 1_000:.0f} kb")
print(f"  Done in {time.time() - t0:.4f}s\n")

# Optional: inspect the LD matrix for a small window as a sanity check
print("  Quick LD matrix inspection (50 SNPs around position 4.5 Mb, chr1):")
check_mask = (snp_chr == "1") & (snp_bp >= 4_400_000) & (snp_bp <= 4_900_000)
check_idx  = np.where(check_mask)[0][:50]
R_check    = ld_approx.compute_ld_matrix(check_idx)
off_diag   = R_check.fill_diagonal_(0).sum().item() / (len(check_idx) * (len(check_idx) - 1))
R_check.fill_diagonal_(1.0)  # restore
print(f"    Size: {R_check.shape}")
print(f"    Mean off-diagonal r: {off_diag:.4f}")
print(f"    Min eigenvalue: {torch.linalg.eigvalsh(R_check).min().item():.6f}\n")


# ---------------------------------------------------------------------------
# 4. Define gene locations
# ---------------------------------------------------------------------------

print("Step 4: Building gene location table...")
t0 = time.time()

gene_rows = []

# Background genes: chr1 (25 genes) + chr2 (15 genes)
for chrom, n_genes, spacing, offset in [
    ("1", 25, 400_000, 1_000_000),
    ("2", 15, 300_000, 1_000_000),
]:
    for k in range(n_genes):
        gs = offset + k * spacing
        ge = gs + 150_000
        gene_rows.append({
            "gene_id":   f"BGENE_{chrom}_{k:03d}",
            "chr":       chrom,
            "start":     gs,
            "end":       ge,
            "gene_name": f"BGENE_{chrom}_{k:03d}",
        })

# Signal genes
gene_rows.append({
    "gene_id":   STRONG_GENE_NAME,
    "chr":       STRONG_CHR,
    "start":     STRONG_START,
    "end":       STRONG_END,
    "gene_name": STRONG_GENE_NAME,
})
gene_rows.append({
    "gene_id":   MODERATE_GENE_NAME,
    "chr":       MODERATE_CHR,
    "start":     MODERATE_START,
    "end":       MODERATE_END,
    "gene_name": MODERATE_GENE_NAME,
})

gene_locations = pd.DataFrame(gene_rows)
print(f"  {len(gene_locations)} genes defined")
print(f"  Done in {time.time() - t0:.4f}s\n")


# ---------------------------------------------------------------------------
# 5. Annotate SNPs to genes
# ---------------------------------------------------------------------------

print("Step 5: Annotating SNPs to genes...")
t0 = time.time()

annotations = annotate_snps_to_genes(
    snp_chr=snp_chr,
    snp_bp=snp_bp,
    gene_locations=gene_locations,
    window=10_000,
)

print(f"  {len(annotations)} genes with >=1 SNP annotated")
print(f"  Done in {time.time() - t0:.2f}s\n")


# ---------------------------------------------------------------------------
# 6. Run the gene-level test using LDDecayApprox as the ld_computer
#
#    GeneTestGPU accepts any object with a .compute_ld_matrix(snp_indices)
#    method — both LDComputer (real reference) and LDDecayApprox (decay
#    approximation) satisfy this interface.
# ---------------------------------------------------------------------------

print("Step 6: Running gene-level test with LDDecayApprox...")
t0 = time.time()

tester = GeneTestGPU(
    ld_computer=ld_approx,    # <-- the key difference from the quickstart example
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
# 7. Print results and validate signal recovery
# ---------------------------------------------------------------------------

print("=" * 60)
print("TOP 10 GENE HITS")
print("=" * 60)
print(results_df.head(10)[["gene_name", "chr", "n_snps", "n_effective", "stat", "p", "z"]]
      .to_string(index=False))

print()
print("=" * 60)
print("SIGNAL GENE SUMMARY")
print("=" * 60)

for gene_name in [STRONG_GENE_NAME, MODERATE_GENE_NAME]:
    row_df = results_df[results_df["gene_name"] == gene_name]
    if row_df.empty:
        print(f"  {gene_name}: NOT FOUND in results")
        continue
    rank = results_df.index[results_df["gene_name"] == gene_name][0] + 1
    row  = row_df.iloc[0]
    print(f"  {gene_name}")
    print(f"    Rank:     #{rank} of {len(results_df)}")
    print(f"    p-value:  {row['p']:.3e}")
    print(f"    z-score:  {row['z']:.2f}")
    print(f"    F-stat:   {row['stat']:.2f}")
    print(f"    SNPs:     {row['n_snps']} ({row['n_effective']} effective)")
    print()

print("=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"  Gene test:    {t_gene_test:.2f}s")
print(f"  Genes tested: {len(results_df)}")
if len(results_df) > 0 and t_gene_test > 0:
    print(f"  Throughput:   {len(results_df) / t_gene_test:.0f} genes/s")

print()
print("NOTE: LDDecayApprox uses no reference panel.")
print("      For production analyses, prefer LDComputer.load_plink_bed()")
print("      with a population-matched reference panel (e.g., 1000G EUR).")


if __name__ == "__main__":
    pass  # All code runs at module level; guard is here for import safety.
