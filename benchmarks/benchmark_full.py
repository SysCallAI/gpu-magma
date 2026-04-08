"""
GPU-MAGMA Full-Genome Benchmark on A100
========================================
1000 Genomes EUR (503 samples) x all 22 chromosomes
Uses MAGMA gene location file for ~18K protein-coding genes.

Data source: https://huggingface.co/datasets/mikpam168/1kg-eur-grch37-plink
"""
import time, os, sys
import numpy as np
import pandas as pd
import torch

# ---------- config ----------
BED_DIR       = "/workspace/1kg_eur"
GENE_LOC_FILE = "/workspace/gene_locations.tsv"
DEVICE        = "cuda"
HF_DATASET    = "mikpam168/1kg-eur-grch37-plink"

print("=" * 70)
print("GPU-MAGMA Full-Genome Benchmark")
print("=" * 70)
print(f"Device:  {torch.cuda.get_device_name(0)}")
print(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print()

# ---------- Step 0: Download data if missing ----------
def download_data():
    missing = [c for c in range(1, 23)
               if not os.path.exists(f"{BED_DIR}/1kg_eur_chr{c}.bed")]

    if not missing:
        print(f"All 22 chromosomes present in {BED_DIR}")
    else:
        print(f"Downloading {len(missing)} chromosome(s) from {HF_DATASET}...")
        t0 = time.time()
        from huggingface_hub import snapshot_download
        snapshot_download(HF_DATASET, repo_type="dataset", local_dir=BED_DIR)
        print(f"  Downloaded in {time.time()-t0:.1f}s")

    if not os.path.exists(GENE_LOC_FILE):
        raise FileNotFoundError(
            f"Gene location file not found: {GENE_LOC_FILE}\n"
            "Generate it with: gpu_magma.annotate.generate_gene_locations_from_ensembl()"
        )


download_data()
print()

# ---------- Step 1: Build index ----------
from gpu_magma import annotate_snps_to_genes, GeneAnnotation
from gpu_magma.ld_real import PerChromLDComputer, RealLDGeneTest

print("Step 1: Building chromosome index...")
t0 = time.time()
ld = PerChromLDComputer(bed_dir=BED_DIR, bed_prefix="1kg_eur_chr", device=DEVICE)
t_index = time.time() - t0
print(f"  Index built in {t_index:.1f}s\n")

# ---------- Step 2: Load gene locations ----------
print("Step 2: Loading MAGMA gene locations...")
t0 = time.time()

gene_df = pd.read_csv(
    GENE_LOC_FILE, sep="\t", header=None,
    names=["gene_id", "chr", "start", "end", "strand", "gene_name"]
)
gene_df["chr"] = gene_df["chr"].astype(str)
gene_df = gene_df[gene_df["chr"].isin([str(c) for c in range(1, 23)])].copy()
print(f"  {len(gene_df)} autosomal genes loaded")
print(f"  Done in {time.time()-t0:.2f}s\n")

# ---------- Step 3: Generate synthetic GWAS from BIM positions ----------
print("Step 3: Generating synthetic GWAS from BIM positions...")
t0 = time.time()

all_chr = []
all_bp = []
for chrom in range(1, 23):
    bim_path = f"{BED_DIR}/1kg_eur_chr{chrom}.bim"
    if not os.path.exists(bim_path):
        continue
    bim = pd.read_csv(bim_path, sep="\t", header=None, usecols=[0, 3], names=["chr", "bp"])
    all_chr.append(np.full(len(bim), str(chrom)))
    all_bp.append(bim["bp"].values)

gwas_chr = np.concatenate(all_chr)
gwas_bp = np.concatenate(all_bp)
n_snps = len(gwas_chr)

rng = np.random.default_rng(42)
gwas_pvalues = rng.uniform(0, 1, n_snps).astype(np.float64)

# Inject signals in 5 random genes
signal_genes = gene_df.sample(5, random_state=42)
for _, sg in signal_genes.iterrows():
    mask = (gwas_chr == str(sg["chr"])) & (gwas_bp >= sg["start"]) & (gwas_bp <= sg["end"])
    signal_idx = np.where(mask)[0]
    if len(signal_idx) > 0:
        n_sig = min(30, len(signal_idx))
        chosen = rng.choice(signal_idx, size=n_sig, replace=False)
        gwas_pvalues[chosen] = rng.uniform(1e-10, 1e-5, n_sig)

t_gwas = time.time() - t0
print(f"  {n_snps:,} total SNPs across 22 chromosomes")
print(f"  5 signal genes injected")
print(f"  Done in {t_gwas:.1f}s\n")

# ---------- Step 4: Annotate SNPs to genes (searchsorted) ----------
print("Step 4: Annotating SNPs to genes (vectorized searchsorted)...")
t0 = time.time()

annotations = annotate_snps_to_genes(
    snp_chr=gwas_chr,
    snp_bp=gwas_bp,
    gene_locations=gene_df,
    window=10_000,
)

t_annotate = time.time() - t0
n_annotated = len(annotations)
snp_counts = [a.n_snps for a in annotations]
print(f"  {n_annotated} genes with >=1 SNP")
print(f"  Annotation done in {t_annotate:.2f}s")
print(f"  SNPs/gene: min={min(snp_counts)}, median={np.median(snp_counts):.0f}, "
      f"mean={np.mean(snp_counts):.0f}, max={max(snp_counts)}")
print()

# ---------- Step 5: Run gene-level test ----------
print("Step 5: Running gene-level test (RealLDGeneTest)...")
sys.stdout.flush()
t0 = time.time()

tester = RealLDGeneTest(
    ld_computer=ld,
    device=DEVICE,
    max_snps_per_gene=500,
    eigenvalue_threshold=1e-5,
)

results = tester.run(
    snp_pvalues=gwas_pvalues,
    gene_annotations=annotations,
    gwas_chr=gwas_chr.astype(int),
    gwas_bp=gwas_bp,
    verbose=True,
)

t_gene_test = time.time() - t0
print(f"\n  Gene test done in {t_gene_test:.1f}s")

# ---------- Step 6: Results ----------
print()
print("=" * 70)
print("TOP 15 GENE HITS")
print("=" * 70)
cols = ["gene_name", "chr", "n_snps", "n_effective", "stat", "p", "z"]
print(results.head(15)[cols].to_string(index=False))

mem_alloc = torch.cuda.max_memory_allocated() / 1e9
mem_reserved = torch.cuda.max_memory_reserved() / 1e9

print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"  Total SNPs:       {n_snps:,}")
print(f"  Genes annotated:  {n_annotated}")
print(f"  Genes tested:     {len(results)}")
print()
print("  TIMING:")
print(f"    Index build:    {t_index:.1f}s")
print(f"    GWAS generation:{t_gwas:.1f}s")
print(f"    Annotation:     {t_annotate:.2f}s  (searchsorted)")
print(f"    Gene test:      {t_gene_test:.1f}s")
if len(results) > 0 and t_gene_test > 0:
    print(f"    Throughput:     {len(results)/t_gene_test:.0f} genes/s")
print()
print("  GPU MEMORY:")
print(f"    Peak allocated: {mem_alloc:.2f} GB")
print(f"    Peak reserved:  {mem_reserved:.2f} GB")
print()

# Signal gene recovery
print("  SIGNAL RECOVERY:")
for _, sg in signal_genes.iterrows():
    name = sg["gene_name"]
    row = results[results["gene_name"] == name]
    if not row.empty:
        rank = row.index[0] + 1
        p = row.iloc[0]["p"]
        print(f"    {name}: rank #{rank}/{len(results)}, p={p:.2e}")
    else:
        print(f"    {name}: NOT IN RESULTS (no matched SNPs)")

print()
print("Done.")
