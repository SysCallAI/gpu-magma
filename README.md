# GPU-MAGMA

GPU-accelerated gene-level association testing for GWAS summary statistics.

GPU-MAGMA reimplements the [MAGMA](https://ctg.cncr.nl/software/magma) gene analysis algorithm (de Leeuw et al., 2015, *PLOS Computational Biology*) using PyTorch, enabling massively parallel gene-level testing on NVIDIA GPUs. What takes hours on CPU completes in seconds on a single A100.

## Why

MAGMA is the standard tool for aggregating SNP-level GWAS associations to gene-level scores. But the original is CPU-only and single-threaded. As GWAS datasets grow to billions of rows and biobanks scale past a million participants, CPU MAGMA becomes a bottleneck.

GPU-MAGMA solves this by moving the core linear algebra (LD matrix computation, eigendecomposition, projection) to GPU. The algorithm is embarrassingly parallel across ~20,000 genes.

**No GPU-native MAGMA implementation existed before this project.**

## Algorithm

For each gene *g* with *k* mapped SNPs:

```
1. Annotate: Map SNPs to gene regions (positional window, default +/-10kb)
2. Extract:  Z-scores from GWAS p-values via inverse normal transform
3. LD:       Compute k x k LD correlation matrix R from reference genotypes
4. Eigen:    Decompose R = U Lambda U^T, threshold near-zero eigenvalues
5. Project:  W = Lambda^{-1/2} U^T Z  (decorrelated z-scores)
6. Test:     T = sum(W^2) ~ chi2(m), where m = effective eigenvalues
7. P-value:  Gene-level significance from chi-squared distribution
```

All steps from 3 onward execute on GPU via PyTorch batched operations.

## Installation

```bash
pip install gpu-magma
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0 (with CUDA support)
- numpy, pandas, scipy
- pandas-plink (for reading PLINK BED/BIM/FAM reference files)

```bash
# Full install with all dependencies
pip install gpu-magma[all]
```

## Quickstart

```python
import numpy as np
import pandas as pd
from gpu_magma import annotate_snps_to_genes, GeneTestGPU, LDComputer

# 1. Load your GWAS summary statistics
gwas = pd.read_csv("gwas_summary_stats.tsv", sep="\t")
snp_chr = gwas["CHR"].values
snp_bp = gwas["BP"].values
snp_pvalues = gwas["P"].values

# 2. Load gene locations (MAGMA format or BED)
from gpu_magma.annotate import load_gene_locations
genes = load_gene_locations("gene_locations_grch37.tsv")

# 3. Annotate SNPs to genes
annotations = annotate_snps_to_genes(snp_chr, snp_bp, genes, window=10_000)
print(f"Annotated {len(annotations)} genes")

# 4. Load LD reference panel
ld = LDComputer(device="cuda")
ld.load_plink_bed("path/to/1kg_eur")  # PLINK BED/BIM/FAM prefix

# 5. Run gene-level test
tester = GeneTestGPU(ld_computer=ld, device="cuda")
results = tester.run(snp_pvalues, annotations)

# 6. Results
print(results.head(20))
results.to_csv("gene_results.csv", index=False)
```

### Output columns

| Column | Description |
|--------|-------------|
| `gene_id` | Ensembl gene ID |
| `gene_name` | HGNC symbol |
| `chr` | Chromosome |
| `start` | Gene start position |
| `end` | Gene end position |
| `n_snps` | Number of GWAS SNPs mapped to gene |
| `n_effective` | Effective number of independent signals (eigenvalues above threshold) |
| `stat` | Chi-squared test statistic |
| `p` | Gene-level p-value |
| `z` | Z-score (from p-value) |

## LD Reference Panel

GPU-MAGMA requires an LD reference panel in PLINK BED format. We provide a preprocessed 1000 Genomes EUR panel on Hugging Face:

**[`mikpam168/1kg-eur-grch37-plink`](https://huggingface.co/datasets/mikpam168/1kg-eur-grch37-plink)**

This contains per-chromosome PLINK files for 503 European-ancestry samples from the 1000 Genomes Phase 3 project, aligned to GRCh37/hg19.

```python
# Download with huggingface_hub
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="mikpam168/1kg-eur-grch37-plink",
    repo_type="dataset",
    local_dir="./1kg_eur",
)
```

### Using per-chromosome LD (recommended for large GWAS)

For genome-wide analyses, use `PerChromLDComputer` which lazily loads one chromosome at a time to manage GPU memory:

```python
from gpu_magma.ld_real import PerChromLDComputer, RealLDGeneTest

# Per-chromosome LD (lazy loading, memory efficient)
ld = PerChromLDComputer(
    bed_dir="./1kg_eur",
    bed_prefix="1kg_eur_chr",  # expects 1kg_eur_chr1.bed, 1kg_eur_chr2.bed, ...
    device="cuda",
)

# Run with real LD
tester = RealLDGeneTest(ld_computer=ld, device="cuda")
results = tester.run(
    snp_pvalues=gwas["P"].values,
    gene_annotations=annotations,
    gwas_chr=gwas["CHR"].values,
    gwas_bp=gwas["BP"].values,
)
```

### No reference panel? Use LD decay approximation

If you don't have a reference panel, `LDDecayApprox` estimates LD from physical distance using an exponential decay model:

```python
from gpu_magma.ld_decay import LDDecayApprox

# Approximate LD (no reference panel needed)
ld = LDDecayApprox(
    snp_positions=gwas["BP"].values,
    decay_rate=50_000,  # 50kb half-life (EUR default)
    device="cuda",
)

tester = GeneTestGPU(ld_computer=ld, device="cuda")
results = tester.run(snp_pvalues, annotations)
```

Note: LD decay approximation is less accurate than a real reference panel and may inflate the number of significant genes. Use a real panel for publication-quality results.

## Three LD Strategies

| Strategy | Accuracy | Memory | Speed | When to use |
|----------|----------|--------|-------|-------------|
| `LDComputer` | Highest | High (full genome in VRAM) | Fastest | Small reference, plenty of VRAM |
| `PerChromLDComputer` | Highest | Moderate (one chr at a time) | Fast | Genome-wide with real LD |
| `LDDecayApprox` | Approximate | Minimal | Fastest | Exploratory analysis, no reference panel |

## Performance

Benchmarked on PGC3 Schizophrenia GWAS (37.5M SNPs, 18,627 genes tested):

| Platform | Time | Notes |
|----------|------|-------|
| CPU MAGMA v1.10 | ~2-4 hours | Single-threaded |
| **GPU-MAGMA (A100 80GB)** | **~45 seconds** | Real 1000G EUR LD |
| **GPU-MAGMA (RTX 4090 24GB)** | **~2 minutes** | Per-chromosome loading |

GPU-MAGMA achieves ~200x speedup on A100 compared to CPU MAGMA.

## Gene locations

GPU-MAGMA can load gene locations from MAGMA-format files or BED files. If you don't have a gene location file, you can generate one from Ensembl:

```python
from gpu_magma.annotate import generate_gene_locations_from_ensembl

genes = generate_gene_locations_from_ensembl(
    build="GRCh37",
    output_path="gene_locations_grch37.tsv",
)
```

## Configuration

Key parameters and their defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 10,000 bp | SNP-to-gene annotation window |
| `eigenvalue_threshold` | 1e-5 | Minimum eigenvalue to retain |
| `max_snps_per_gene` | 500 | Truncate large genes (keeps top SNPs by p-value) |
| `decay_rate` | 50,000 bp | LD decay half-life for approximation mode |

## Modules

```
gpu_magma/
  __init__.py       # Public API: annotate_snps_to_genes, GeneTestGPU, LDComputer
  annotate.py       # SNP-to-gene positional annotation
  gene_test.py      # GPU gene-level association test (MAGMA mean model)
  ld.py             # LD computation from PLINK reference (full genome)
  ld_real.py        # Per-chromosome LD with lazy loading
  ld_decay.py       # LD approximation from physical distance
```

## Citation

If you use GPU-MAGMA in your research, please cite:

```bibtex
@software{gpu_magma,
  title={GPU-MAGMA: GPU-accelerated gene-level GWAS association testing},
  author={SysCallAI},
  year={2026},
  url={https://github.com/SysCallAI/gpu-magma}
}
```

And the original MAGMA paper:

```bibtex
@article{de_leeuw_magma_2015,
  title={MAGMA: Generalized Gene-Set Analysis of GWAS Data},
  author={de Leeuw, Christiaan A and Mooij, Joris M and Heskes, Tom and Posthuma, Danielle},
  journal={PLOS Computational Biology},
  volume={11},
  number={4},
  pages={e1004219},
  year={2015}
}
```

## License

Apache 2.0
