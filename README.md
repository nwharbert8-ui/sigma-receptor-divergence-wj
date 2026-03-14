# Sigma Receptor Co-Expression Architecture Divergence (WJ-Native)

Weighted Jaccard analysis of continuous genome-wide Spearman correlation vectors reveals that SIGMAR1 (sigma-1) and TMEM97 (sigma-2) share 96.4% of their global co-expression architecture but diverge at their most functionally relevant partners (binary Jaccard = 0.100). Companion code for Harbert (2026), Frontiers in Pharmacology.

## Key Findings

- **WJ = 0.964** on continuous correlation vectors (SIGMAR1 vs TMEM97, BA9)
- **Binary J = 0.100** on top 5% co-expression networks (147 shared genes)
- **WJ-binary dissociation**: divergence concentrates at functional tails, not genome-wide
- SIGMAR1 tails: mitochondrial translation, TCA cycle metabolism
- TMEM97 tails: ubiquitin-proteasome, neurodegeneration pathways
- Hippocampal convergence is tail-specific (binary J: 0.100→0.260; WJ: 0.964→0.975)
- HSPA5 (BiP) as undesigned architectural validation (6 lowest WJ values)

## Dataset

- **Source**: GTEx v8 (dbGaP: phs000424.v8.p2)
- **Primary region**: Brain–Frontal Cortex (BA9), n = 209 samples, 16,225 expressed genes
- **Replication**: Putamen, Hippocampus, Nucleus accumbens, Anterior cingulate cortex (BA24)
- **Targets**: SIGMAR1, TMEM97, EIF2S1, PELO, LTN1, NEMF, HSPA5

## Methodology

1. Genome-wide Spearman correlations for each target vs all expressed genes
2. **Primary**: Weighted Jaccard on continuous correlation vectors (threshold-free)
3. **Secondary**: Binary Jaccard on top 5% network membership (tail localization)
4. Permutation testing (1,000 iterations, seed = 42)
5. FDR correction (Benjamini-Hochberg) across all 21 pairwise comparisons
6. gProfiler GO/KEGG/Reactome enrichment on divergent tails
7. Cell-type deconvolution sensitivity (53 markers, 6 cell types)
8. Covariate adjustment (age, sex)

## Requirements

```
scipy>=1.11
pandas>=1.5
numpy>=1.24
matplotlib>=3.7
matplotlib-venn>=0.11
gprofiler-official>=1.0
python-docx>=0.8
openpyxl>=3.0
requests>=2.28
statsmodels>=0.14
```

## Usage

```bash
pip install -r requirements.txt
python MS3_sigma_divergence_wj_pipeline.py
```

The pipeline downloads GTEx v8 data (~1.6 GB) on first run, then executes in ~4 minutes. All outputs save to Google Drive paths (configurable in the CONFIG section).

For Colab execution, change `DRIVE_BASE` to `/content/drive/MyDrive/MS3_JNC_Submission`.

## Output

- `MS3_Results/wj_continuous_all_21_pairs.csv` — All 21 WJ values
- `MS3_Results/binary_jaccard_top5pct.csv` — Binary Jaccard on top 5%
- `MS3_Results/wj_vs_binary_comparison.csv` — Side-by-side comparison
- `MS3_Results/*_genome_wide_rankings.csv` — Full rankings for all 7 targets
- `MS3_Results/gProfiler_*.csv` — Enrichment results
- `MS3_Results/provenance.json` — Methodology provenance
- `MS3_Figures/Figure1-3` — Publication figures (300 DPI, PDF + PNG)
- `MS3_Supplementary_Tables/Table_S1-S4` — Supplementary data

## Author

Drake H. Harbert (ORCID: 0009-0007-7740-3616)
Inner Architecture LLC, Canton, OH 44721, USA

## License

MIT
