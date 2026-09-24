# Visium Spatial Transcriptomics Pipeline

A lightweight, modular pipeline for analyzing 10x Genomics Visium spatial
transcriptomics data: quality control, clustering/DEG analysis, and
cell-type deconvolution with [cell2location](https://cell2location.readthedocs.io/).

Developed during an undergraduate research project applying spatial
transcriptomics to tumor microenvironment (TME) characterization.

## Pipeline overview

```
Raw Visium data (.h5)
        │
        ▼
   ┌─────────┐
   │   QC    │  filter low-quality spots/genes, annotate MT/Ribo/HB genes
   └────┬────┘
        │  adata_qc
        ▼
┌───────────────────┐        ┌────────────────────┐
│ Clustering & DEG   │        │   Deconvolution     │
│ (Leiden, UMAP,     │        │  (cell2location vs. │
│  rank_genes_groups)│        │  reference signature)│
└─────────┬──────────┘        └──────────┬──────────┘
          │                              │
          └──────────────┬───────────────┘
                          ▼
              merged .h5ad + .csv outputs
```

`qc.py` and `deconvolution.py` both take/return an `AnnData` object, so they
can be run independently or chained together via `pipeline.py`.

## Repository structure

```
visium-spatial-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── qc.py              # QC filtering (spots, genes, MT%)
    ├── clustering_deg.py  # Normalization, HVG, PCA/UMAP, Leiden, DEG
    ├── deconvolution.py   # cell2location-based cell-type deconvolution
    └── pipeline.py        # End-to-end runner with per-step memory/time logging
```

## Installation

```bash
git clone https://github.com/<your-username>/visium-spatial-pipeline.git
cd visium-spatial-pipeline
pip install -r requirements.txt
```

`cell2location` and `scvi-tools` benefit strongly from a CUDA-capable GPU for
the deconvolution step; `torch.cuda.is_available()` is checked automatically
and the pipeline falls back to CPU otherwise (much slower for large samples).

## Usage

### Run the full pipeline

```bash
cd src
python pipeline.py \
    --visium-dir /path/to/sample/ \
    --count-file filtered_feature_bc_matrix.h5 \
    --reference-csv /path/to/reference_signatures.csv \
    --output-prefix my_sample \
    --resolution 0.5 \
    --n-top-genes 2000 \
    --n-cells-per-location 10 \
    --max-epochs 20000 \
    --batch-size 2400
```

Each step (QC, clustering/DEG, deconvolution) runs in its own subprocess so
that peak RSS memory can be sampled over time — useful for right-sizing
compute resources on large Visium HD datasets. All settings are also
available as CLI flags; run `python pipeline.py --help` for the full list.

Outputs:
- `<prefix>_processed.h5ad` — final AnnData with clusters, DEG results, and
  deconvolution proportions merged into `.obs`
- `<prefix>_deg_results_full.csv` — per-cluster DEG table
- `<prefix>_deconv_obs.csv` — per-spot cell-type proportions
- `<prefix>_pipeline_performance_log.csv` — elapsed time / memory usage log
- `<prefix>_figures/` — UMAP, spatial cluster plot, DEG heatmap and dotplot

### Run steps individually

```python
from qc import run_qc_pipeline
from clustering_deg import clustering_deg
from deconvolution import run_cell2location_deconvolution

adata_qc = run_qc_pipeline(path="/path/to/sample/", count_file="filtered_feature_bc_matrix.h5")

adata_clustered = clustering_deg(adata_qc.copy(), n_top_genes=2000, resolution=0.5, output_prefix="my_sample")

adata_deconv = run_cell2location_deconvolution(
    adata=adata_qc,
    reference_csv="/path/to/reference_signatures.csv",
    n_cells_per_location=10,
    max_epochs=20000,
    batch_size=2400,
)
```

## Reference signature format

`deconvolution.py` expects a CSV with genes as rows (index) and cell types as
columns, containing the average expression signature per cell type (e.g. the
output of `cell2location`'s regression model on a matched scRNA-seq
reference).

## Notes

- QC thresholds (`min_genes=500`, `min_cells=5`, `pct_counts_mt < 30`) were
  tuned for this project's samples — adjust in `qc.py` for other datasets.
- `sc.tl.leiden` requires `leidenalg` and `python-igraph`.
- Deconvolution training (`max_epochs`) can take a long time on CPU; a GPU is
  strongly recommended for datasets with many spots.

## License

Add a license (e.g. MIT) if you plan to share this publicly.
