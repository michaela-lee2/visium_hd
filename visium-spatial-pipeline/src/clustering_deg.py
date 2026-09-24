"""
Clustering (Leiden) and differential expression analysis for QC'd Visium data.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc


def clustering_deg(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    resolution: float = 0.5,
    output_prefix: str = "scanpy_output",
) -> sc.AnnData:
    """
    Run normalization, HVG selection, PCA/UMAP, Leiden clustering, and
    cluster-wise DEG analysis on a post-QC AnnData object.

    Parameters
    ----------
    adata
        AnnData object to analyze. Should contain raw counts after QC.
    n_top_genes
        Number of highly variable genes (HVGs) to select.
    resolution
        Leiden clustering resolution. Higher values yield more clusters.
    output_prefix
        Prefix used for all output files (figures, CSV, h5ad).

    Returns
    -------
    scanpy.AnnData
        AnnData object with clustering and DEG results added.
    """
    print(f"--- Starting analysis: {output_prefix} ---")
    print(f"Highly variable genes (HVG): {n_top_genes}")

    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(6, 6))

    figure_dir = f"./{output_prefix}_figures/"
    os.makedirs(figure_dir, exist_ok=True)
    sc.settings.figdir = figure_dir

    # =======================================================
    # 1. Clustering
    # =======================================================
    print("\n[1/3] Clustering...")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    sc.pp.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=resolution, key_added="clusters")
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color=["clusters"],
        legend_loc="on data",
        title=f"Leiden Clustering (HVG={n_top_genes})",
        save=f"_{output_prefix}_clustering_umap.png",
    )

    # Visium spatial coordinates must be float64 for sc.pl.spatial
    adata.obsm["spatial"] = adata.obsm["spatial"].astype(np.float64)
    sc.pl.spatial(
        adata,
        color="clusters",
        size=1.3,
        img_key="hires",
        save=f"_{output_prefix}_clustering_spatial_plot.png",
    )

    # =======================================================
    # 2. Cluster-wise DEG analysis
    # =======================================================
    print("\n[2/3] DEG analysis...")

    sc.tl.rank_genes_groups(adata, groupby="clusters", method="t-test")

    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=5,
        groupby="clusters",
        show_gene_labels=True,
        swap_axes=True,
        vmin=-3,
        vmax=3,
        cmap="bwr",
        save=f"_{output_prefix}_deg_heatmap.png",
    )

    sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=5, groupby="clusters", save=f"_{output_prefix}_deg_dotplot.png"
    )

    # =======================================================
    # 3. Save results
    # =======================================================
    print("\n[3/3] Saving results...")

    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    deg_full_results = pd.DataFrame(
        {
            group + "_" + key: result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", "scores", "pvals", "pvals_adj"]
        }
    )
    deg_csv_path = f"./{output_prefix}_deg_results_full.csv"
    deg_full_results.to_csv(deg_csv_path)

    adata_output_path = f"./{output_prefix}_processed.h5ad"
    adata.write(adata_output_path)

    print("\n--- Analysis complete ---")
    print(f"Figures: '{figure_dir}'")
    print(f"DEG results: '{deg_csv_path}'")
    print(f"Processed AnnData: '{adata_output_path}'")

    return adata


if __name__ == "__main__":
    input_file = "./Visium_AfterQC.h5ad"
    adata_to_process = sc.read_h5ad(input_file)

    processed_adata = clustering_deg(
        adata=adata_to_process.copy(),
        n_top_genes=2000,
        resolution=0.5,
        output_prefix="Visium_run1",
    )
