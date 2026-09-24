"""
Quality control for Visium spatial transcriptomics data.

Loads a 10x Visium sample, annotates mitochondrial / ribosomal / hemoglobin
genes, computes standard QC metrics, and filters low-quality spots and genes.
"""

import os
import scanpy as sc


def run_qc_pipeline(path: str, count_file: str) -> sc.AnnData:
    """
    Run quality control on Visium spatial transcriptomics data.

    Parameters
    ----------
    path
        Directory containing the Visium data (spatial/, filtered_feature_bc_matrix.h5, etc.).
    count_file
        Name of the .h5 count file (e.g. "filtered_feature_bc_matrix.h5").

    Returns
    -------
    scanpy.AnnData
        Filtered AnnData object after QC. Mitochondrial gene counts are kept
        separately in `adata.obsm["MT"]` and removed from `.var`.
    """

    print(f"\n🚀 [1] Loading Visium data from:\n{os.path.join(path, count_file)}")
    adata = sc.read_visium(path=path, count_file=count_file)

    # -- Step 2: Annotate gene categories --
    print("🔍 [2] Annotating gene categories (MT, Ribo, HB)...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)

    # -- Step 3: Calculate QC metrics --
    print("🧪 [3] Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # -- Step 4: Filter cells and genes --
    print("⚙️  [4] Filtering low-quality cells and genes...")
    adata_qc = adata.copy()
    initial_n_cells = adata_qc.n_obs
    initial_n_genes = adata_qc.n_vars

    sc.pp.filter_cells(adata_qc, min_genes=500)
    sc.pp.filter_genes(adata_qc, min_cells=5)
    adata_qc = adata_qc[adata_qc.obs["pct_counts_mt"] < 30].copy()

    # -- Step 5: Final cleanup --
    print("🧹 [5] Cleaning up metadata...")
    adata_qc.obs_names_make_unique()
    adata_qc.var_names_make_unique()
    adata_qc.var["SYMBOL"] = adata_qc.var_names
    adata_qc.var["MT_gene"] = adata_qc.var["SYMBOL"].str.startswith("MT-")

    # -- Step 6: Remove MT genes from .var, keep counts separately --
    print("🧬 [6] Extracting MT gene counts (saved in .obsm['MT'])...")
    adata_qc.obsm["MT"] = adata_qc[:, adata_qc.var["MT_gene"]].X.toarray()
    adata_qc = adata_qc[:, ~adata_qc.var["MT_gene"]]

    # -- Step 7: Summary report --
    final_n_cells = adata_qc.n_obs
    final_n_genes = adata_qc.n_vars
    cell_ratio = final_n_cells / initial_n_cells * 100
    gene_ratio = final_n_genes / initial_n_genes * 100

    print("\n📊 [7] QC Summary:")
    print(f" - Cells kept: {final_n_cells} / {initial_n_cells} ({cell_ratio:.2f}%)")
    print(f" - Genes kept: {final_n_genes} / {initial_n_genes} ({gene_ratio:.2f}%)")
    if gene_ratio < 50:
        print(" ⚠️  WARNING: More than 50% of genes were removed. Sample quality may be poor.")
    else:
        print(" ✅ Sample passed gene-level QC.")

    return adata_qc
