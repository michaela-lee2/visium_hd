"""
Cell-type deconvolution of Visium spots with cell2location.

Given a Visium AnnData object and a reference cell-type signature matrix,
estimates per-spot cell-type abundance and converts it into proportions.
"""

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import cell2location
from cell2location.models import Cell2location

warnings.filterwarnings(action="default")


def run_cell2location_deconvolution(
    adata: sc.AnnData = None,
    visium_path: str = None,
    reference_csv: str = None,
    n_cells_per_location: int = 30,
    max_epochs: int = 5000,
    batch_size: int = 5000,
) -> sc.AnnData:
    """
    Run cell2location deconvolution on a Visium sample.

    Parameters
    ----------
    adata
        Post-QC Visium AnnData object. If None, loaded from `visium_path`.
    visium_path
        Path to a .h5ad file to load if `adata` is not provided.
    reference_csv
        Path to a CSV of reference cell-type signatures (genes x cell types).
    n_cells_per_location
        Expected number of cells per Visium spot (cell2location prior).
    max_epochs
        Number of training epochs.
    batch_size
        Training batch size.

    Returns
    -------
    scanpy.AnnData
        AnnData with estimated cell-type abundances (`q05_<celltype>`) and
        proportions (`<celltype>`) added to `.obs`.
    """
    if adata is None:
        if visium_path is None:
            raise ValueError("Either 'adata' or 'visium_path' must be provided.")
        adata = sc.read_h5ad(visium_path)

    if reference_csv is None:
        raise ValueError("A reference signature CSV file must be provided.")

    # Load reference signatures
    inf_aver = pd.read_csv(reference_csv, index_col=0)

    # Find common genes and subset both datasets
    intersect = np.intersect1d(adata.var_names, inf_aver.index)
    adata = adata[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # Setup and train cell2location
    Cell2location.setup_anndata(adata=adata)

    mod = Cell2location(
        adata,
        cell_state_df=inf_aver,
        N_cells_per_location=n_cells_per_location,
        detection_alpha=20,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mod.train(max_epochs=max_epochs, batch_size=batch_size, train_size=1)

    # Extract posterior estimates
    adata = mod.export_posterior(
        adata, sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs}
    )

    # Replace spaces in factor names (compatibility with downstream tools, e.g. R)
    factor_names = np.array([name.replace(" ", "_") for name in adata.uns["mod"]["factor_names"]])
    adata.uns["mod"]["factor_names"] = factor_names

    # q05 estimate = estimated cell counts per spot per cell type
    abundance_columns = [f"q05_{name}" for name in factor_names]
    adata.obs[abundance_columns] = adata.obsm["q05_cell_abundance_w_sf"]

    # Convert counts to per-spot proportions
    total_abundance = adata.obs[abundance_columns].sum(axis=1)
    proportion_columns = list(factor_names)
    adata.obs[proportion_columns] = adata.obs[abundance_columns].div(total_abundance, axis=0).fillna(0)

    return adata
