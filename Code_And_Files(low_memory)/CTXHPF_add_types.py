#!/usr/bin/env python3
import pandas as pd
import anndata
import json

# Load your AnnData object:
adata = anndata.read_h5ad("/user/work/pr21872/Homologs/samap_directory/sam1_mouse.h5ad")

# Load metadata
CTXHPF_metadata = pd.read_csv(
    '/user/work/pr21872/Homologs/h5df2hda5/Metadata_CTXHPF_Cells_Link_Mouse_Paper.csv',
    usecols=['cell_type_alias_id', 'sample_name']
)

# Load cell_type to subclass_id dictionary
with open('/user/work/pr21872/Homologs/samap_directory/add_cell_types/CTXHPFcluster_to_MouseSupertype_mapping.json', 'r') as fp:
    cell_types = json.load(fp)

# Load the subclass mapping CSV
subclass_df = pd.read_csv('/user/work/pr21872/Homologs/samap_directory/add_cell_types/mouse_data_cluster_metadata_from_paper.csv', usecols=['supertype_id', 'supertype_label'])

# Convert subclass_id to float to match the values in cell_types
subclass_df['supertype_id'] = subclass_df['supertype_id'].astype(float)

# Drop duplicates to ensure unique mapping from ID to label
subclass_mapping = subclass_df.drop_duplicates('supertype_id').set_index('supertype_id')['supertype_label'].to_dict()

# Update the values in cell_types dictionary
cell_types_labeled = {float(k): subclass_mapping.get(v, v) for k, v in cell_types.items()}

# Add new column using the mapping
CTXHPF_metadata['mouse_supertype'] = CTXHPF_metadata['cell_type_alias_id'].map(cell_types_labeled)

print(CTXHPF_metadata[0:5])
n_nans = CTXHPF_metadata['mouse_supertype'].isna().sum()
print(f"Number of NaNs: {n_nans}")
rows_with_nans = CTXHPF_metadata[CTXHPF_metadata['mouse_supertype'].isna()]['cell_type_alias_id'].unique()
print(rows_with_nans)

# #make sample name the index
CTXHPF_metadata.set_index('sample_name', inplace=True)

# If the cell order might be different, you can reindex to ensure alignment:
adata.obs["Supertype"] = CTXHPF_metadata["mouse_supertype"].reindex(adata.obs.index)

adata.write_h5ad("/user/work/pr21872/Homologs/samap_directory/sam1_mouse_w_Supercelltypes.h5ad")


# import matplotlib.pyplot as plt
# import numpy as np

# # Load the integrated SAMap data
# samap_adata = anndata.read_h5ad("/user/work/pr21872/Homologs/samap_directory/mark2_combined_samap.h5ad")
# umap_key = 'X_umap'

# # Load the individual species data (adjust the paths accordingly)
# mouse_adata = anndata.read_h5ad("/user/work/pr21872/Homologs/mouse_data.h5ad", backed='r')
# macaque_adata = anndata.read_h5ad("/user/work/pr21872/Homologs/macaque_data.h5ad", backed='r')

# def plot_highlighted_umap(samap_adata, mouse_adata, macaque_adata,
#                           mouse_cell_type, macaque_cell_type,
#                           umap_key='X_umap'):
#     """
#     Plots the integrated UMAP projection with highlighted cells for the given mouse and macaque cell types.
    
#     Parameters:
#       samap_adata (AnnData): Integrated SAMap data containing UMAP coordinates and a "species" column.
#       mouse_adata (AnnData): Mouse-specific data containing a "Supertype" column.
#       macaque_adata (AnnData): Macaque-specific data containing a "Subclass" column.
#       mouse_cell_type (str): The mouse cell type (as named in mouse_adata.obs["Supertype"]) to highlight.
#       macaque_cell_type (str): The macaque cell type (as named in macaque_adata.obs["Subclass"]) to highlight.
#       umap_key (str): The key in samap_adata.obsm that holds the UMAP coordinates.
#     Returns:
#       fig: The matplotlib figure with the plotted UMAP.
#     """
#     # Create the base plot and plot all cells in a light gray for context
#     fig, ax = plt.subplots(figsize=(8, 6))
#     coords = samap_adata.obsm[umap_key]
#     ax.scatter(coords[:, 0], coords[:, 1],
#                s=3, c='lightgray', alpha=0.4)
    
#     # Make sure the integrated data has a "species" column:
#     if 'species' not in samap_adata.obs.columns:
#         raise ValueError("The integrated SAMap data does not have a 'species' column to distinguish mouse vs. macaque.")
    
#     # -----------------------
#     # Handle Mouse Cells
#     # -----------------------
#     # Create a boolean mask to select mouse cells in the integrated data.
#     mouse_mask = samap_adata.obs['species'] == 'mouse'
#     mouse_cell_names_in_samap = samap_adata.obs_names[mouse_mask]
    
#     # From the mouse-specific AnnData, select cell names with the specified cell type.
#     # (Assuming the cell names in mouse_adata.obs.index match those in samap_adata.obs_names)
#     mouse_highlight_ids = mouse_adata.obs.index[mouse_adata.obs["Supertype"] == mouse_cell_type]
    
#     # Take the intersection: only mouse cells that are in the integrated dataset
#     mouse_ids_to_highlight = np.intersect1d(mouse_cell_names_in_samap, mouse_highlight_ids)
    
#     # Build a boolean mask for the integrated data where the cell names are in the intersection.
#     mouse_highlight_mask = samap_adata.obs_names.isin(mouse_ids_to_highlight)
    
#     # Plot these mouse cells in blue.
#     coords_mouse = samap_adata.obsm[umap_key][mouse_highlight_mask]
#     ax.scatter(coords_mouse[:, 0], coords_mouse[:, 1],
#                s=3, color='blue', label=f"Mouse: {mouse_cell_type}", alpha=0.8)
    
#     # -----------------------
#     # Handle Macaque Cells
#     # -----------------------
#     macaque_mask = samap_adata.obs['species'] == 'macaque'
#     macaque_cell_names_in_samap = samap_adata.obs_names[macaque_mask]
    
#     macaque_highlight_ids = macaque_adata.obs.index[macaque_adata.obs["Subclass"] == macaque_cell_type]
    
#     macaque_ids_to_highlight = np.intersect1d(macaque_cell_names_in_samap, macaque_highlight_ids)
#     macaque_highlight_mask = samap_adata.obs_names.isin(macaque_ids_to_highlight)
    
#     # Plot these macaque cells in red.
#     coords_macaque = samap_adata.obsm[umap_key][macaque_highlight_mask]
#     ax.scatter(coords_macaque[:, 0], coords_macaque[:, 1],
#                s=3, color='red', label=f"Macaque: {macaque_cell_type}", alpha=0.8)
    
#     # Add legend and labels.
#     ax.legend(title="Highlighted Cell Types")
#     ax.set_title("SAMap Combined UMAP Projection with Highlighted Cell Types")
#     ax.set_xlabel("UMAP 1")
#     ax.set_ylabel("UMAP 2")
#     plt.tight_layout()
    
#     return fig

# # -----------------------------
# # Call the function with desired cell types.
# # Replace "desired_mouse_type" and "desired_macaque_type" with the actual cell type names.
# fig = plot_highlighted_umap(samap_adata, mouse_adata, macaque_adata,
#                             mouse_cell_type="desired_mouse_type",
#                             macaque_cell_type="desired_macaque_type", 
#                             umap_key=umap_key)

# # Save the resulting figure.
# fig.savefig("/user/work/pr21872/Homologs/samap_directory/graphs/highlighted_umap_proj.png", 
#             dpi=600, bbox_inches="tight")
