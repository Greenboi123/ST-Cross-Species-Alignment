#!/usr/bin/env python3
import pandas as pd
import anndata

# Load your AnnData object:
adata = anndata.read_h5ad("/user/work/pr21872/Homologs/samap_directory/sam2_macaque.h5ad")

# Load the CSV file with cell type annotations.
# Assume the CSV has the cell names as its index and a column named "Cluster"
cell_types = pd.read_csv("/user/work/pr21872/Homologs/rds2hda5/macaqueMetadata.csv", index_col=0)

# # Check that the indices match. For example, you can print:
# print(adata.obs.index[:5])
# print(cell_types.index[:5])

# If the cell order might be different, you can reindex to ensure alignment:
adata.obs["Subclass"] = cell_types["FullName"].reindex(adata.obs.index)

# # Optionally, if there are cells in adata without a corresponding entry in the CSV,
# # those will become NaN. You can handle them as needed (e.g., fill with "Unknown"):
# adata.obs["Cluster"] = adata.obs["Cluster"].fillna("Unknown")
adata.write_h5ad("/user/work/pr21872/Homologs/samap_directory/sam2_macaque_w_celltypes.h5ad")