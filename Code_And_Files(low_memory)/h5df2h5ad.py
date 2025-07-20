#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

input_file = r"/user/work/pr21872/Homologs/h5df2hda5/expression_matrix.hdf5"
output_file = r"/user/work/pr21872/Homologs/h5df2hda5/CTXHPF_expression_matrix.h5ad"

with h5py.File(input_file, "r") as f:
    data_group = f["data"]
    # Get the counts dataset (assumed to be stored as (genes, samples))
    ds = data_group["counts"]
    genes = data_group["gene"][:]
    samples = data_group["samples"][:]
    
# Decode gene and sample names if they are byte strings
genes = [g.decode("utf-8") if isinstance(g, bytes) else g for g in genes]
samples = [s.decode("utf-8") if isinstance(s, bytes) else s for s in samples]

# Use the lengths from the gene and sample arrays
n_genes = len(genes)
n_samples = len(samples)

# We need rows as samples and columns as genes,
# so our final shape should be (n_samples, n_genes)
# We'll read the data row-by-row from the transposed view.
# Instead of transposing a huge dense array, we'll iterate over columns of ds.
# Each "row" in the output corresponds to one column in ds.
out_shape = (n_samples, n_genes)

# Prepare lists for CSR components
data_list = []
indices_list = []
indptr = [0]

with h5py.File(input_file, "r") as f:
    ds = f["data"]["counts"]
    # Loop over each sample (i.e. each column in the original matrix)
    for sample_idx in range(n_samples):
        # Read one column (all genes for this sample)
        # Using ds[:, sample_idx] returns a 1D numpy array.
        col_data = ds[:, sample_idx]
        # Find indices of nonzero entries
        nonzero_idx = np.nonzero(col_data)[0]
        nonzero_data = col_data[nonzero_idx]
        data_list.extend(nonzero_data)
        indices_list.extend(nonzero_idx)
        indptr.append(len(data_list))

# Create CSR matrix; note: we are constructing the transposed view, so shape is (n_samples, n_genes)
counts_sparse = sparse.csr_matrix((np.array(data_list, dtype=np.int32),
                                   np.array(indices_list, dtype=np.int32),
                                   np.array(indptr, dtype=np.int32)),
                                  shape=out_shape)

# Check that the number of samples and genes match our expectations
assert counts_sparse.shape == (len(samples), len(genes)), (
    f"Mismatch: counts shape {counts_sparse.shape} vs. {len(samples)} samples and {len(genes)} genes"
)

# Create AnnData object with the sparse counts matrix
adata = ad.AnnData(X=counts_sparse,
                   obs=pd.DataFrame(index=samples),
                   var=pd.DataFrame(index=genes))

# Write the AnnData object to an h5ad file
try:
    adata.write_h5ad(output_file)
    print("File written successfully.")
except Exception as e:
    print("Error during file write:", e)
