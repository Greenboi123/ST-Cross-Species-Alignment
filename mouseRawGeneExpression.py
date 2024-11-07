import os


# Set the desired working directory
new_directory = 'D:'
os.chdir(new_directory)

import h5py

# Construct the path to the HDF5 file
file_path = r"C:\Users\fjpgr\Downloads\Technical Project\expression_matrix.hdf5"

# Open the file using the full path
with h5py.File(file_path, 'r') as file:
    # Access the 'counts' dataset within the 'data' group
    counts_dataset = file['data/counts']
    # Extract the top 5x5 matrix
    top_5x5_matrix = counts_dataset[:5, :5]
    print("5x5 Matrix:")
    print(top_5x5_matrix)
    # Access the 'gene' dataset #every single
    shape_dataset = file['data/shape']
    print("Shape data shape:", shape_dataset.shape)
    for name in shape_dataset:
        print(name)
        # Access the 'gene' dataset #every single
    gene_dataset = file['data/gene']
    
    print("Gene data shape:", gene_dataset.shape)
    for name in gene_dataset[:10]:
        print(name)
    



    





























