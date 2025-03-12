import os

# Set the desired working directory
new_directory = 'D:'
os.chdir(new_directory)

import h5py

# Construct the path to the HDF5 file
file_path = r"C:\Users\fjpgr\Downloads\Technical Project\expression_matrix.hdf5"

# Open the file using the full path
with h5py.File(file_path, 'r') as file:
    # Access the 'data' group and 'samples' dataset
    data_group = file['data']
    samples_dataset = data_group['samples']
    
    # Convert the dataset to a list
    sample_names = list(samples_dataset[:])[::10]  # Slicing to keep every 10th item

# Define the path for the output .txt file
output_file_path = r"C:\Users\fjpgr\Downloads\Technical Project\mouseSamplesRepresentation.txt"

# Save the list as a .txt file, one name per line
with open(output_file_path, 'w') as f:
    for name in sample_names:
        f.write(f"{name}\n")

print(f"Sample names saved to {output_file_path}")