import h5py

with h5py.File(r"D:\Macaque\txtFiles\expression_matrix.hdf5", 'r') as f:
    data_group = f["data"]
    print("Keys in 'data' group:", list(data_group.keys()))
    
    for key in data_group.keys():
        ds = data_group[key]
        print(f"\nKey: {key}")
        
        # Check if it's a dataset and print some basic info
        if isinstance(ds, h5py.Dataset):
            print(" - Dataset shape:", ds.shape)
            print(" - Dataset dtype:", ds.dtype)
            # Print a sample of the data. Adjust slicing as needed.
            # If it's an array, you can print the first few elements.
            if ds.ndim == 1:
                # For 1D data, print first 10 entries or the whole dataset if it's small.
                sample = ds[0:10] if ds.shape[0] > 10 else ds[()]
                print(" - Sample data:", sample)
            elif ds.ndim == 2:
                # For 2D data, print the first 3 rows and columns.
                sample = ds[0:3, 0:3] if ds.shape[0] > 3 and ds.shape[1] > 3 else ds[()]
                print(" - Sample data:\n", sample)
            else:
                print(" - Data has more than 2 dimensions; consider inspecting specific slices.")
        else:
            print(" - This key is not a dataset.")
