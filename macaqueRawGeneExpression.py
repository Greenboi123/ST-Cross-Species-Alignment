
import os

# Set the desired working directory
new_directory = 'D:'
os.chdir(new_directory)

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy.sparse
import numpy as np

# Import necessary R packages
base = importr('base')
Matrix = importr('Matrix')

# Load the RDS file
rds_file_path = r"C:\Users\fjpgr\Downloads\Technical Project\snRNA.sparseMatrix_Monkey1.counts.rds"
rds_data = base.readRDS(rds_file_path)

# Check if the object is a dgCMatrix
if ro.r['inherits'](rds_data, "dgCMatrix")[0]:  # Check if it's dgCMatrix
    # Access slots using the @ operator
    data = scipy.sparse.csc_matrix(
        (rds_data.slots['x'], rds_data.slots['i'], rds_data.slots['p']),
        shape=(rds_data.slots['Dim'][0], rds_data.slots['Dim'][1])
    )
    
    # Extract a 5x5 subset and convert it to dense format
    dense_submatrix = data[:5, :5].todense()
    print("5x5 Submatrix (Dense Format):")
    print(np.array(dense_submatrix))

    # Check for row and column names in dimnames
    dimnames = rds_data.slots['Dimnames']
    row_names = dimnames[0] if dimnames[0] != ro.NULL else None
    col_names = dimnames[1] if dimnames[1] != ro.NULL else None

    # Optionally display row and column names if they exist
    if row_names and col_names:
        print("Row names:", row_names[:5])
        print("Column names:", col_names[:5])
else:
    print("The object is not a dgCMatrix.")



# ----- Metadata matrix ----- #
# Load the RDS file
# file_path = r"C:\Users\fjpgr\Downloads\Technical Project\snRNA.metadata.2monkeys.rds" 
# rds_data = base.readRDS(file_path)

# # Check the type of the R object
# print("R object type:", type(rds_data))

# # Print the structure of the R object
# str_func = ro.r['str']
# str_func(rds_data)

