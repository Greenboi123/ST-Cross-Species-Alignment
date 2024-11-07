import os

# Set the desired working directory
new_directory = 'D:'
os.chdir(new_directory)

import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import scipy.sparse

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy.sparse

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
    
    # Check for row and column names in dimnames
    dimnames = rds_data.slots['Dimnames']
    row_names = dimnames[0] if dimnames[0] != ro.NULL else None
    col_names = dimnames[1] if dimnames[1] != ro.NULL else None
    
    # Display row and column names if they exist
    # Check if row and column names exist, then save them to files
    # if row_names:
    #     with open("row_names.txt", "w") as row_file:
    #         for name in row_names:
    #             row_file.write(f"{name}\n")
    #     print("Row names saved to row_names.txt.")
    # else:
    #     print("No row names found.")

    if col_names:
        with open("col_names.txt", "w") as col_file:
            for name in col_names[::10]:  # Only keep every 10th name:
                col_file.write(f"{name}\n")
        print("Column names saved to col_names.txt.")
    else:
        print("No column names found.")

        
    # Display the structure of the matrix
    print(f"Matrix shape: {data.shape}")
    print(f"Non-zero elements: {data.nnz}")
else:
    print("The object is not a dgCMatrix.")
