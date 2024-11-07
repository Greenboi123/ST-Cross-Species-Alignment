import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import io
import sys

# Load R base package
base = importr('base')

# Load the RDS file
file_path = r"C:\Users\fjpgr\Downloads\Technical Project\snRNA.metadata.2monkeys.rds"  # Replace with your file path
rds_data = base.readRDS(file_path)

# Define the output file path
output_file_path = r"C:\Users\fjpgr\Downloads\Technical Project\r_object_structure.txt"

# Redirect the output of str() to a file
str_func = ro.r['str']
with open(output_file_path, 'w') as f:
    # Redirect stdout to capture the output of str_func
    old_stdout = sys.stdout
    sys.stdout = f
    str_func(rds_data)
    sys.stdout = old_stdout

print(f"R object structure saved to {output_file_path}")
