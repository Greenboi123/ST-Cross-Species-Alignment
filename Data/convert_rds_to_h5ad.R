#!/usr/bin/env Rscript

# Set CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Set personal library path (using R_LIBS_USER, or default to ~/R/library if not set)
personal_lib <- Sys.getenv("R_LIBS_USER")
if (personal_lib == "") {
  personal_lib <- "~/R/library"
}
if (!dir.exists(personal_lib)) {
  dir.create(personal_lib, recursive = TRUE)
}
.libPaths(c(normalizePath(personal_lib), .libPaths()))

# Ensure BiocManager is installed in your personal library
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", lib = personal_lib)

# Install packages only if not already installed
if (!requireNamespace("zellkonverter", quietly = TRUE))
    BiocManager::install("zellkonverter", lib = personal_lib)

if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    BiocManager::install("SingleCellExperiment", lib = personal_lib)

# Load required packages
library(zellkonverter)
library(SingleCellExperiment)

# Define input and output file paths
input_file <- "/user/work/pr21872/Homologs/rds2hda5/snRNA.sparseMatrix_Monkey1.counts.rds"    # update this with the full path to your .rds file
output_file <- "snRNAsparseMatrix_Monkey1.h5ad"        # desired output file name

# Read the RDS file
obj <- readRDS(input_file)

# Check if the object is a SingleCellExperiment, if not, wrap it
if (!inherits(obj, "SingleCellExperiment")) {
  # Assuming 'obj' is a dgCMatrix, we wrap it as the 'counts' assay.
  sce <- SingleCellExperiment(assays = list(counts = obj))
} else {
  sce <- obj
}

# Write out as an h5ad file
writeH5AD(sce, output_file)
