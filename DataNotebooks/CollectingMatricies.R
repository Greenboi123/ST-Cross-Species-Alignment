library(data.table)
library(Matrix)

# Define directories for input and output
txt_dir <- "D:\\Macaque\\txtFiles"
rds_dir <- "D:\\Macaque\\cellFiles"

# List all .txt files (with full paths) in the input directory
files <- list.files(txt_dir, pattern = "*.txt", full.names = TRUE)

process_file <- function(file) {
  # 1. Read the current file (assuming tab-delimited) and select only the needed columns
  dt <- fread(file, sep = "\t", 
              select = c("cell_id", "gene", "gene_area", "x", "y", "rx", "ry", "expr"))
  
  # 2. Extract unique metadata for each cell_id
  metadata <- unique(dt[, .(cell_id, gene_area, x, y, rx, ry)])
  
  # 3. Convert gene and cell_id columns to factors for efficient memory usage
  dt[, gene := factor(gene)]
  dt[, cell_id := factor(cell_id)]
  
  # 4. Build the sparse matrix (rows = genes, columns = cell_ids, values = expr)
  sparse_expr <- sparseMatrix(i = as.integer(dt$gene),
                              j = as.integer(dt$cell_id),
                              x = dt$expr,
                              dims = c(nlevels(dt$gene), nlevels(dt$cell_id)),
                              dimnames = list(levels(dt$gene), levels(dt$cell_id)))
  
  # 5. Extract the chip value from the file name (assumes filenames like "total_gene_T100...")
  chip_value <- sub("^(total_gene_T\\d+).*", "\\1", basename(file))
  
  # 6. Attach metadata as an attribute to the sparse matrix
  attr(sparse_expr, "metadata") <- metadata
  
  # 7. Define the output file path using the chip value
  out_file <- file.path(rds_dir, paste0(chip_value, ".rds"))
  
  # 8. Save the sparse matrix (with metadata attached) to an .rds file
  saveRDS(sparse_expr, file = out_file)
  cat("Processed and saved file:", out_file, "\n")
  
  # 9. Clean up memory after processing each file
  rm(dt, metadata, sparse_expr)
  gc()
}

# Process each file in the list using lapply
lapply(files, process_file)
