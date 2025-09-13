#!/usr/bin/env bash
#SBATCH --job-name=Mapping_Genes    # A descriptive name.
#SBATCH --time=26:00:00             # Walltime
#SBATCH --nodes=1                   # Use 1 compute node.
#SBATCH --tasks-per-node=1          # 1 task per node.
#SBATCH --cpus-per-task=32          # Reserve 32 CPUs for multi-threading.
#SBATCH --mem=64G                   # Total memory for the job per node.
#SBATCH --account=semt031404        # Account details
#SBATCH -o %x_out_%j.txt            # Output file (job name and ID will be substituted)
#SBATCH -e %x_err_%j.txt            # Error file

# Load your HPC environment
source ~/initMamba.sh
mamba activate SAMap

# Change to the directory from which you submitted the job.
cd "${SLURM_SUBMIT_DIR}"

# Record job details.
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Job is running on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

# Define variables for the BLAST mapping.
# Update these paths to point to your actual FASTA files.
transcriptome1="${WORK}/Homologs/MacaqueFASTA/Macaca_fascicularis.Macaca_fascicularis_6.0.cdna.all.fa"
transcriptome2="${WORK}/Homologs/MouseFASTA/Mus_musculus.GRCm39.cdna.all.fa"
# Specify whether each file is nucleotide (nucl) or protein (prot)
type1="nucl"
type2="nucl"
# Two-character species identifiers (these will label your output directory).
id1="mq"  # For macaque
id2="mm"  # For mouse
# Set the number of threads (adjust as needed).
threads=32

# Construct the command to run map_genes.sh.
cmd="bash map_genes.sh --tr1 ${transcriptome1} --t1 ${type1} --n1 ${id1} --tr2 ${transcriptome2} --t2 ${type2} --n2 ${id2} --threads ${threads}"

# Print the command for sanity check.
printf "Executing command:\n%s\n\n" "${cmd}"

# Execute the command using srun.
time ${cmd}

printf "\n\n"
echo "Ended on: $(date)"
