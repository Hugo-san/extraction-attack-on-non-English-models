#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sat Mar 16 2024 20:38:16 GMT+1100

# Partition for the job:
#SBATCH --partition=gpu-a100
#SBATCH -p gpu-a100
# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="Extracting training data from LLMs across languages"

# The project ID which this job should run under:
#SBATCH --account="punim2230"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem-per-gpu=200g

# Use this email address:
#SBATCH --mail-user=ruijieh2@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-6:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load foss/2022a
module load SciPy-bundle/2022.05
module load GCCcore/11.3.0; module load Python/3.10.4

# The job command(s):

python3 extraction_temperature_German.py --N 100000 --batch_size 100 --pretrained_model_name1 "benjamin/gerpt2-large" --pretrained_model_name2 "benjamin/gerpt2" --window_size 50 --best_n 100 --text_len 128  --wet_file "Sampled_German_CC.txt"
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
