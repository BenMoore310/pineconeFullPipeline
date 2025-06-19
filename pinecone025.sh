#!/bin/bash
#SBATCH --export=ALL # export all environment variables to the batch job
#SBATCH -D . # set working directory to .
#SBATCH -p pq # submit to the parallel queue
#SBATCH --time=60:00:00 # maximum walltime for the job
#SBATCH -A Research_Project-T124701 # research project to submit under
#SBATCH --nodes=2 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=bm424@exeter.ac.uk # email address

#load software modules
module use /lustre/shared/isca_compute/modules/all
module load OpenFOAM/v2406-foss-2023a
#module load OpenFOAM/v1912-foss-2019b
source $FOAM_BASH



#code to run
./allRun
