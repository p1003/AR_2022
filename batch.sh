#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 9
#SBATCH --time=00:00:39
#SBATCH --partition=plgrid
#SBATCH -A plgccbmc11-cpu

module load scipy-bundle/2021.10-intel-2021b

export SLURM_OVERLAP=1

SUBMIT_DIR=$(pwd)
WORKSPACE=${SCRATCH}/${SLURM_JOB_ID}
mkdir -p ${WORKSPACE}
cd ${WORKSPACE}

cp ${HOME}/ar/main.py ${WORKSPACE}

mpiexec python3 main.py

cp *.npy ${HOME}/ar

cd ${HOME}
rm -r ${WORKSPACE}