#!/bin/bash

## Presets
#SBATCH --job-name=chain_dynamics

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=13:00:00
#SBATCH --ntasks=20
#SBATCH --mem=60G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python run_spiking_sims.py

deactivate

## Exit
exit 0
