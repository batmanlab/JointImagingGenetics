#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --exclusive


echo "$@"
"$@"
