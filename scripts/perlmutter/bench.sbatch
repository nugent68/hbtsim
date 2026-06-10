#!/bin/bash
# Spectral |V|^2 benchmark on one Perlmutter A100:
#     sbatch ~/binary/scripts/perlmutter/bench.sbatch
# Find your GPU account (the _g suffix) with:
#     sacctmgr show assoc user=$USER format=account%20 -n -P | sort -u
#SBATCH -A m2218_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:20:00
#SBATCH -J hbtsim-bench
#SBATCH -o /pscratch/sd/n/nugent/hbtsim-bench-%j.out

module load python
source activate jax-gpu-env 2>/dev/null || conda activate jax-gpu-env
cd "$HOME/binary"
# don't let XLA preallocate 75% of the GPU: on a shared GPU (login node,
# or shared QOS neighbors) that collides with other users' allocations
# and starves the cuFFT workspace
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== GPU ==="
srun python scripts/bench_spectral.py --channels 320

echo
echo "=== CPU reference (same node) ==="
JAX_PLATFORMS=cpu python scripts/bench_spectral.py --channels 320 --chunk 4 --repeat 1
