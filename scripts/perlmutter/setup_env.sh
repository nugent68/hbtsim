#!/bin/bash
# One-time setup of the jax-gpu-env conda environment on a Perlmutter
# login node:
#
#     git clone https://github.com/nugent68/binary.git ~/binary
#     bash ~/binary/scripts/perlmutter/setup_env.sh
#
# Per current NERSC guidance the pip "jax[cuda12]" wheels bundle their own
# CUDA/cuDNN libraries: do NOT module load cudatoolkit/cudnn/nccl -- their
# LD_LIBRARY_PATH entries can shadow the bundled libraries and break JAX
# initialization.  (Fallback if wheel/driver drift ever bites: NERSC's
# preferred container route, Shifter with --image=nvcr.io/nvidia/jax:XX.).
#
# The env lives in $HOME (~/.conda/envs), which is NOT subject to the
# $SCRATCH 8-week purge.
set -euo pipefail

module load python
conda create -y -n jax-gpu-env python=3.12 pip
source activate jax-gpu-env 2>/dev/null || conda activate jax-gpu-env

pip install --upgrade "jax[cuda12]" numpy scipy matplotlib pytest
pip install -e "$HOME/binary"

echo
echo "Environment ready.  Sanity check (login nodes have no GPUs, so"
echo "this prints CpuDevice here; CudaDevice inside a GPU allocation):"
python -c "import jax; print(jax.__version__, jax.devices())"
