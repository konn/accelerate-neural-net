#!/usr/bin/env bash
#PBS -q SQUID
#PBS --group=hp220285
#PBS -l elapstim_req=1:00:00,cpunum_job=16,gpunum_job=1,memsz_job=4GB

set -euxo pipefail

module load BaseCPU BaseGPU
cd "${PBS_O_WORKDIR}" 
WORK_DIR="$(readlink -f /sqfs/work/hp220285/z6b161)"
export SINGULARITY_BIND="${WORK_DIR},${PBS_O_WORKDIR}"
time singularity run --nv "/sqfs/work/hp220285/z6b161/accelerate-singularity-sandbox/devenv.sif" /sqfs/work/hp220285/z6b161/accelerate-singularity-sandbox/accelerate-playground/dist-newstyle/build/x86_64-linux/ghc-8.10.7/accelerate-playground-0.1.0.0/x/mnist/build/mnist/mnist --backend CPU +RTS -s
