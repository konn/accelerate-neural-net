#!/usr/bin/env bash
#PBS -q SQUID
#PBS --group=hp220285
#PBS -l elapstim_req=1:00:00,cpunum_job=32,gpunum_job=1,memsz_job=4GB

set -euxo pipefail

module load BaseCPU BaseGPU
WORK_DIR="$(readlink -f /sqfs/work/hp220285/z6b161)"
export SINGULARITY_BIND="${WORK_DIR},${PBS_O_WORKDIR}"
time singularity run --nv "/sqfs/work/hp220285/z6b161/accelerate-singularity-sandbox/devenv.sif" cabal v2-exec -- mnist +RTS -s
