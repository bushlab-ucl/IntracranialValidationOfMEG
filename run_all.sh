#!/usr/bin/env bash
# Reproduce every result and figure panel, from the public release.
#
# Usage: ./run_all.sh [preproc|analysis|figures]   (default: all three)
#
# Needs config.yaml (see config.example.yaml), the Python environment
# (environment.yml) and, for the preprocessing, FSL (FSLDIR set).
#
# Approximate run times on a 10-core laptop: preproc 1-2 h, analysis
# 30-40 min, figures ~1 min.

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Single-threaded linear algebra per process: the scripts parallelise over
# sessions, and multi-threaded BLAS on top of that makes some analyses (e.g.
# the CCA) an order of magnitude slower.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1

PREPROC=(
    01_seeg        # SEEG: bipolar montage, bad segments, IED windows
    02_meg         # MEG: filtering, bad segments and channels
    03_surfaces    # RHINO surfaces and MNI registration (FSL)
    04_coreg_bem   # coregistration, BEM
    05_ve          # virtual electrodes at the contacts (broadband, per band)
    06_null_ves    # random-location and matched-depth null VEs
)
ANALYSIS=(
    cca_sensor           # Fig 1D
    cca_ve               # Fig S1
    psd                  # Fig 2A, S2
    band_power           # Fig 2B
    band_power_null      # Fig S3
    amp_modulation       # Fig 3
    amp_modulation_null  # Fig S4
    hipp_power           # Fig 4A
    hipp_power_null      # Fig S6
    hipp_amp_modulation  # Fig 4B
    hipp_cca_sensor      # Fig S5
    aec                  # Fig S7
)

run() {
    echo "[$(date +%H:%M:%S)] $1"
    python "$1"
}

stage="${1:-all}"
if [[ ! "$stage" =~ ^(all|preproc|analysis|figures)$ ]]; then
    echo "usage: $0 [preproc|analysis|figures]" >&2
    exit 1
fi
if [[ ! -f config.yaml ]]; then
    echo "config.yaml not found: copy config.example.yaml and edit it" >&2
    exit 1
fi

if [[ "$stage" == all || "$stage" == preproc ]]; then
    : "${FSLDIR:?FSLDIR is not set (FSL is needed for the preprocessing)}"
    for s in "${PREPROC[@]}"; do run "scripts/preproc/$s.py"; done
fi
if [[ "$stage" == all || "$stage" == analysis ]]; then
    # The analyses use up to two threads each
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
           VECLIB_MAXIMUM_THREADS=2
    for s in "${ANALYSIS[@]}"; do run "scripts/analysis/$s.py"; done
fi
if [[ "$stage" == all || "$stage" == figures ]]; then
    for f in scripts/figures/*.py; do run "$f"; done
fi
echo "[$(date +%H:%M:%S)] done"
