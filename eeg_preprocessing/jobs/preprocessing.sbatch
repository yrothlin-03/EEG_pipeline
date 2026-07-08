#!/bin/bash
#SBATCH -p CPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocessing
#SBATCH --chdir=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS
#SBATCH --output=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/eeg_preprocessing/logs/preprocessing/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/eeg_preprocessing/logs/preprocessing/%x_%j.err

set -euo pipefail

DATASET="${1:-${DATASET:-}}"

PROJECT_ROOT="/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS"
PREPROCESSING_DIR="${PROJECT_ROOT}/eeg_preprocessing"
CONFIG_PATH="${PREPROCESSING_DIR}/configs/preprocessing.yaml"
SLURM_LOG_DIR="${PREPROCESSING_DIR}/logs/preprocessing"

mkdir -p "${SLURM_LOG_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] Starting ${SLURM_JOB_NAME:-preprocessing} on ${HOSTNAME}"
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Dataset: ${DATASET:-<from config>}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-8}"

cd "${PREPROCESSING_DIR}"

PYTHON_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${DATASET}" ]]; then
    PYTHON_ARGS+=(--dataset "${DATASET}")
fi

micromamba run -n rech202 python main_preprocessing.py "${PYTHON_ARGS[@]}"

echo "[$(date)] Finished ${SLURM_JOB_NAME:-preprocessing}"
