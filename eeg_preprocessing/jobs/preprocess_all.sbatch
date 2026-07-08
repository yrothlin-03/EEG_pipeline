#!/bin/bash
#SBATCH -p CPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --job-name=preprocess_all
#SBATCH --chdir=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS
#SBATCH --output=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/eeg_preprocessing/logs/preprocessing/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/eeg_preprocessing/logs/preprocessing/%x_%j.err

set -uo pipefail

# Datasets to preprocess sequentially.
# Override from the command line, e.g.:
#   sbatch preprocess_all.sbatch FACED SEEDV PHYSIONETMI
# or via the DATASETS env var (space-separated):
#   DATASETS="FACED SEEDV" sbatch preprocess_all.sbatch
DEFAULT_DATASETS=(FACED SEEDV PHYSIONETMI SLEEPEDF TUEG TUAB BCI2A BCI2020 KARAONE)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
elif [[ -n "${DATASETS:-}" ]]; then
    # shellcheck disable=SC2206
    DATASETS=(${DATASETS})
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

PROJECT_ROOT="/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS"
PREPROCESSING_DIR="${PROJECT_ROOT}/eeg_preprocessing"
CONFIG_PATH="${PREPROCESSING_DIR}/configs/preprocessing.yaml"
SLURM_LOG_DIR="${PREPROCESSING_DIR}/logs/preprocessing"

mkdir -p "${SLURM_LOG_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] Starting ${SLURM_JOB_NAME:-preprocess_all} on ${HOSTNAME}"
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Datasets: ${DATASETS[*]}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-8}"

cd "${PREPROCESSING_DIR}"

failed=()
succeeded=()

for dataset in "${DATASETS[@]}"; do
    echo "================================================================"
    echo "[$(date)] >>> Preprocessing ${dataset}"
    echo "================================================================"

    if micromamba run -n rech202 python main_preprocessing.py \
        --config "${CONFIG_PATH}" \
        --dataset "${dataset}"; then
        echo "[$(date)] <<< Finished ${dataset} (OK)"
        succeeded+=("${dataset}")
    else
        status=$?
        echo "[$(date)] <<< FAILED ${dataset} (exit ${status})" >&2
        failed+=("${dataset}")
    fi
done

echo "================================================================"
echo "[$(date)] Finished ${SLURM_JOB_NAME:-preprocess_all}"
echo "Succeeded (${#succeeded[@]}): ${succeeded[*]:-none}"
echo "Failed    (${#failed[@]}): ${failed[*]:-none}"

# Non-zero exit if any dataset failed, so Slurm marks the job accordingly.
if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
