#!/usr/bin/env bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=3month_DATATOP_aggregate
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=achille.fourtoy@scai.fraunhofer.de

set -euo pipefail

script_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
script_dir="$(cd "${script_dir}" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
models_dir="${repo_root}/MultiNSDEs/Causal/models"

if [[ ! -f "${script_dir}/config1.sh" ]]; then
    echo "Could not find config file: ${script_dir}/config1.sh" >&2
    exit 1
fi

source "${script_dir}/config1.sh"

train_folds_=("${train_folds[@]}")
if [[ ${#train_folds_[@]} -eq 0 ]]; then
    train_folds_=("$train_fold")
fi

for fold in "${train_folds_[@]}"; do
    raw_dir="${save_path}${dataset_}/${name}/Fold${fold}/samples/Val_Imgs_Sampling_Prior/Raw_Output"
    if [[ ! -d "${raw_dir}" ]]; then
        echo "Missing fold output directory: ${raw_dir}" >&2
        exit 1
    fi

    shopt -s nullglob
    risk_files=("${raw_dir}"/Risk_TE_*EP*.csv)
    baseline_files=("${raw_dir}"/Baseline_Risk_TE_EP*.csv)
    shopt -u nullglob

    if (( ${#risk_files[@]} == 0 || ${#baseline_files[@]} == 0 )); then
        echo "Missing Risk_TE or Baseline_Risk_TE files for Fold${fold} in ${raw_dir}" >&2
        exit 1
    fi

done

ate_extra_args=()
if [[ -n "${survival_target_time:-}" ]]; then
    ate_extra_args+=(--survival_target_time="$survival_target_time")
fi

cd "${models_dir}"
python get_ATE.py --exp_name="$name" --dataset="$dataset_" --save_path="$save_path" "${ate_extra_args[@]}"

echo "Aggregation complete: ${save_path}${dataset_}/${name}/ATE_Output.csv"
