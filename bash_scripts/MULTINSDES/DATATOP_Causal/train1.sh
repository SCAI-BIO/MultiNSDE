#!/usr/bin/env bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --job-name=3month_causal_DATATOP
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=achille.fourtoy@scai.fraunhofer.de
#SBATCH --array=0-2

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

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
	if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#train_folds_[@]} )); then
		echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} for train_folds size ${#train_folds_[@]}" >&2
		exit 1
	fi
	current_train_fold="${train_folds_[${SLURM_ARRAY_TASK_ID}]}"
else
	current_train_fold="${train_folds_[0]}"
	if (( ${#train_folds_[@]} > 1 )); then
		echo "SLURM_ARRAY_TASK_ID not set; running only first fold (${current_train_fold})."
	fi
fi

or_samples_dir=${save_path}${dataset_}/${or_model_exp_name:-OR_ATEmodels}/samples
required_files=(
	"${train_dir}${dataset_}/${dosesdata_fname}"
	"${train_dir}${dataset_}/${longdata_fname}"
	"${train_dir}${dataset_}/${staticdata_fname}"
	"${or_samples_dir}/${ps_scores_fname}"
	"${or_samples_dir}/${do_scores_fname}"
)

if [[ "$time_to_event" == "1" ]]; then
	required_files+=("${train_dir}${dataset_}/${timetoevent_fname}")
fi

for required_file in "${required_files[@]}"; do
	if [[ ! -f "$required_file" ]]; then
		echo "Missing required input: $required_file" >&2
		exit 1
	fi
done

nruns=30
samples_from="Sampling_Prior"
Val_Scenario_=(1)

ate_extra_args=()
if [[ -n "${survival_target_time:-}" ]]; then
	ate_extra_args+=(--survival_target_time="$survival_target_time")
fi

survival_args=()
if [[ -n "${survival_target_time:-}" ]]; then
	survival_args+=(--survival_target_time="$survival_target_time")
fi

dirname=$save_path$dataset_/$name/Fold${current_train_fold}
mkdir -p "${dirname}"

runfiles_name=$dirname/run_files
mkdir -p "${runfiles_name}"

git -C "$repo_root" rev-parse HEAD > "${runfiles_name}/git_info_train.txt"
cp "${script_dir}/config1.sh" "${runfiles_name}/"
cp "${script_dir}/train1.sh" "${runfiles_name}/"

cd "$models_dir"
python main.py --exp_name=$name --train_fold=$current_train_fold --type_hivae=$type_hivae --num_lenc=$num_enc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --long_ldim=$long_ldim --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --epoch_init=$epoch_init --num_epochs=$epochs --patience=$patience --inv_ic_loss=$inv_ic_loss --lambda_RecLong=$l_reclong --lambda_KLLong=$l_kllong --lambda_IC=$l_ic --lambda_TE=$l_te --lambda_OR_TRT=$l_or_trt --lambda_OR_DO=$l_or_do --lambda_RecStat=$l_recstat --comb_long_loss=$comb_longloss --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --timetoevent_fname=$timetoevent_fname --time_to_event=$time_to_event --ps_scores_fname=$ps_scores_fname --do_scores_fname=$do_scores_fname --GPU=$gpu --batch_size=$bs --lr=$lr --save_path=$save_path --train_dir=$train_dir --save_freq=$s_freq --print_freq=$p_freq --clipping=$clipping "${survival_args[@]}"

epochh=$(cd "${dirname}/models" && ls -t Ckpt_*.pth | head -1 | sed 's/[^0-9]*\([0-9]*\).*/\1/')

echo "Unless from_best=1, this is the epoch that will be used for validation: $epochh"

echo "$epochh" > "$dirname/models/epoch_number.txt"
git -C "$repo_root" rev-parse HEAD > "${runfiles_name}/git_info_val.txt"

for Val_Scenario in ${Val_Scenario_[@]}; do
	python main.py --mode=val --from_best=1 --train_fold=$current_train_fold --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --type_hivae=$type_hivae --num_lenc=$num_enc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --long_ldim=$long_ldim --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --time_to_event=$time_to_event --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --timetoevent_fname=$timetoevent_fname --ps_scores_fname=$ps_scores_fname --do_scores_fname=$do_scores_fname --GPU=$gpu --save_path=$save_path --train_dir=$train_dir --clipping=$clipping "${survival_args[@]}"
	python get_raw_data.py --train_fold=$current_train_fold --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --from_best=1 --dataset=$dataset_ --save_path=$save_path --train_dir=$train_dir --time_to_event=$time_to_event
done

echo "Per-fold job complete for Fold${current_train_fold}."
echo "Run final aggregation once after all array tasks finish:"
echo "cd ${models_dir} && sbatch aggregate1.sh"