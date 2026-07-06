#!/usr/bin/env bash

#SBATCH --partition=medium
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --job-name=OR_ATE
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=achille.fourtoy@scai.fraunhofer.de

# Dataset
dataset_=DATATOP_Causal
data_fname='original_causal_DATATOP_nodeath_static.csv'
ps_types_fname='original_causal_OR_PS_data_DATATOP_types.csv'
do_types_fname='original_causal_OR_DO_data_DATATOP_types.csv'
timetoevent_fname='collapsed_time_to_event_3month.csv'
survival_target='RMST'
survival_target_time=4
csf_num_trees=2000
train_dir=/home/afourtoy/Datasets/collapsed_time_points/3month_extra_per_visit/
save_path=/home/afourtoy/Documents/Development_NSDE/models/
calibrate_pi_model=1
calibrate_g_model=1

name=OR_ATEmodels
cd ../../../MultiNSDEs/Causal/OR_ATEmodels/

extra_args=()
if [[ -n "${survival_target_time:-}" ]]; then
	extra_args+=(--survival_target_time="$survival_target_time")
fi

python main.py \
	--exp_name=$name \
	--dataset=$dataset_ \
	--calibrate_pi_model=$calibrate_pi_model \
	--calibrate_g_model=$calibrate_g_model \
	--data_fname=$data_fname \
	--ps_types_fname=$ps_types_fname \
	--do_types_fname=$do_types_fname \
	--timetoevent_fname=$timetoevent_fname \
	--survival_target=$survival_target \
	--csf_num_trees=$csf_num_trees \
	--save_path=$save_path \
	--train_dir=$train_dir \
	"${extra_args[@]}"
