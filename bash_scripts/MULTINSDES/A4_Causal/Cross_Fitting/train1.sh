#!/usr/bin/env bash

#SBATCH --job-name=A4_Causal
#SBATCH --array=0-2

source ./config1.sh
train_folds=(1 2 3)
train_fold=${train_folds[$SLURM_ARRAY_TASK_ID]}
if [[ $train_fold -eq 3 ]]; then
    l_or_trt=0.01
else
    l_or_trt=0.0001
fi

dirname=$save_path$dataset_/$name/Fold${train_fold}
mkdir -p "${dirname}"

runfiles_name=$dirname/run_files
mkdir -p "${runfiles_name}"

git rev-parse HEAD > ${runfiles_name}/git_info_train.txt
cp ./config1.sh ${runfiles_name}/

cd ../../../../MultiNSDEs/Causal/models/
python main.py --exp_name=$name --train_fold=$train_fold --type_hivae=$type_hivae --num_lenc=$num_enc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --long_ldim=$long_ldim --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --epoch_init=$epoch_init --num_epochs=$epochs --patience=$patience --inv_ic_loss=$inv_ic_loss --lambda_RecLong=$l_reclong --lambda_KLLong=$l_kllong --lambda_IC=$l_ic --lambda_OR_TRT=$l_or_trt --lambda_OR_DO=$l_or_do --lambda_RecStat=$l_recstat --comb_long_loss=$comb_longloss --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --ps_scores_fname=$ps_scores_fname --do_scores_fname=$do_scores_fname --GPU=$gpu --batch_size=$bs --lr=$lr --save_path=$save_path --train_dir=$train_dir --save_freq=$s_freq --print_freq=$p_freq --clipping=$clipping

#getting the epoch of the last model that was ran
epochh=$(cd ${dirname}/models && ls -t Ckpt_*.pth | head -1 | sed 's/[^0-9]*\([0-9]*\).*/\1/')

echo "Unless from_best=1, this is the epoch that will be used for validation: $epochh"

echo $epochh > $dirname/models/epoch_number.txt

nruns=30
nruns=2
samples_from="Sampling_Prior"
Val_Scenario_=(1)
for Val_Scenario in ${Val_Scenario_[@]}; do
    python main.py --mode=val --from_best=1 --train_fold=$train_fold --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --type_hivae=$type_hivae --num_lenc=$num_enc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --long_ldim=$long_ldim --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --ps_scores_fname=$ps_scores_fname --do_scores_fname=$do_scores_fname --GPU=$gpu --save_path=$save_path --train_dir=$train_dir --clipping=$clipping
    python get_raw_data.py --train_fold=$train_fold  --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --from_best=1 --dataset=$dataset_ --save_path=$save_path --train_dir=$train_dir
done
python get_ATE.py --exp_name=$name --dataset=$dataset_ --save_path=$save_path