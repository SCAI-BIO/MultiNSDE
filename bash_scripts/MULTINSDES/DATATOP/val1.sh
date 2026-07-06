#!/usr/bin/env bash

source ./config1.sh

dirname=$save_path$dataset_/$name/Fold${train_fold}

#saving git commit from repository that does the validation
echo $dirname

runfiles_name=$dirname/run_files
git rev-parse HEAD > ${runfiles_name}/git_info_val.txt
cp ./val1.sh ${runfiles_name}/

#getting the epoch of the last model that was ran
epochh=$(cd ${dirname}/models && ls -t Ckpt_*.pth | head -1 | sed 's/[^0-9]*\([0-9]*\).*/\1/')

echo "Unless from_best=1, this is the epoch that will be used for validation: $epochh"

echo $epochh > $dirname/models/epoch_number.txt

cd ../../../MultiNSDEs/Prognosis/models/

nruns=30
Val_Scenario_=(0 1 2 3 4 5)
# To be sure we validate it using the correct class type
real_longtypes_fname='original_long_main_file_types_ordinal.csv'
real_statictypes_fname='original_static_types_ordinal.csv'

samples_from="Sampling_PSD"
for Val_Scenario in ${Val_Scenario_[@]}; do
    python main.py --mode=val --from_best=1 --extrapolation=$extrapolation --train_fold=$train_fold --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --type_hivae=$type_hivae --num_lenc=$num_enc --type_lenc=$type_lenc --rev_lenc=$rev_lenc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_rnn_enc=$nl_rnn_enc --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --rhs_ldim=$rhs_ldim --long_ldim=$long_ldim --type_dec=$type_ldec --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --GPU=$gpu --save_path=$save_path --train_dir=$train_dir --clipping=$clipping --real_longtypes_fname=$real_longtypes_fname --real_statictypes_fname=$real_statictypes_fname
    python get_raw_data.py --train_fold=$train_fold --extrapolation=$extrapolation  --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --from_best=1 --dataset=$dataset_ --save_path=$save_path --train_dir=$train_dir

    if [[ $Val_Scenario -eq 2 || $Val_Scenario -eq 3 || $Val_Scenario -eq 4 || $Val_Scenario -eq 5 ]]; then
        python get_scores_counterfactual.py --train_fold=$train_fold  --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --from_best=1 --dataset=$dataset_ --save_path=$save_path --train_dir=$train_dir
    fi

    if [[ $Val_Scenario -eq 1 || $Val_Scenario -eq 0 ]]; then
        python Metrics/metrics.py --extrapolation=$extrapolation --exp_name=$name --train_fold=$train_fold --Val_Scenario=$Val_Scenario --val_data_type=$samples_from --from_best=1 --save_path=$save_path --train_dir=$train_dir --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$real_longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$real_statictypes_fname --dosesdata_fname=$dosesdata_fname
        python get_scores_long.py --train_fold=$train_fold --extrapolation=$extrapolation --val_data_type=$samples_from --Val_Scenario=$Val_Scenario --nruns_ppd=$nruns --exp_name=$name --from_best=1 --dataset=$dataset_ --save_path=$save_path --train_dir=$train_dir
    fi
done