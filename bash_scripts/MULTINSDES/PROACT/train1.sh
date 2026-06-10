#!/usr/bin/env bash

source ./config1.sh

dirname=$save_path$dataset_/$name/Fold${train_fold}
mkdir -p "${dirname}"

runfiles_name=$dirname/run_files
mkdir -p "${runfiles_name}"

git rev-parse HEAD > ${runfiles_name}/git_info_train.txt
cp ./config1.sh ${runfiles_name}/

cd ../../../MultiNSDEs/Prognosis/models/
python main.py --exp_name=$name --train_fold=$train_fold --extrapolation=$extrapolation --type_hivae=$type_hivae --num_lenc=$num_enc --type_lenc=$type_lenc --rev_lenc=$rev_lenc --long_implayer=$long_impl --mse_head=$mse_head --nlayers_rnn_enc=$nl_rnn_enc --nlayers_mlp_lenc=$nl_mlp_lenc --nhidden_lenc=$nhidden_lenc --act_mean=$act_mean --act_var=$act_var --norm_mean=$norm_mean --norm_var=$norm_var --rhs_ldim=$rhs_ldim --long_ldim=$long_ldim --type_dec=$type_ldec --drop_dec=$d_dec --nlayers_mlp_ldec=$nl_mlp_dec --nhidden_ldec=$nhidden_ldec --act_dec=$act_dec --norm_dec=$norm_dec --hivae_nl_mlp=$hivae_nl_mlp --hivae_norm=$hivae_norm --hivae_implayer=$hivae_impl --stat_ldim=$stat_ldim --s_dim_static=$s_dim_static --nlayers_projec=$nl_projec --nhidden_projec=$nhidden_projec --act_proj=$act_projec --norm_proj=$norm_projec --IC_size=$ic_size --s_dim_IC=$s_dim_ic --sde_noise_type=$sde_noise_type --sde_noise_init=$sde_noise_init --type_dynamics_lerner=$type_dynamics_lerner --nlayers_DE=$nl_nde --nhidden_de=$nhidden_nde --act_de=$act_nde --norm_de=$norm_nde --dl_static_data=$dl_static_data --norm_time=$norm_time --ANDE=$ANDE --ANDE_dim=$ANDE_dim --solver=$solv --method_solver=$method --rtol=$rtol --atol=$atol --epoch_init=$epoch_init --num_epochs=$epochs --patience=$patience --inv_ic_loss=$inv_ic_loss --lambda_RecLong=$l_reclong --lambda_KLLong=$l_kllong --lambda_IC=$l_ic --lambda_RecStat=$l_recstat --comb_long_loss=$comb_longloss --log_scaler=$log_scaler --scaler=$scaler --dataset=$dataset_ --longdata_fname=$longdata_fname --longtypes_fname=$longtypes_fname --staticdata_fname=$staticdata_fname --statictypes_fname=$statictypes_fname --dosesdata_fname=$dosesdata_fname --GPU=$gpu --batch_size=$bs --lr=$lr --save_path=$save_path --train_dir=$train_dir --save_freq=$s_freq --print_freq=$p_freq --clipping=$clipping

cd ../../../bash_scripts/MULTINDES/PROACT/
bash val1.sh