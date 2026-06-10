#!/usr/bin/env bash

# Dataset
dataset_=A4_Causal
data_fname='Causal_OR_data.csv'
ps_types_fname='Causal_OR_PS_types.csv'
do_types_fname='Causal_OR_DO_types.csv'
train_dir=/home/valderramanino/SYNTHIA/Development_SDG/data/
save_path=/home/valderramanino/SYNTHIA/Development_SDG/models/
calibrate_pi_model=1
calibrate_g_model=1

name=OR_ATEmodels
cd ../../../MultiNSDEs/Causal/OR_ATEmodels/
python main.py --exp_name=$name --dataset=$dataset_ --calibrate_pi_model=$calibrate_pi_model --calibrate_g_model=$calibrate_g_model --data_fname=$data_fname --ps_types_fname=$ps_types_fname --do_types_fname=$do_types_fname --save_path=$save_path --train_dir=$train_dir

