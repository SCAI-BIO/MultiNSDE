#!/usr/bin/env bash

# Dataset
dataset_=A4
longdata_fname='sep_merged_scores_test.csv,sep_merged_cog_test.csv,sep_merged_long_imgfeats.csv'
longtypes_fname='sep_merged_scores_test_types_adapted.csv,sep_merged_cog_test_types_adapted.csv,sep_merged_long_imgfeats_types.csv'
staticdata_fname='sep_merged_static_data.csv'
statictypes_fname='sep_merged_static_types_adapted.csv'
dosesdata_fname='sep_merged_doses.csv'
train_dir=/home/valderramanino/SYNTHIA/Development_SDG/data/
save_path=/home/valderramanino/SYNTHIA/Development_SDG/models/

# Training
gpu=-1
lr=0.001 # Learning rate
bs=512 # batch size
p_freq=1 # training loss will be printed every p_freq iters
s_freq=100 # model weights will be saved every s_freq epochs
# If you want to continue training the model, choose an epoch when you have
# saved the model weights otherwise take 1 to train the model
epoch_init=1
epochs=2000 # Num of epochs
patience=500
log_scaler=none
scaler=robust
train_fold=1
extrapolation=1

# Loss 
l_recstat=0.4 # lambda for the static HIVAE loss 
l_reclong=1.0 # lambda for the longitudinal reconstruction loss
l_kllong=1.0 # lambda for the longitudinal KL (beta-VAE)
l_ic=1.0 # lambda for the IC loss

# Longitudinal Encoder
num_enc=3 # If 2, img feats will be used, if 3 lab test data also is used
type_lenc=LSTM # Type of encoder
rev_lenc=1 # If 1 means that the encoder is running backwards
nl_rnn_enc=1 # Number of layers of the RNN
nl_mlp_lenc=1 # Number of layers of the MLP for learning the Mean and Variance
nhidden_lenc=181 # Hidden state of the Longitudinal encoder
act_mean=none # Activation of the MLP that learns the Mean (Long ICs)
act_var=none # Activation of the MLP that learns the Variance (Long ICs
norm_mean=none # Normalization of the MLP that learns the Mean (Long ICs)
norm_var=none # Normalization of the MLP that learns the Variance (Long ICs)
rhs_ldim=24 # Latent dimension of the long information for the RHS.It is only used if type_hivae = IC_BL_HIVAE
long_ldim=44 # Longitudinal latent dimension (Z0_long)
long_impl=1 # Defines if there is missing values in the long data
comb_longloss=avg

# Longitudinal Decoder
type_ldec=DIST # Type of encoder
nl_mlp_dec=1 #  If type_ldec == DIST, then it is th number of layers for "y layer", otherwise number of layers of the MLP longitudinal decoder
nhidden_ldec=64 # Hidden state of the Longitudinal decoder. Only used for OrdReg deecoders
act_dec=none # Activation of the Longitudinal decoder
norm_dec=instance # Normalization of the Longitudinal decoder
d_dec=0.2 # Only for MLP decoder
mse_head=1
clipping=0 # Define if the model uses clipping for the MSE heads
inv_ic_loss=0 # Wassertein regularization

# HI-VAE
type_hivae=SLR_BL_HIVAE # If static_data=0, please use Z0_HIVAE as well
stat_ldim=15 # Static latent dimension (Z0_HIVAE)
s_dim_static=6 # Dimension of the sn of the Gaussian Mixture of the static data
hivae_norm=1 # Defines if the data needs to be normalized following HI-VAE original implementation
hivae_impl=0 # Defines if there is missing values in the static data
hivae_nl_mlp=1 # Number of layers for the "y" layer of HIVAE

# Projection layer
nl_projec=1
nhidden_projec=50 # Hidden state of the Longitudinal encoder. Only used if nl_project > 1
act_projec=none # Activation of the the projection MLP
norm_projec=none # Activation of the the projection MLP
ic_size=40 # Initial conditions size. Only used if nl_projec > 0
s_dim_ic=4 # Dimension of the sn of the Gaussian Mixture of the IC. Only used if we use COMB_HIVAE
ANDE=0 # use ANDE method? Only works if one use NODE as dynamics learner
ANDE_dim=0 # Number of additional compartments if ANDE is used. If  0<ANDE_dim<1 then we calculate the ANDE comparments based on ic_size

# Dynamics lerner specifications
type_dynamics_lerner=NSDE_DRHS_Mix
dl_static_data=CONCAT # ADD/CONCAT
sde_noise_type=learned_ind
sde_noise_init=0.1
norm_time=1

# SDE/ODE
solv=Normal # solver type
method=heun # solver method
rtol=1e-5 # tolerance
atol=1e-5 # tolerance
nl_nde=2 #  Number of layers of the Neural SDE/ODE
nhidden_nde=150 #  Hidden state of the Neural SDE/ODE
act_nde=none # Activation of the Neural SDE/ODE
norm_nde=instance # Normalization of the Neural SDE/ODE

name=Extrapolation_MultiNSDE_OfficialParams