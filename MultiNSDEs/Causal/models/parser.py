def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train',
                        choices=['train', 'val', 'cv'])
    parser.add_argument('--dataset', type=str,
                        default='A4_Causal',
                        choices=['A4_Causal','DATATOP_Causal'])
    parser.add_argument('--longdata_fname', type=str,
                        default='long_data.csv')
    parser.add_argument('--longtypes_fname', type=str,
                        default='long_types.csv')
    parser.add_argument('--staticdata_fname', type=str,
                        default='static_data.csv')
    parser.add_argument('--statictypes_fname', type=str,
                        default='static_types.csv')
    parser.add_argument('--dosesdata_fname', type=str,
                        default='')
    parser.add_argument('--timetoevent_fname', type=str,
                        default='')
    parser.add_argument('--ps_scores_fname', type=str,
                        default='')
    parser.add_argument('--do_scores_fname', type=str,
                        default='')
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--train_dir', type=str,
                        default='/home/jovyan/IBD_MultiNODEs/data/')
    parser.add_argument('--save_path', type=str,
                        default='/home/jovyan/IBD_MultiNODEs/models/')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--val_data_type', type=str,
                        default='Sampling_Prior',
                        choices=['Sampling_Prior'])
    parser.add_argument('--Val_Scenario', type=int,
                        default=1,
                        choices=[1])
    parser.add_argument('--extrapolation', type=int,
                        default=0,
                        choices=[0],
                        help='Keeping it just for compatibility')
    parser.add_argument('--HIVAE_MAP_sampling', type=int,
                        default=0)
    parser.add_argument('--nruns_ppd', type=int,
                        default=5,
                        help='Number of runs for appx the PPD per patient')
    parser.add_argument('--train_fold', type=int,
                        default=0,
                        help='0 for using full data for training')
    parser.add_argument('--kfold', type=int,
                        default=5,
                        help='Number of folds for CV')
    parser.add_argument('--scaler', type=str,
                        default='minmax',
                        choices=['minmax',
                                 'robust',
                                 'none'])
    parser.add_argument('--log_scaler', type=str,
                        default='none',
                        choices=['log',
                                 'log1p',
                                 'none'])

    # Longitudinal encoder
    parser.add_argument('--num_lenc', type=int,
                        default=1)
    parser.add_argument('--nlayers_mlp_lenc', type=str,
                        default='2')
    parser.add_argument('--nhidden_lenc', type=str,
                        default='150')
    parser.add_argument('--act_mean', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--act_var', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_mean', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--norm_var', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--long_ldim', type=str,
                        default='47')
    parser.add_argument('--long_implayer', type=int,
                        default=1)
    parser.add_argument('--comb_long_loss', type=str,
                        default='avg',
                        choices=['avg', 'sum'])

    # HI-VAE
    parser.add_argument('--type_hivae', type=str,
                        default='SLR_BL_Causal_HIVAE', # Way to reconstruct Static data, w/wo BL info
                        choices=['SLR_BL_Causal_HIVAE', # Slide 7 Causal Version of slide 1
                                 'IC_BL_Causal_HIVAE']) # Slide 8 Causal Version of slide 1
    parser.add_argument('--hivae_norm', type=int,
                        default=1)
    parser.add_argument('--hivae_implayer', type=int,
                        default=0)
    parser.add_argument('--stat_ldim', type=int,
                        default=3,
                        help='Dimension of the zn of the Gaussian Mixture of the static data')
    parser.add_argument('--s_dim_static', type=int,
                        default=6,
                        help='Dimension of the sn of the Gaussian Mixture of the static data')
    parser.add_argument('--hivae_nl_mlp', type=int,
                        default=1,
                        help='Number of layers for the "y" layer of HIVAE')

    # Decoder parameters
    parser.add_argument('--drop_dec', type=float,
                        default=0.3)
    parser.add_argument('--nlayers_mlp_ldec', type=int,
                        default=2)
    parser.add_argument('--nhidden_ldec', type=int,
                        default=209)
    parser.add_argument('--act_dec', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_dec', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--mse_head', type=int,
                        default=0)
    parser.add_argument('--clipping', type=int,
                        default=0)
    parser.add_argument('--inv_ic_loss', type=int,
                        default=0)
    parser.add_argument('--time_to_event', type=int,
                        default=0)
    parser.add_argument('--or_losses', type=int,
                        default=1)

    # Projection
    parser.add_argument('--IC_size', type=int,
                        default=25)
    parser.add_argument('--s_dim_IC', type=int,
                        default=15,
                        help='Dimension of the sn of the Gaussian Mixture of the IC')
    parser.add_argument('--nlayers_projec', type=int,
                        default=1)
    parser.add_argument('--nhidden_projec', type=int,
                        default=50)
    parser.add_argument('--act_proj', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_proj', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--ANDE', type=int,
                        default=0)
    parser.add_argument('--ANDE_dim', type=int,
                        default=5)
    parser.add_argument('--ANDE_dim_perc', type=float,
                        default=0)

    # Dynamics learner specifications
    parser.add_argument('--type_dynamics_lerner', type=str,
                        default='NSDE_DRHS_Mix',
                        choices=['NSDE_DRHS_Concat',
                                 'NSDE_DRHS_Mix',
                                 'NSDE_DRHS_Mix_Stat',
                                 'NSDE_DRHS_Split'],
                        help='System to learn the longitudinal dynamics')
    parser.add_argument('--dl_static_data', type=str,
                        default='CONCAT')
    parser.add_argument('--norm_time', type=int,
                        default=1)
    parser.add_argument('--use_months_time', type=int,
                        default=1)
    parser.add_argument('--sde_noise_type', type=str,
                        default='fixed',
                        choices=['fixed', 'learned_all',
                                'learned_ind'])
    parser.add_argument('--sde_noise_init', type=float,
                        default=0.1)

    # SDE/ODE specifications
    parser.add_argument('--solver', type=str,
                        default='Adjoint',
                        choices=['Adjoint', 'Normal'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', type=str,
                        default='heun',
                        choices=['heun', 'euler_heun', 'adjoint_reversible_heun',
                                 'reversible_heun', 'milstein', 'euler', 'midpoint',
                                 'log_ode', 'milstein_ito', 'srk'],
                        help='Solver for the system')
    parser.add_argument('--rtol', type=float,
                        default=1e-7,
                        help='rtol option for the solver')
    parser.add_argument('--atol', type=float,
                        default=1e-8,
                        help='atol option for the solver')
    parser.add_argument('--nlayers_DE', type=int,
                        default=1,
                        choices=[1, 2, 3])
    parser.add_argument('--nhidden_de', type=int,
                        default=215,
                        help='hidden states of the NODE')
    parser.add_argument('--act_de', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_de', type=str,
                        default='none',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--time_bn_de', type=str,
                        default='none',
                        choices=['none', 'TBN', 'TABN'])
    parser.add_argument('--time_bn_hdim', type=int,
                        default=16)

    # Training specifications
    parser.add_argument('--batch_size', type=int,
                        default=227)
    parser.add_argument('--from_best', type=int,
                        default=0)
    parser.add_argument('--lr', type=float,
                        default=0.001)
    parser.add_argument('--num_epochs', type=int,
                        default=3000)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--patience', type=int,
                        default=200)
    parser.add_argument('--lambda_RecStat', type=float,
                        default=1.0)
    parser.add_argument('--lambda_KLLong', type=float,
                        default=1.0)
    parser.add_argument('--lambda_RecLong', type=float,
                        default=5.0)
    parser.add_argument('--lambda_IC', type=float,
                        default=10.0)
    parser.add_argument('--lambda_TE', type=float,
                        default=10.0)
    parser.add_argument('--lambda_OR_TRT', type=float,
                        default=1e-3)
    parser.add_argument('--lambda_OR_DO', type=float,
                        default=1e-3)
    parser.add_argument('--save_freq', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--prior_std', type=float,
                        default=1.0)
    parser.add_argument('--prior_mean', type=float,
                        default=0.0) 
    parser.add_argument('--learn_mean', type=int,
                        default=1)
    config = parser.parse_args()
    return config

def adapt_parser(config):
    import os
    num_enc = config.num_lenc
    # Adding try exception ecause optuna. If an exception occurs 
    # it is because we are not optimizing those values
    # String variables
    try:
        config.act_mean = config.act_mean.split(',')
        if len(config.act_mean) != num_enc:
            config.act_mean = config.act_mean * num_enc
    except:
        pass            
    try:
        config.act_var = config.act_var.split(',')
        if len(config.act_var) != num_enc:
            config.act_var = config.act_var * num_enc
    except:
        pass
    try:
        config.norm_mean = config.norm_mean.split(',')
        if len(config.norm_mean) != num_enc:
            config.norm_mean = config.norm_mean * num_enc
    except:
        pass
    try:
        config.norm_var = config.norm_var.split(',')        
        if len(config.norm_var) != num_enc:
            config.norm_var = config.norm_var * num_enc
    except:
        pass

    # Int variables
    try:
        config.nlayers_mlp_lenc = [int(i) for i in config.nlayers_mlp_lenc.split(',')]
        if len(config.nlayers_mlp_lenc) != num_enc:
            config.nlayers_mlp_lenc = config.nlayers_mlp_lenc * num_enc
    except:
        pass
    try:
        config.nhidden_lenc = [int(i) for i in config.nhidden_lenc.split(',')]
        if len(config.nhidden_lenc) != num_enc:
            config.nhidden_lenc = config.nhidden_lenc * num_enc
    except:
        pass
    try:
        config.long_ldim = [int(i) for i in config.long_ldim.split(',')]
        if len(config.long_ldim) != num_enc:
            config.long_ldim = config.long_ldim * num_enc
    except:
        pass

    if '_Split' in config.type_dynamics_lerner:
        assert int(config.rhs_ldim) % 2 == 0, 'The RHS dimension from the LongData needs to be divisible by 2'
    try:
        config.rhs_ldim = [int(i) for i in config.rhs_ldim.split(',')]
        if len(config.rhs_ldim) != num_enc:
            config.rhs_ldim = config.rhs_ldim * num_enc        
    except:
        pass
    return config