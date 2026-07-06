def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train',
                        choices=['train', 'val'])
    parser.add_argument('--dataset', type=str,
                        default='A4',
                        choices=['A4','PROACT','DATATOP'])
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=666)
    parser.add_argument('--longdata_fname', type=str,
                        default='long_data.csv')
    parser.add_argument('--longtypes_fname', type=str,
                        default='long_types.csv')
    parser.add_argument('--real_longtypes_fname', type=str,
                        default='long_types.csv')
    parser.add_argument('--staticdata_fname', type=str,
                        default='static_data.csv')
    parser.add_argument('--statictypes_fname', type=str,
                        default='static_types.csv')
    parser.add_argument('--real_statictypes_fname', type=str,
                        default='static_types.csv')
    parser.add_argument('--dosesdata_fname', type=str,
                        default='')
    parser.add_argument('--train_dir', type=str,
                        default='/home/valderramanino/Data_Generation/data')
    parser.add_argument('--save_path', type=str,
                        default='/home/valderramanino/Data_Generation/models/MultiNODEs/')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--val_data_type', type=str,
                        default='Sampling_PSD',
                        choices=['Sampling_PSD',
                                 'Sampling_PPD'])
    parser.add_argument('--Val_Scenario', type=int,
                        default=1,
                        choices=[0, 1])
    parser.add_argument('--nruns_ppd', type=int,
                        default=5,
                        help='Number of runs for appx the PPD per patient')
    parser.add_argument('--train_fold', type=int,
                        default=1,
                        help='1 for using full data for training')
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
    parser.add_argument('--extrapolation', type=int,
                        default=0,
                        choices=[0, 1])
    
    # Longitudinal encoder
    parser.add_argument('--type_lenc', type=str,
                        default='LSTM',
                        choices=['RNN', 'LSTM',
                                 'ODELSTM',
                                 'ODERNN',])
    parser.add_argument('--rev_lenc', type=int,
                        default=1)
    parser.add_argument('--encoder_implayer', type=int,
                        default=1)
    parser.add_argument('--n_long_var', type=int,
                        default=25,
                        help='Number of longitudinal variables of the dataset')
    parser.add_argument('--nlayers_rnn_enc', type=int,
                        default=2)
    parser.add_argument('--nlayers_mlp_lenc', type=int,
                        default=2)
    parser.add_argument('--nhidden_lenc', type=int,
                        default=150)
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
    parser.add_argument('--long_ldim', type=int,
                        default=47)
    parser.add_argument('--long_implayer', type=int,
                        default=1)
    
    # HI-VAE
    parser.add_argument('--static_data', type=int,
                        default=1)
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

    # Decoder parameters
    parser.add_argument('--type_dec', type=str,
                        default='MLP',
                        choices=['RNN', 'LSTM', 'MLP'])
    parser.add_argument('--drop_dec', type=float,
                        default=0.3)
    parser.add_argument('--nlayers_rnn_dec', type=int,
                        default=2)
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

    # ODE specifications
    parser.add_argument('--solver', type=str,
                        default='Adjoint',
                        choices=['Adjoint', 'Normal'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', type=str,
                        default='dopri5',
                        choices=['dopri5', 'dopri8', 'bosh3',
                                'fehlberg2', 'adaptive_heun'],
                        help='Solver for the ODE system')
    parser.add_argument('--rtol', type=float,
                        default=1e-7,
                        help='rtol option for the odeint solver')
    parser.add_argument('--atol', type=float,
                        default=1e-8,
                        help='atol option for the odeint solver')
    parser.add_argument('--t_steps', type=str,
                        default='0,3,6,9,12,18,24,30,36,42,48,54',
                        help='Time steps in months')
    parser.add_argument('--nlayers_ODE', type=int,
                        default=1,
                        choices=[1, 2, 3])
    parser.add_argument('--nhidden_ode', type=int,
                        default=215,
                        help='hidden states of the NODE')
    parser.add_argument('--act_ode', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_ode', type=str,
                        default='none',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--ode_static_data', type=str,
                        default='IC',
                        choices=['ADD', 'ADD_NN', 'IC'])
    parser.add_argument('--nlayers_ODE_staticdata', type=int,
                        default=1,
                        choices=[1, 2, 3])
    parser.add_argument('--act_ode_staticdata', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_ode_staticdata', type=str,
                        default='none',
                        choices=['none', 'instance', 'batch'])

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
    parser.add_argument('--lamba_ELBO', type=float,
                        default=0.96)
    parser.add_argument('--save_freq', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--prior_std', type=float,
                        default=1.0)
    parser.add_argument('--prior_mean', type=float,
                        default=0.0) 

    # Specific simulation parameters
    parser.add_argument('--time_max', type=int,
                        default=1,
                        help='Upper limit of simulated time')
    parser.add_argument('--time_min', type=int,
                        default=0,
                        help='Lower limit of simulated time')
    parser.add_argument('--time_steps', type=int,
                        default=2000,
                        help='Number of steps in the simulated time')
    parser.add_argument('--s_prob',
                        default=[],
                        help='vector with propability of s during prior sampling')

    # These 2 variables describe the level of noise in the 
    # reparametrization trick when using posterior sampling
    parser.add_argument('--learn_mean', type=int,
                        default=1)
    parser.add_argument('--sigma_long', type=int,
                        default=1) 
    parser.add_argument('--sigma_stat', type=int,
                        default=1)
    # Describes how often a complete population is generated
    parser.add_argument('--N_pop', type=int,
                        default=1)

    config = parser.parse_args()
    return config