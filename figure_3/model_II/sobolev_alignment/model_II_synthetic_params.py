import numpy as np
import pandas as pd
from pickle import load

data_folder = './data/'

null_model_M = 20000

# General Sobolev Alignment parameters
frob_norm_source = True
batch_name = None
continuous_covariate_names = None
log_input = True
no_posterior_collapse = True
mean_center = False
unit_std = False

same_model_sim_threshold = 0.95

krr_param_possibilities = {
    'method': ['falkon'],
    'kernel': ['matern'],
    'M': [2500],
    'penalization': [1e-6],
    'kernel_params': [
        {'sigma': s, 'nu': n} 
        for s in [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12.5, 15., 17.5, 20., 22.5, 25., 30., 40., 50.]
        for n in [.5, 1.5, 2.5, np.inf]
    ],
    'maxiter': [20],
    'falkon_options': [{
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 6
        }]
}

same_model_param_possibilities_penalization = np.logspace(-7,-1,14)
same_model_param_possibilities_M = 5000

default_krr_params = {
    'method': 'falkon',
    'kernel': 'matern',
    'M': 40,
    'penalization': 1e-5,
    'kernel_params': {'sigma': 5., 'nu': 0.5},
    'falkon_options': {
        #'cpu_preconditioner': True,
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 5
    },
}


def read_scvi_params(output_folder):
    return [read_one_scvi_param(output_folder, x) for x in ['source', 'target']]

def read_one_scvi_param(output_folder, data_type):
    return load(open('%s/params/optimal_hyperopt_params_%s.pkl'%(output_folder, data_type), 'rb'))

def read_optimal_kernel_param(output_folder):
    return load(open('%s/params/optimal_KRR_param.pkl'%(output_folder), 'rb'))

def read_optimal_KRR_param(output_folder, data_source):
    return load(open('%s/params/KRR_params_%s.pkl'%(output_folder, data_source), 'rb'))

def process_error_df(df):
    latent_error_df = pd.DataFrame(df['latent'])
    factor_error_df = pd.concat({
        x: pd.DataFrame(df['factor'][x]) 
        for x in df['factor']
    })
    return [latent_error_df, factor_error_df]

def shuffle_matrix(A):
    return np.array([
        A_col[np.random.permutation(A.shape[0])] for A_col in A.T
    ]).reshape(A.shape)

same_model_param_possibilities = {
    'method': ['falkon'],
    'kernel': ['matern'],
    'M': [5000],
    'penalization': np.logspace(-7,-1,14),
    'kernel_params': [
        {'sigma': s, 'nu': n} 
        for s in [2.]
        for n in [2.5]
    ],
    'maxiter': [20],
    'falkon_options': [{
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 6
        }]
}
