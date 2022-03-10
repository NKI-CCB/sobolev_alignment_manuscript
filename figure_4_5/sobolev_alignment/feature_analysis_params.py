import numpy as np
import pandas as pd

null_model_PV_sim_M = 20000
null_model_GSEA_M = 500
n_gradient_samples = 5*int(10**4)

data_folder = '../data/Kinker_Kim/'

# General Sobolev Alignment parameters
frob_norm_source = True
batch_name = 'pool'
continuous_covariate_names = None
log_input = True
no_posterior_collapse = True
mean_center = False
unit_std = False

def process_error_df(df):
    latent_error_df = pd.DataFrame(df['latent'])
    factor_error_df = pd.concat({
        x: pd.DataFrame(df['factor'][x]) 
        for x in df['factor']
    })
    return [latent_error_df, factor_error_df]

source_krr_params = {
    'method': 'falkon',
    'kernel': 'matern',
    'M': 40000,
    'penalization': 10**(-5),
    'kernel_params': {'sigma': 25., 'nu': 0.5},
    'falkon_options': {
        #'cpu_preconditioner': True,
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 5
    },
}

target_krr_params = {
    'method': 'falkon',
    'kernel': 'matern',
    'M': 40000,
    'penalization': 10**(-5.3),
    'kernel_params': {'sigma': 25., 'nu': 0.5},
    'falkon_options': {
        #'cpu_preconditioner': True,
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 5
    },
}

default_krr_params = {
    'method': 'falkon',
    'kernel': 'matern',
    'M': 40,
    'penalization': 1e-5,
    'kernel_params': {'sigma': 25., 'nu': 0.5},
    'falkon_options': {
        #'cpu_preconditioner': True,
        'max_cpu_mem': 2**(8*4.8),
        'never_store_kernel': False,
        'max_gpu_mem': 2**(8*4.2),
        'num_fmm_streams': 5
    },
}


cell_line_scvi_params = {
    'model': {
        'dispersion': 'gene-cell',
        'gene_likelihood': 'nb',
        'n_hidden': 512,
        'n_latent': 21,
        'n_layers': 2,
        'dropout_rate': 0.3
    },
    'plan': {
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'reduce_lr_on_plateau': False,
    },
    'train': {
        'early_stopping': True,
    }
}

tumor_scvi_params = {
    'model': {
        'dispersion': 'gene-cell',
        'gene_likelihood': 'zinb',
        'n_hidden': 256,
        'n_latent': 20,
        'n_layers': 2,
        'dropout_rate': 0.
    },
    'plan': {
        'lr': 0.0001,
        'weight_decay': 0.00001,
        'reduce_lr_on_plateau': True,
    },
    'train': {
        'early_stopping': True
    }
}

