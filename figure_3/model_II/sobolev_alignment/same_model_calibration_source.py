"""
ANALYSIS OF APPROXIMATION WHEN COMPARISON SOURCE-SOURCE AND TARGET-TARGET
SOURCE ELEMENT
Model II

The idea here is to see whether penalization has an impact on the alignment.

2021/11/05
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import re
from anndata import AnnData
import torch
import scipy
from pickle import dump, load
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from joblib.externals.loky import get_reusable_executor

from sobolev_alignment import SobolevAlignment

# Import params
from model_II_synthetic_params import *
from read_data import read_data


n_artificial_samples = None
tmp_file = None
# Data and saving folders
opts, args = getopt.getopt(sys.argv[1:],'o:d:n:t:j:',['output=', 'data=', 'artifsamples=', 'temp=', 'job='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)
    elif opt in ("-d", "--data"):
        data_subfolder = str(arg)
    elif opt in ('-n', '--artifsamples'):
        n_artificial_samples = int(arg)
    elif opt in ('-t', '--temp'):
        tmp_file = str(arg)
    elif opt in ('-j', '--job'):
        n_jobs = int(arg)
n_artificial_samples = n_artificial_samples if n_artificial_samples is not None else 10**6
n_artificial_samples = int(n_artificial_samples)
tmp_file = tmp_file if tmp_file is not None else '/tmp/SM/'


###
# IMPORT DATA
###

X_source, X_target = read_data(data_folder, data_subfolder)
gc.collect()

###
# LOAD SOBOLEV ALIGNMENT
###

sobolev_alignment_clf = SobolevAlignment.load('%s/sobolev_alignment_model/'%(output_folder), with_model=True, with_krr=False)
sobolev_alignment_clf.scvi_models['target'] = sobolev_alignment_clf.scvi_models['source']
gc.collect()

###
# Artificial sampling
###

sobolev_alignment_clf.n_jobs = 1
sobolev_alignment_clf.fit(
    X_source=X_source,
    X_target=X_source,
    source_batch_name=batch_name,
    target_batch_name=batch_name,
    continuous_covariate_names=continuous_covariate_names,
    n_artificial_samples=n_artificial_samples,
    fit_vae=False,
    sample_artificial=True,
    krr_approx=True,
    n_samples_per_sample_batch=10**6,
    frac_save_artificial=10**4/n_artificial_samples,
    save_mmap=tmp_file,
    log_input=log_input,
    no_posterior_collapse=no_posterior_collapse,
    mean_center=mean_center,
    unit_std=unit_std,
    frob_norm_source=frob_norm_source
)
gc.collect()

###
# KRR APPROX
###

# Process results
same_model_param_possibilities = read_optimal_kernel_param(output_folder)
same_model_param_possibilities = {
    e: [same_model_param_possibilities[e]]
    for e in same_model_param_possibilities
}
same_model_param_possibilities['penalization'] = same_model_param_possibilities_penalization
same_model_param_possibilities['M'] = [same_model_param_possibilities_M]

krr_param_grid = ParameterGrid(same_model_param_possibilities)

# Folder for saving
if 'comparison_same_model_source' not in os.listdir(output_folder):
    os.mkdir('%s/comparison_same_model_source'%(output_folder))
output_folder = '%s/comparison_same_model_source/'%(output_folder)

latent_results_krr_error_df = {}
factor_results_krr_error_df = {}
principal_angles_df = {}

def process_error_df(df):
    latent_error_df = pd.DataFrame(df['latent'])
    factor_error_df = pd.concat({
        x: pd.DataFrame(df['factor'][x]) 
        for x in df['factor']
    })
    return [latent_error_df, factor_error_df]

for krr_params in krr_param_grid: 
    param_id = 'source_kernel_%s_M_%s_penalization_%s_params_%s_maxiter_%s'%(
        krr_params['kernel'],
        krr_params['M'],
        '$'.join(['%s:%s'%(e,f) for e,f in krr_params['kernel_params'].items()]),
        krr_params['penalization'],
        krr_params['maxiter']
    )
    
    if 'latent_%s.csv'%(param_id) is os.listdir(output_folder):
        continue
    print('START %s'%(param_id), flush=True)
    
    sobolev_alignment_clf.krr_params = {
        'source': krr_params,
        'target': krr_params
    }

    print('\t START KRR TRAINING', flush=True)
    sobolev_alignment_clf.fit(
        X_source=X_source,
        X_target=X_source,
        fit_vae=False,
        sample_artificial=False,
        krr_approx=True,
        n_samples_per_sample_batch=10**6,
        save_mmap=tmp_file,
        log_input=True,
        no_posterior_collapse=True
    )
    gc.collect()

    # Compute_error
    print('\t START ERROR COMPUTING', flush=True)
    krr_approx_error = sobolev_alignment_clf.compute_error(size=10**4)
    processed_error_df = {x: process_error_df(df) for x, df in krr_approx_error.items()}
    processed_latent_error_df = {x: df[0] for x, df in processed_error_df.items()}
    processed_latent_error_df = pd.concat(processed_latent_error_df)
    processed_factor_error_df = {x: df[1] for x, df in processed_error_df.items()}
    processed_factor_error_df = pd.concat(processed_factor_error_df)
    
    #Save error logs
    processed_latent_error_df.to_csv('%s/latent_%s.csv'%(output_folder, param_id))
    processed_factor_error_df.to_csv('%s/factor_%s.csv'%(output_folder, param_id))
    
    latent_results_krr_error_df[param_id] = processed_latent_error_df
    factor_results_krr_error_df[param_id] = processed_factor_error_df
    
    principal_angles_df[param_id] = sobolev_alignment_clf.principal_angles
    pd.DataFrame(principal_angles_df).to_csv('%s/source_principal_angles.csv'%(output_folder))
    torch.cuda.empty_cache()