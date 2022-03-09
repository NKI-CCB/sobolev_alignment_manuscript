"""
This script performs the feature attribution by Taylor expansion, specifically:
- Loading Sobolev Alignment with VAE trained.
- Sample artificial data form the VAE distribution.
- Train KRR on these artificial data and align the latent factors.
- Using the trained KRR, compute the values of the projection on Hilbert polynomials.
- Save the results.
"""

import os, sys, getopt
import pandas as pd
import numpy as np
from anndata import AnnData
import torch
from pickle import dump, load
from copy import deepcopy
import gc

from sobolev_alignment import SobolevAlignment

# Import params
from model_III_synthetic_params import *
from read_data import read_data


# Import parameters
n_artificial_samples = None
tmp_file = None
n_iter = 1
opts, args = getopt.getopt(sys.argv[1:],'o:d:n:t:j:i:',['output=', 'data=', 'artifsamples=', 'temp=', 'job=', 'iter='])
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
    elif opt in ('-i', '--iter'):
        n_iter = int(arg)
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


###
# Feature analysis by weight estimation
###

print('START FEATURE ANALYSIS \n\t %s ITERATIONS'%(n_iter), flush=True)

source_krr_params = read_optimal_KRR_param(output_folder, 'source')
target_krr_params = read_optimal_KRR_param(output_folder, 'target')

for iter_idx in range(n_iter):
    for nu in [source_krr_params['kernel_params']['nu'], np.inf]:
        source_krr_params['kernel_params']['nu'] = nu
        target_krr_params['kernel_params']['nu'] = nu
        sobolev_alignment_clf.krr_params = {
            'source': source_krr_params,
            'target': target_krr_params
        }

        # Make saving folder or pass if already computed
        iter_output_folder = '%s/iter_%s_nu_%s/'%(
            output_folder, 
            iter_idx,
            'laplacian' if nu == 0.5 else ('gaussian' if nu == np.inf else nu)
        )
        print('\t\t START %s'%(iter_output_folder))
        if iter_output_folder.replace('%s/'%(output_folder), '').replace('/', '') not in os.listdir(output_folder):
            os.mkdir(iter_output_folder)
        else:
            continue

        sobolev_alignment_clf.n_jobs = 1
        print('\t START ITER %s'%(iter_idx), flush=True)
        sobolev_alignment_clf.fit(
            X_source=X_source,
            X_target=X_target,
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
        sobolev_alignment_clf.frob_norm_source = None
        sobolev_alignment_clf.save('%s/sobolev_alignment_model/'%(iter_output_folder), with_krr=True, with_model=False)

        print('\t START ERROR COMPUTING', flush=True)
        krr_approx_error = sobolev_alignment_clf.compute_error(size=10**4)
        processed_error_df = {x: process_error_df(df) for x, df in krr_approx_error.items()}
        processed_latent_error_df = {x: df[0] for x, df in processed_error_df.items()}
        processed_latent_error_df = pd.concat(processed_latent_error_df)
        processed_factor_error_df = {x: df[1] for x, df in processed_error_df.items()}
        processed_factor_error_df = pd.concat(processed_factor_error_df)
        
        #Save error logs
        processed_latent_error_df.to_csv('%s/latent_error.csv'%(iter_output_folder))
        processed_factor_error_df.to_csv('%s/factor_error.csv'%(iter_output_folder))

        torch.cuda.empty_cache()
        gc.collect()

gc.collect()
sys.exit("FINISH ORTHONORMAL BASIS ATTRIBUTION")