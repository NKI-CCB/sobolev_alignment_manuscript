"""
This script computes a null model for the GSEA analysis by:
- Loading Sobolev Alignment with VAE trained.
- Sample artificial data form the VAE distribution.
- Permute the sample in the source (compared to embedding).
- Compute the PVs.
- Save coefficients.
We do it here for a significantly lower number of anchor points, for computational
purposes (getting permutations up to 100s is not feasible otherwise)
"""

import os, sys, getopt
import pandas as pd
import numpy as np
from anndata import AnnData
import torch
import scipy
from copy import deepcopy
import gc

from sobolev_alignment import SobolevAlignment

# Import params
from feature_analysis_params import *
from read_data import read_data


# Import parameters
n_artificial_samples = None
tmp_file = None
n_perm = 10
opts, args = getopt.getopt(sys.argv[1:],'o:d:n:t:j:i:p:',['output=', 'data=', 'artifsamples=', 'temp=', 'job=', 'iter=', 'perm='])
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
    elif opt in ('-p', '--perm'):
        n_perm = int(arg)
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

sobolev_alignment_clf = SobolevAlignment.load(
    '%s/sobolev_alignment_model/'%(output_folder), 
    with_model=True, 
    with_krr=False
)

###
# NULL MODELS TO ESTIMATE THE NUMBER OF COMMON PVS.
###

print('START GSEA NULL MODEL', flush=True)
permuted_sobolev_alignment_clf = deepcopy(sobolev_alignment_clf)

# Lower number of anchor points (M) for computation scalability 
permuted_sobolev_alignment_clf.krr_params = {
    'source': deepcopy(source_krr_params),
    'target': deepcopy(target_krr_params)
}
for k in permuted_sobolev_alignment_clf.krr_params:
    permuted_sobolev_alignment_clf.krr_params[k]['M'] = null_model_GSEA_M

# Generate data: 10x less than normal for computation scalability
permuted_sobolev_alignment_clf.n_jobs = 1
permuted_sobolev_alignment_clf.fit(
    X_source=X_source,
    X_target=X_target,   
    source_batch_name=batch_name,
    target_batch_name=batch_name,
    continuous_covariate_names=continuous_covariate_names,
    n_artificial_samples=int(n_artificial_samples/10),
    fit_vae=False,
    sample_artificial=True,
    krr_approx=True,
    n_samples_per_sample_batch=10**6,
    save_mmap=tmp_file,
    log_input=log_input,
    frac_save_artificial=1.,
    mean_center=mean_center,
    unit_std=unit_std,
    frob_norm_source=frob_norm_source
)
X_source_t = deepcopy(permuted_sobolev_alignment_clf.artificial_samples_['source'])
X_target_t = deepcopy(permuted_sobolev_alignment_clf.artificial_samples_['target'])
gc.collect()

nu = source_krr_params['kernel_params']['nu']
null_model_sim = {}
for perm_idx in range(n_perm):
    print('START PARAM %s'%(perm_idx), flush=True)

    iter_output_folder = '%s/GSEA_null/iter_%s_nu_%s/'%(
        output_folder, 
        perm_idx,
        'laplacian' if nu == 0.5 else ('gaussian' if nu == np.inf else nu)
    )

    if os.path.isdir(iter_output_folder):
        continue

    # Permute order of samples in source
    source_sample_idx = np.arange(X_source_t.shape[0])
    np.random.shuffle(source_sample_idx)
    permuted_sobolev_alignment_clf.artificial_samples_['source'] = X_source_t[source_sample_idx]

    # Permute order of samples in target
    target_sample_idx = np.arange(X_target_t.shape[0])
    np.random.shuffle(target_sample_idx)
    permuted_sobolev_alignment_clf.artificial_samples_['target'] = X_target_t[target_sample_idx]

    # Train KRR on gene-level permuted data
    permuted_sobolev_alignment_clf.fit(
        X_source=permuted_sobolev_alignment_clf.training_data['source'],
        X_target=permuted_sobolev_alignment_clf.training_data['target'],
        source_batch_name=batch_name,
        target_batch_name=batch_name,
        continuous_covariate_names=continuous_covariate_names,
        fit_vae=False,
        sample_artificial=False,
        krr_approx=True,
        save_mmap=tmp_file,
        log_input=log_input,
        frac_save_artificial=1.,
        mean_center=mean_center,
        unit_std=unit_std,
        frob_norm_source=frob_norm_source
    )
    gc.collect()

    permuted_sobolev_alignment_clf.save(iter_output_folder, with_krr=True, with_model=False)

gc.collect()
sys.exit("FINISH NULL GSEA MODEL")