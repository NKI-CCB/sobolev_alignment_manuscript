"""
COMPUTATION OF FEATURE WEIGHTS

Compute feature weights given fitted Sobolev Alignment classifiers
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import re
from anndata import AnnData
import scipy
from pickle import dump, load
from copy import deepcopy
import gc
import scanpy
from time import process_time

sys.path.insert(0, '/home/s.mourragui/science/sobolev_alignment/src/sobolev_alignment/')
from sobolev_alignment import SobolevAlignment, KRRApprox

# Import parameters
max_order = 2
opts, args = getopt.getopt(sys.argv[1:],'o:j:m:',['output=', 'job=', 'max_order='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)
    elif opt in ('-j', '--job'):
        n_jobs = int(arg)
    elif opt in ('-m', '--max_order'):
        max_order = int(arg)
###
# IMPORT DATA
###

X_combined = pd.read_csv(
    '%s/combined_counts.csv'%(output_folder),
    sep=',', 
    index_col=[0,1]
)
X_source = X_combined.loc['CELL_LINE']
X_target = X_combined.loc['TUMOR']


assert X_source.shape[1] == X_combined.shape[1]
assert X_target.shape[1] == X_combined.shape[1]
assert X_source.shape[0] + X_target.shape[0] == X_combined.shape[0]

X_source = AnnData(
    X_source.values, 
    obs=pd.DataFrame(np.array([np.array(e) for e in X_source.index]),
                     columns=['UMI'])
)
X_target = AnnData(
    X_target.values, 
    obs=pd.DataFrame(np.array([np.array(e) for e in X_target.index]),
                     columns=['UMI'])
)

gene_names = np.array(X_combined.columns)

###
# LOAD AND COMPUTE FEATURE WEIGHTS
###

iter_folders = [e for e in os.listdir(output_folder) if 'iter' in e and 'gradient' not in e]
print(iter_folders)

for order in range(1,max_order+1):
    for iter_subfolder in iter_folders:
        print('START ITER %s ORDER %s'%(iter_subfolder, order))
        iter_folder = '%s/%s/'%(output_folder, iter_subfolder)

        if 'latent_factors_linear_weights_source_order_%s.csv'%(order) in os.listdir(iter_folder) and 'latent_factors_linear_weights_target_order_%s.csv'%(order) in os.listdir(iter_folder):
            continue

        sobolev_alignment_clf = SobolevAlignment.load(
            '%s/sobolev_alignment_model/'%(iter_folder), 
            with_model=False
        )
        sobolev_alignment_clf.training_data = {
            'source': X_source,
            'target': X_target
        }
        sobolev_alignment_clf.n_jobs = n_jobs

        # Recompute PVs to compute the unaligned ones
        sobolev_alignment_clf._compute_principal_vectors(all_PVs=True)

        sobolev_alignment_clf.feature_analysis(max_order=order, gene_names=gene_names)
        gc.collect()
        for x in ['source', 'target']:
            sobolev_alignment_clf.factor_level_feature_weights_df[x].to_csv(
                '%s/latent_factors_linear_weights_%s_order_%s.csv'%(iter_folder, x, order)
            )
            sobolev_alignment_clf.pv_level_feature_weights_df[x].to_csv(
                '%s/PV_linear_weights_%s_order_%s.csv'%(iter_folder, x, order)
            )
            sobolev_alignment_clf.factor_level_feature_weights_df[x].to_pickle(
                '%s/latent_factors_linear_weights_%s_order_%s.gz'%(iter_folder, x, order),
                compression='gzip'
            )
            sobolev_alignment_clf.pv_level_feature_weights_df[x].to_pickle(
                '%s/PV_linear_weights_%s_order_%s.gz'%(iter_folder, x, order)
            )
