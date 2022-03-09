"""
ANALYSIS OF HYPEROPT RESULTS

2021/11/08
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import torch
from pickle import dump, load

print('START HYPEROPT PROCESSING', flush=True)

# Data and saving folders
output_folder = None
opts, args = getopt.getopt(sys.argv[1:],'o:',['output='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)

if output_folder is None:
    sys.exit("NO HYPEROPT RESULTS")

for data_type in ['source', 'target']:
    hyperopt_results_folder = '%s/hyperopt_%s/'%(output_folder, data_type)

    # Process potential results files
    hyperopt_results_subfolders = os.listdir(hyperopt_results_folder)

    if len(hyperopt_results_subfolders) > 1:
        print('WARNING: MORE THAN ONE FOLDER')
    elif len(hyperopt_results_subfolders) < 1:
        sys.exit('WARNING: NO HYPEROPT FOLDER')
        
    hyperopt_results_folder += hyperopt_results_subfolders[0] + '/'

    # Load and process results
    results_df = pd.read_csv('%s/hyperopt_results.csv'%(hyperopt_results_folder), index_col=0)
    results_df = results_df.dropna()
    results_df = results_df.sort_values('valid_reconstruction', ascending=False)

    # Save all hyperopt
    results_df.to_csv('%s/params/hyperopt_results_%s.csv'%(output_folder, data_type))
    
    # Save best result
    optimal_parameters = dict(results_df.sort_values('valid_reconstruction', ascending=False).iloc[0])
    optimal_parameters = {
        'model': {
            'dispersion': optimal_parameters['dispersion'],
            'gene_likelihood': optimal_parameters['likelihood'],
            'n_hidden': optimal_parameters['n_hidden'],
            'n_latent': optimal_parameters['n_latent'],
            'n_layers': optimal_parameters['n_layers'],
            'dropout_rate': optimal_parameters['dropout_rate']
        },
        'plan': {
            'lr': optimal_parameters['learning_rate'],
            'weight_decay': optimal_parameters['weight_decay'],
            'reduce_lr_on_plateau': optimal_parameters['reduce_lr_on_plateau'],
        },
        'train': {
            'early_stopping': optimal_parameters['early_stopping'],
        }
    }
    dump(optimal_parameters, open('%s/params/optimal_hyperopt_params_%s.pkl'%(output_folder, data_type), 'wb'))

sys.exit('HYPEROPT RESULTS SUCCESSFULLY PROCESSED')
