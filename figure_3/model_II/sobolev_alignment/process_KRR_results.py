"""
ANALYSIS OF KRR APPROX RESULTS

2021/11/08
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import re
import torch
from pickle import dump, load

# Import params
from model_II_synthetic_params import *

print('START KRR RESULTS PROCESSING', flush=True)

# Data and saving folders
output_folder = None
opts, args = getopt.getopt(sys.argv[1:],'o:',['output='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)

if output_folder is None:
    sys.exit("NO KRR RESULTS")

# Read latent factors
latent_error_df = {
    x.replace('latent_', '').replace('.csv', ''):
    pd.read_csv('%s/KRR_error/%s'%(output_folder, x), header=0, index_col=[0,1]) 
    for x in os.listdir('%s/KRR_error/'%(output_folder)) if 'latent_kernel_' in x
}
latent_error_df = pd.concat(latent_error_df).reset_index()
if len(latent_error_df.columns) == 5:
    latent_error_df.columns = ['params', 'data_source', 'data_generation'] + ['MSE', 'reconstruction_error']
elif len(latent_error_df.columns) == 6:
    latent_error_df.columns = ['params', 'data_source', 'data_generation'] + ['MSE', 'reconstruction_error', 'spearman']

# Save the latent errors
latent_error_df.to_csv('%s/KRR_approximation_error.csv'%(output_folder))

# Process the results: put them in pivot table to compute average of spearman correlation
latent_error_df = latent_error_df.pivot_table(
    values=latent_error_df.columns[3:], 
    index=['data_source', 'params'], 
    columns=['data_generation']
)
latent_spearman_df = pd.concat({
    x: latent_error_df.loc[x][('spearman','input')]
    for x in ['source', 'target']
}, axis=1)
latent_spearman_df['combined'] = np.sum(latent_spearman_df, axis=1) / latent_spearman_df.shape[1]
latent_spearman_df = latent_spearman_df.sort_values('combined', ascending=False)

# Process results
top_KRR_results = latent_spearman_df.index[0]

optimal_sigma = float(re.findall('sigma:[0-9\.]*', top_KRR_results)[0].replace('sigma:', ''))
optimal_sigma = float(re.findall('sigma:[0-9\.]*', top_KRR_results)[0].replace('sigma:', ''))

optimal_nu = float(re.findall('nu:[0-9\.]*', top_KRR_results)[0].replace('nu:', ''))
optimal_nu = float(re.findall('nu:[0-9\.]*', top_KRR_results)[0].replace('nu:', ''))


optimal_krr_params = {
    e: krr_param_possibilities[e][0]
    for e in krr_param_possibilities
}
optimal_krr_params['kernel_params'] = {
    'sigma': optimal_sigma,
    'nu': optimal_nu
}

dump(optimal_krr_params, open('%s/params/optimal_KRR_param.pkl'%(output_folder), 'wb'))

sys.exit('KRR RESULTS SUCCESSFULLY PROCESSED')

