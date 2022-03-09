"""
ANALYSIS OF SAME MODEL ALIGNMENT

2021/11/08
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import torch
from pickle import dump, load
import re

# Import params
from model_II_synthetic_params import *

print('START SAME MODEL PROCESSING', flush=True)

# Data and saving folders
output_folder = None
opts, args = getopt.getopt(sys.argv[1:],'o:',['output='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)

if output_folder is None:
    sys.exit("NO SAME MODEL RESULTS")

# Import results
for data_source in ['source', 'target']:
    principal_angles_df = pd.read_csv(
        '%s/comparison_same_model_%s/%s_principal_angles.csv'%(output_folder, data_source, data_source),
        index_col=0
    )
    min_principal_angles_df = pd.DataFrame(np.min(principal_angles_df))
    min_principal_angles_df = pd.DataFrame(np.min(principal_angles_df))
    min_principal_angles_df.columns = ['min_sim']
    min_principal_angles_df['penalisation'] = [
        float(re.findall(r'params_([\.0-9e\-]*)', e)[0])
        for e in min_principal_angles_df.index.get_level_values(0)
    ]

    # Select best model
    selected_model = min_principal_angles_df[min_principal_angles_df['min_sim'] > same_model_sim_threshold]
    selected_model = np.min(selected_model['penalisation'])

    krr_params = read_optimal_kernel_param(output_folder)
    krr_params['penalization'] = selected_model

    dump(krr_params, open('%s/params/KRR_params_%s.pkl'%(output_folder, data_source), 'wb'))
    with open('%s/params/KRR_params_%s.txt'%(output_folder, data_source), 'w') as file:
        file.write(str(krr_params))
    del krr_params


sys.exit("SAME MODEL ALIGNMENT PROCESSED SUCCESSFULLY")


