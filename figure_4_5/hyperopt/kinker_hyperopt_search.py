"""
SEARCH OF HYPERPARAMETER FOR KINKER ET AL
2021/04/21 - Soufiane Mourragui

This scripts performs the Bayesian search for hyper-parameters in
scVI using hyperopt. The idea is to find the optimal combination of 
hyper-parameters that gives the best performance in left-out samples.

Comparison of performances are returned.
Launched on Rossmann to benefit from GPU acceleration.
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
import scipy
import seaborn as sns
import umap
import uuid, time, datetime
import torch

from anndata import AnnData
import scvi
import scanpy as sc
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from typing import Any, Callable, Dict, List, TextIO, Type, Union
import anndata
from functools import partial, wraps
import multiprocessing
from subprocess import Popen
from logging.handlers import QueueHandler, QueueListener
from queue import Empty
import logging
import pickle
import threading

# scVI import
import scvi
from scvi.model.base import BaseModelClass
from scvi.train import Trainer

from sklearn.model_selection import train_test_split

###
# Parameters
###

test_size = None
max_eval = None
opts, args = getopt.getopt(sys.argv[1:],'o:d:c:m:',['output=', 'data=', 'cvfold=', 'maxeval='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)
    elif opt in ("-d", "--data"):
        data_subfolder = str(arg)
    elif opt in ('-c', '--cvfold'):
        test_size = float(arg)
    elif opt in ('-m', '--maxeval'):
        max_eval = int(arg)
test_size = test_size if test_size is not None else 0.1
max_eval = max_eval if max_eval is not None else 100

# Data and saving folders
cell_line_folder = '../data/Kinker_Kim/'

# Create output_folder with specific save id
if data_subfolder not in os.listdir(output_folder):
    os.mkdir('%s/%s'%(output_folder, data_subfolder))
output_folder = '%s/%s'%(output_folder, data_subfolder)

save_id = 'cell_line_' + '{:%B_%d_%Y_}'.format(datetime.datetime.now()) + str(uuid.uuid4()).split('-')[1]
if save_id not in os.listdir(output_folder):
    os.mkdir('%s/%s'%(output_folder, save_id))
output_folder = '%s/%s'%(output_folder, save_id)

# Hyperopt params
space = {
    'n_hidden': hp.choice('n_hidden', [32, 64, 128, 256, 512, 1024]),
    'n_latent': 5 + hp.randint('n_latent', 20),
    'n_layers': 1 + hp.randint('n_layers', 5),
    'dropout_rate': hp.choice('dropout_rate', [0., 0.1, 0.3, 0.5, 0.7]),
    'likelihood': hp.choice('likelihood', ['zinb', 'nb']),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.005, 0.001, 0.0005, 0.0001]),
    'reduce_lr_on_plateau': hp.choice('reduce_lr_on_plateau', [True, False]),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'weight_decay': hp.choice('weight_decay', [0.01, 0.001, 0.0001, 0.00001, 0.]),
    'dispersion': hp.choice('dispersion', ['gene', 'gene-batch', 'gene-cell'])
}
dump(space, open('%s/param_space.pkl'%(output_folder), 'wb'))

###
# Read and process data
###

# Read data
cell_line_data_file = '%s/%s/cell_line_count.csv'%(cell_line_folder, 
                                                data_subfolder)
cell_line_data_df = pd.read_csv(cell_line_data_file, sep=',', index_col=[0,1,2])
cell_line_data_df.index.names = ['UMI', 'sample', 'pool']

# Format and divide test and train
train_data_df, test_data_df = train_test_split(cell_line_data_df, test_size=test_size)
train_data_an = AnnData(
    train_data_df.values, 
    obs=pd.DataFrame(np.array([np.array(e) for e in train_data_df.index]),
                     columns=['UMI', 'sample', 'pool'])
)

test_data_an = AnnData(
    test_data_df.values, 
    obs=pd.DataFrame(np.array([np.array(e) for e in test_data_df.index]),
                     columns=['UMI', 'sample', 'pool'])
)
test_data_an.to_df().to_csv('%s/test_samples.csv'%(output_folder))

random_train_data_an = np.random.choice(train_data_an.obs.index, size=test_data_an.shape[0])
random_train_data_an = train_data_an[random_train_data_an]

###
# Format to scVI
###

scvi.data.setup_anndata(train_data_an, batch_key='pool')

###
# Objective function definition
###

def _objective_function(params):
    
    n_hidden = params['n_hidden']
    n_layers = params['n_layers']
    n_latent = params['n_latent']
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    reduce_lr_on_plateau = params['reduce_lr_on_plateau']
    dispersion = params['dispersion']
    likelihood = params['likelihood']
    early_stopping = params['early_stopping']
    
    clf = scvi.model.SCVI(
        train_data_an, 
        n_hidden=n_hidden, 
        n_latent=n_latent, 
        n_layers=n_layers, 
        dropout_rate=dropout_rate, 
        dispersion=dispersion, 
        gene_likelihood=likelihood
    )
    
    plan_kwargs = {
        'lr': learning_rate,
        'reduce_lr_on_plateau': reduce_lr_on_plateau, 
        'weight_decay': weight_decay
    }
    trainer_kwargs = {
    }
    
    try:
        clf.train(use_gpu=1,
                  early_stopping=early_stopping, 
                  plan_kwargs=plan_kwargs,
                  **trainer_kwargs)

        # Reconstruction error
        test_reconstruction_error = clf.get_reconstruction_error(adata=test_data_an)['reconstruction_loss']
        train_reconstruction_error = clf.get_reconstruction_error()['reconstruction_loss']
        test_size_train_reconstruction_error = clf.get_reconstruction_error(adata=random_train_data_an)['reconstruction_loss']

        # ELBO
        test_elbo = clf.get_elbo(adata=test_data_an)
        train_elbo = clf.get_elbo()
        del clf

        results_dict = {
            'loss': - test_elbo,
            'loss_choice': 'ELBO',
            'train_reconstruction': train_reconstruction_error,
            'train_reconstruction_test_size': test_size_train_reconstruction_error,
            'valid_reconstruction': test_reconstruction_error,
            'train_ELBO': train_elbo,
            'test_ELBO': test_elbo,
            'status': STATUS_OK,
        }
    except Exception as err:
        results_dict = {'status': STATUS_FAIL,
                    'loss': np.iinfo(np.uint64).max}
        print('\n\n\n ERROR: \n %s \n\n\n\n'%(err))
        results_dict.update(params)

    results_dict.update(params)
    return results_dict

###
# Bayesian search
###

# Launch bayesian search
save_hyperopt_res = Trials()
best = fmin(_objective_function, space, algo=tpe.suggest, max_evals=max_eval, return_argmin=False, trials=save_hyperopt_res)

# Save
pd.DataFrame(save_hyperopt_res.results).to_csv('%s/hyperopt_results.csv'%(output_folder))
with open('%s/hyperopt_argmin.csv'%(output_folder), 'w') as f:
    f.write(str(best))
dump(save_hyperopt_res, open('%s/hyperopt_trials.pkl'%(output_folder), 'wb'))

