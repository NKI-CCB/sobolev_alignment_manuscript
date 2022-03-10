import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import reduce
import os, sys
import gc
from .gene_set_enrichment import compute_gene_set_enrichment_interactions


def compute_permuted_es(
    output_folder,
    iter_idx,
    file_radical,
    interaction_scaling_coef,
    gene_set, 
    PV_number, 
    interations_std,
    std_filter=None,
    gene_names=None,
    n_interactions_subset=-1,
    power=1
):
    """
    Open the sample permutation of a SobolevAlignment instance and compute the ES.
    """
    print('START ITER %s'%(iter_idx))
    gc.collect()
    weights = read_and_process_sample_permutated_interactions(
        output_folder,
        iter_idx,
        file_radical,
        interaction_scaling_coef
    )
    weights = correct_std(weights, interations_std, PV_number)
    if std_filter is not None:
        weights = weights.loc[std_filter]
    if n_interactions_subset > 0:
         weights = weights.sample(n_interactions_subset)

    for x in weights.columns:
        weights[x] = weights[x].astype(np.float16)
    gc.collect()
    
    permuted_es =  compute_gene_set_enrichment_interactions(
        gene_set,
        weights,
        PV_number,
        gene_names=gene_names,
        power=power
    )[1]
    
    del weights
    gc.collect()
    
    return permuted_es


def correct_std(permuted_interaction_weights, all_interations_std, PV_number):
    interaction_weights_corrected = permuted_interaction_weights.merge(
        all_interations_std,
        left_index=True,
        right_index=True
    )

    interaction_weights_corrected = interaction_weights_corrected[['PV %s'%(PV_number), 'std']]
    interaction_weights_corrected['PV %s_norm'%(PV_number)] = interaction_weights_corrected['PV %s'%(PV_number)]  * interaction_weights_corrected['std']
    return interaction_weights_corrected[['PV %s_norm'%(PV_number)]]
    


def compute_std_one_interaction(df, offset, gene_A, gene_B):
    """
    df: dataframe with data (sample x gene).
    offset: matrix of sample offset.
    gene_A, gene_B: two genes to screen for
    """
    inter_std = np.multiply(
        np.multiply(df[gene_A], df[gene_B]),
        offset
    )
    return [gene_A, gene_B, np.std(inter_std)]


def _compute_all_interactions_std_one_gene(df, offset, gene_A):
    return [
        compute_std_one_interaction(df, offset, gene_A, gene_B)
        for gene_B in df.columns if gene_A <= gene_B
    ]


def compute_all_interactions_std(df, offset):
    return Parallel(n_jobs=30, verbose=1)(
        delayed(_compute_all_interactions_std_one_gene)(df, offset, gene_A)
        for gene_A in df.columns
    )


def read_interaction_weights(
    output_folder,
    iter_idx,
    file_radical
):
    """
    Read the interaction terms of two folders: one with Gaussian and one with Laplacian kernel.
    Proceed to correct the weights to harmonize the norms.

    - output_folder: Output folder of the "feature_analysis" experiment.
    - iter_idx: Iteration ID considered.
    - file_radical: Name of the file to consider.
    """
    # Read interaction terms
    print('START READING INTERACTIONS %s'%(file_radical), flush=True)
    PV_inter_features = _read_interaction_terms(
        iter_idx=iter_idx, 
        file_radical=file_radical, 
        output_folder=output_folder
    )
    PV_inter_features = _process_interaction_terms(PV_inter_features)
    
    # Aggregate scaling coefficients (to re-use later in permutation testing)
    source_interaction_scaling_coef = PV_inter_features[['kernel', 'factor', 'multiplying_factor', 'is_squared']]
    source_interaction_scaling_coef = source_interaction_scaling_coef.drop_duplicates()
    source_interaction_scaling_coef = source_interaction_scaling_coef.set_index('kernel')
    source_interaction_scaling_coef = source_interaction_scaling_coef.loc['laplacian']

    # Restrict to Laplacian and return both the scaling coefficients of Laplacian kernel
    # and the corrected values in a gene x PV matrix
    return source_interaction_scaling_coef, _restrict_laplacian_kernel(PV_inter_features)


def _read_interaction_terms(iter_idx, file_radical, output_folder):
    """
    Read one interaction files and correct between Gaussian and Laplacian terms.
    """

    # Read the different interaction terms
    feature_df = pd.concat({
        k: pd.read_pickle(
            '%s/iter_%s_nu_%s/%s.gz'%(output_folder, iter_idx, k, file_radical),
            compression='gzip'
        ) for k in ['laplacian', 'gaussian']
    }).T
    feature_df = feature_df.reset_index().melt(id_vars='index')
    feature_df.columns = ['interaction', 'kernel', 'factor', 'weight']
    feature_df['square_weight'] = np.square(feature_df['weight'])
    feature_df['is_squared'] = feature_df['interaction'].str.contains(r'(\^2)')
    
    # # Compute Laplacian reweighting
    norm_df = feature_df.groupby(['kernel', 'factor', 'is_squared']).agg('sum')
    norm_df = norm_df.pivot_table(columns='kernel', index=['factor', 'is_squared'])['square_weight']
    norm_df['multiplying_factor'] = norm_df['gaussian'] / norm_df['laplacian']
    norm_df['multiplying_factor'] = np.sqrt(norm_df['multiplying_factor'])

    # Aggregate with all weights
    feature_df = feature_df.merge(norm_df, on=['factor', 'is_squared'])
    del feature_df['gaussian'], feature_df['laplacian']#, feature_df['is_squared']
    feature_df.loc[feature_df['kernel'] == 'gaussian', 'multiplying_factor'] = 1
    feature_df['corrected_weight'] = feature_df['weight'] * feature_df['multiplying_factor']
    feature_df['square_corrected_weight'] = np.square(feature_df['corrected_weight'])
    
    return feature_df


def _process_interaction_terms(PV_inter_features):
    # Process gene name
    PV_inter_features[['gene_A', 'gene_B']] = PV_inter_features['interaction'].str.split('*', expand=True)
    is_square_gene = pd.isnull(PV_inter_features['gene_B'])
    PV_inter_features.loc[is_square_gene, 'gene_B'] = PV_inter_features.loc[is_square_gene, 'gene_A']
    for x in ['gene_A', 'gene_B']:
        PV_inter_features[x] = PV_inter_features[x].str.replace('(\^[1-9])$', '')
    del PV_inter_features['interaction']
    
    return PV_inter_features


def _restrict_laplacian_kernel(PV_inter_features):
    # Restrict to Laplacian features
    laplacian_PV_inter_features = PV_inter_features.loc[PV_inter_features['kernel'] == 'laplacian']
    laplacian_PV_inter_features = laplacian_PV_inter_features[['factor', 'gene_A', 'gene_B', 'corrected_weight']]

    return laplacian_PV_inter_features.pivot_table(
        values='corrected_weight',
        columns='factor',
        index=['gene_A', 'gene_B']
    )

def _compute_kernel_param(data_source, sobolev_alignment_clf):
    clf = sobolev_alignment_clf['gaussian']
    sigma = clf.krr_params[data_source]['kernel_params']['sigma']
    return 1 / (2 * sigma**2)


def read_and_process_sample_permutated_interactions(
    output_folder, 
    iter_idx, 
    file_radical,
    interaction_scaling_coef
):
    """
    Read, process and correct for Laplacian difference for one sample-permuted file.
    
    - output_folder: folder of one "feature_analysis" experiment.
    
    - iter_idx: index of the iteration to study.
    
    - file_radical: file to look for inside the iteration subfolder.
    
    - interaction_scaling_coef: scaling coefficients DataFrame for Laplacian kernel (given as 
    first output of read_interaction_weights)
    """
    # Read interactions file
    permuted_interactions_df = pd.read_pickle(
        '%s/GSEA_null/iter_%s_nu_laplacian/%s.gz'%(output_folder, iter_idx, file_radical),
        compression='gzip'
    )
    
    # Processing
    permuted_interactions_df = permuted_interactions_df.reset_index().melt(id_vars='index')
    permuted_interactions_df.columns = ['factor', 'interaction', 'weight']
    permuted_interactions_df['square_weight'] = np.square(permuted_interactions_df['weight'])
    permuted_interactions_df['is_squared'] = permuted_interactions_df['interaction'].str.contains(r'(\^2)')
    
    # Merge with scaling for interactions
    permuted_interactions_df = permuted_interactions_df.merge(
        interaction_scaling_coef,
        on=['factor', 'is_squared']
    )
    permuted_interactions_df['corrected_weight'] = permuted_interactions_df['weight'] * permuted_interactions_df['multiplying_factor']
    permuted_interactions_df['square_corrected_weight'] = np.square(permuted_interactions_df['corrected_weight'])
    
    # Format interactions
    permuted_interactions_df = _process_interaction_terms(permuted_interactions_df)
    permuted_interactions_df = permuted_interactions_df[['factor', 'gene_A', 'gene_B', 'corrected_weight']]

    return permuted_interactions_df.pivot_table(
        values='corrected_weight',
        columns='factor',
        index=['gene_A', 'gene_B']
    )
    