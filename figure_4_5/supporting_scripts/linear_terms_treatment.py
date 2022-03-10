import pandas as pd
import numpy as np
import torch
from copy import deepcopy

def read_linear_weights(
    output_folder,
    file_radical,
    n_iter
):
    # Read factors
    linear_features = read_latent_factors(
        output_folder,
        file_radical,
        n_iter
    )
    
    # Format factors
    linear_features = process_df(linear_features, square_value=True)
    
    # Correct scaling coefficient between Gaussian and Laplacian
    PV_norm_comparison = process_PV_linear_norm_comparison(
        linear_features.groupby(['kernel', 'factor', 'iter']).agg('sum')
    )
    linear_features = correct_linear_features(linear_features, PV_norm_comparison)
    assess_equality_contribution(linear_features)

    # Process the gene names
    linear_features['variable'] = linear_features['variable'].str.extract(r'([A-Za-z0-9-]*)\^1')
    
    return linear_features


def compute_gene_std_frame(X_log_input, sobolev_alignment_clf):
    """
    Given a set of model points and a dictionary of Sobolev Alignment instances, return the standard
    deviation of the linear terms, i.e. genes multiplied by the Gaussian term.

    Input:
        - X_log_input: 
            dictionary with torch.Tensor elements indicating the model points.
        - sobolev_alignment_clf :
            dictionary with SobolevAlignment instances

    Output:
        - Dictionary with frames indicating the standard deviation of each feature in source and target. 
    """
    # Compute scaling factor for kernel
    gamma = {s: _compute_kernel_param(s, sobolev_alignment_clf) for s in ['source', 'target']}
    assert gamma['source'] == gamma['target']
    gamma = gamma['source']
    
    # Compute standard deviation
    all_genes_std = {
        x: compute_gene_std(
            X_log_input[x],
            np.exp(- gamma * np.square(np.linalg.norm(X_log_input[x], axis=1)))
        )
        for x in ['source', 'target']
    }
    all_genes_std = {
        x: pd.DataFrame(all_genes_std[x], index=['feature_std']).T
        for x in ['source', 'target']
    }
    
    return all_genes_std


def linear_weights_std_scaling(PV_linear_features, all_genes_std, data_source):    
    # Merge dataset and correct
    df = deepcopy(PV_linear_features)
    df = df.merge(
        all_genes_std[data_source],
        left_on='variable',
        right_index=True
    )
    df['standardized_value'] = df['corrected_value'] * df['feature_std']
    
    return df



def process_df(df, square_value=False):
    """
    Transforms dictionary of dictionary of pd.DataFrame into a combined
    pd.DataFrame. If square_value, then returns an additional column with
    squared value.
    """
    df = pd.concat({d:pd.concat(df[d]) for d in df})
    df.index.names = ['kernel', 'iter', 'factor']
    df = df.reset_index().melt(id_vars=['kernel', 'factor', 'iter'])
    if square_value:
        df['square_value'] = np.square(df['value'])
    return df


def read_latent_factors(output_folder, file_radical, n_iter):
    """
    Read all files related to linear factors.
    output_folder: Folder to read from.
    file_radical: Which files to read (remove csv).
    n_iter: number of iterations
    """
    results_df = {}
    for kernel_type in ['laplacian', 'gaussian']:
        results_df[kernel_type] = {}
        for iter_idx in range(n_iter+1):
            iter_folder = '%s/iter_%s_nu_%s/'%(output_folder, iter_idx, kernel_type)
            results_df[kernel_type]['iter_%s'%(iter_idx)] = pd.read_csv(
                '%s/%s.csv'%(iter_folder, file_radical),
                index_col=0
            )
    return results_df

def aggregate_norm_comparison(factor_df, M_matrix):
    """
    Aggregate factor information with cosine similarity matrix, which contains the
    true norm.
    """
    return factor_df.groupby(['kernel', 'factor', 'iter']).agg('sum').merge(
        M_matrix.set_index(['kernel', 'factor', 'iter']),
        left_index=True,
        right_index=True,
        suffixes=('_linear', '_factor')
    )

def process_PV_linear_norm_comparison(df):
    """
    Compute the normalising factor, for each PV and each iteration, between Laplace
    and Gaussian kernel. This would then be used to rescale Laplace elements.
    """
    df = df.reset_index().pivot_table(
        values='square_value', 
        columns='kernel', 
        index=['factor', 'iter']
    )
    df['multiplying_factor'] = df['gaussian'] / df['laplacian']
    df['multiplying_factor'] = np.sqrt(df['multiplying_factor'])
    return df

def correct_linear_features(feature_df, corr_df):
    """
    Rescale Laplace PVs linear elements
    """
    df = feature_df.merge(
        corr_df.reset_index(), 
        how='left', 
        on=['factor', 'iter']
    )
    df.loc[df['kernel'] == 'gaussian', 'multiplying_factor'] = 1.
    df['corrected_value'] = df['value'] * df['multiplying_factor']
    df['square_corrected_value'] = np.square(df['corrected_value'])
    return df


def assess_equality_contribution(df):
    corr_df = df.groupby(['kernel', 'factor', 'iter']).agg('sum').pivot_table(
        values='square_corrected_value', 
            columns='kernel', 
            index=['factor', 'iter']
    ).corr().values
    np.testing.assert_almost_equal(corr_df, np.ones(shape=(2,2)))


def process_linear_weights(df):
    """
    Transforms the matrix into the form genes x PVs.
    """
    linear_weights = df[df['kernel'] == 'laplacian']
    linear_weights = linear_weights.pivot_table(
        values='corrected_value', 
        index='factor', 
        columns='variable'
    )
    linear_weights.columns = [e.replace('^1', '') for e in linear_weights.columns]
    return linear_weights.T


def compute_gene_std(X_log_df, offset):
    gene_std = pd.DataFrame(
        np.diag(offset).dot(X_log_df),
        columns=X_log_df.columns
    )
    gene_std = np.std(gene_std)
    return gene_std.to_dict()

def _compute_kernel_param(data_source, sobolev_alignment_clf):
    clf = sobolev_alignment_clf['gaussian']
    sigma = clf.krr_params[data_source]['kernel_params']['sigma']
    return 1 / (2 * sigma**2)