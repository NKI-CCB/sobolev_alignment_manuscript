import pandas as pd
import numpy as np
import torch

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


def read_latent_factors(output_folder, file_radical, n_iter, kernel_name='laplacian'):
    """
    Read all files related to linear factors.
    output_folder: Folder to read from.
    file_radical: Which files to read (remove csv).
    n_iter: number of iterations
    """
    results_df = {}
    for kernel_type in [kernel_name, 'gaussian']:
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

def process_PV_linear_norm_comparison(df, kernel_name='laplacian'):
    """
    Compute the normalising factor, for each PV and each iteration, between Laplace
    and Gaussian kernel. This would then be used to rescale Laplace elements.
    """
    df = df.reset_index().pivot_table(
        values='square_value', 
        columns='kernel', 
        index=['factor', 'iter']
    )
    df['multiplying_factor'] = df['gaussian'] / df[kernel_name]
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


def process_linear_weights(df, kernel_name='laplacian'):
    """
    Transforms the matrix into the form genes x PVs.
    """
    linear_weights = df[df['kernel'] == kernel_name]
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