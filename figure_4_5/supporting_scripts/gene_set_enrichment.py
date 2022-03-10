import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
import gc


def import_gene_set(gene_set):
    # Import gene sets
    with open(gene_set, 'r') as gene_set_file:
        gene_set_df = gene_set_file.read().split('\n')
        gene_set_df = [gs.split('\t') for gs in gene_set_df]
        gene_set_df = {gs[0]:gs[2:] for gs in gene_set_df}
        
    return gene_set_df


def filter_gene_sets(gene_sets, gene_names, min_size=10, max_size=100):
    filtered_gene_sets = []
    for gs in gene_sets:
        n_genes = np.intersect1d(gene_names, gene_sets[gs]).shape[0]
        if n_genes > min_size and n_genes < max_size:
            filtered_gene_sets.append(gs)

    gene_sets = {
        k: gene_sets[k]
        for k in filtered_gene_sets
    }
    print('%s GENE SETS FILTERED'%(len(gene_sets)))
    return gene_sets



def compute_gene_set_weights(gene_set_df, weights_gene_set_df, PV_number, power=1):
    for gs in gene_set_df:
        weights_gene_set_df[gs] = 0
        for gene_comp in ['gene_A', 'gene_B']:
            gene_idxs = weights_gene_set_df[gene_comp].isin(gene_set_df[gs])
            if power > 0:
                weights_gene_set_df.loc[gene_idxs,gs] = np.power(np.abs(weights_gene_set_df[gene_idxs]['PV %s_norm'%(PV_number)]), power)
            else:
                weights_gene_set_df.loc[gene_idxs,gs] = np.abs(weights_gene_set_df[gene_idxs]['PV %s_norm'%(PV_number)])

    weights_gene_set_df = weights_gene_set_df.set_index(['gene_A', 'gene_B', 'PV %s_norm'%(PV_number)])
    relevant_columns = [e for e in weights_gene_set_df.columns if len(e) > 0]
    weights_gene_set_df = weights_gene_set_df[relevant_columns]
    
    return weights_gene_set_df


def integrate_interaction_pairs(weights_gene_set_df):
    weights_interaction_gene_set_df = pd.DataFrame(index=weights_gene_set_df.index)
    for gsA_idx, gsA in enumerate(weights_gene_set_df.columns):
        for gsB_idx, gsB in enumerate(weights_gene_set_df.columns[gsA_idx:]):
            weights_interaction_gene_set_df.loc[:,'%s-%s'%(gsA,gsB)] = np.sqrt(
                weights_gene_set_df[gsA] * weights_gene_set_df[gsB]
            )

    return weights_interaction_gene_set_df
    

def compute_gene_set_enrichment_interactions(gene_set, interaction_weights_df, PV_number, gene_names=None,power=1):
    gene_set_df = import_gene_set(gene_set)
    if gene_names is not None:
        gene_set_df = filter_gene_sets(gene_set_df, gene_names, min_size=10, max_size=100)
    # weights_gene_set_df = interaction_weights_df[['PV %s_norm'%(PV_number)]].reset_index().sort_values('PV %s_norm'%(PV_number), ascending=False)
    weights_gene_set_df = interaction_weights_df[['PV %s_norm'%(PV_number)]].sort_values('PV %s_norm'%(PV_number), ascending=False)
    weights_gene_set_df = weights_gene_set_df.reset_index()
    weights_gene_set_df['PV %s_norm'%(PV_number)] = weights_gene_set_df['PV %s_norm'%(PV_number)].astype(np.float32)

    # Go through the interactions and score the gene sets involved in the interactions
    print('1. COMPUTING GENE SET WEIGHTS', flush=True)
    weights_gene_set_df = compute_gene_set_weights(gene_set_df, weights_gene_set_df, PV_number, power)
    weights_gene_set_df = weights_gene_set_df.astype(np.float32)
    gc.collect()

    # Integration of weights in pairs of gene set
    weights_interaction_gene_set_df = integrate_interaction_pairs(weights_gene_set_df)
    del weights_gene_set_df
    weights_interaction_gene_set_df = weights_interaction_gene_set_df.astype(np.float32)
    gc.collect()

    # Normalisation coefficients
    print('2. COMPUTING NORMALISATION COEFFICIENTS', flush=True)
    N_H = (weights_interaction_gene_set_df == 0.).sum(axis=0).astype(np.float32)
    N_R = weights_interaction_gene_set_df.sum(axis=0).astype(np.float32)
    
    # Normalisation
    print('3. NORMALISATION COEFFICIENTS', flush=True)
    for gs_idx, gs in enumerate(list(weights_interaction_gene_set_df.columns)):
        if gs_idx % 1000 == 0:
            print('\t\t START GS NO %s OUT OF %s'%(gs_idx, weights_interaction_gene_set_df.shape[1]))
        weights_interaction_gene_set_df[gs] = weights_interaction_gene_set_df[gs] / N_R[gs]
        weights_interaction_gene_set_df[gs] = weights_interaction_gene_set_df[gs].replace(0., -1./N_H[gs])
    
    print('4. CUMULATIVE SUMMATION COEFFICIENTS', flush=True)
    weights_gene_set_df_cumulative_scores = np.cumsum(
        weights_interaction_gene_set_df,#.astype(np.float16), 
        axis=0
    )
    del weights_interaction_gene_set_df
    weights_gene_set_df_cumulative_scores = weights_gene_set_df_cumulative_scores.astype(np.float32)
    gc.collect()

    # Computation of enrichment score
    print('5. ENRICHMENT SCORES', flush=True)
    ES_score = pd.DataFrame(
        [np.min(weights_gene_set_df_cumulative_scores), np.max(weights_gene_set_df_cumulative_scores)]
    ).T
    ES_score.columns = ['neg_ES', 'pos_ES']
    ES_score['ES'] = np.sign(ES_score['neg_ES'] + ES_score['pos_ES']) * np.max(np.abs(ES_score), axis=1)
    
    return weights_gene_set_df_cumulative_scores, ES_score


def integrate_enrichment_scores(ES_df, permuted_ES_df):
    """
    Given the results of compute_gene_set_enrichment_interactions and compute_permuted_normalization_scores, integrate
    the two sets and compute normalized scores
    """
    # Compute normalization scores
    permuted_ES_df['neg_ES_idx'] = (permuted_ES_df['pos_ES'] + permuted_ES_df['neg_ES'] < 0)
    permuted_ES_df.loc[~permuted_ES_df['neg_ES_idx'], 'neg_ES'] = 0
    permuted_ES_df.loc[permuted_ES_df['neg_ES_idx'], 'pos_ES'] = 0

    normalization_ES_coef = permuted_ES_df.groupby(axis=0, level=0).agg('sum')
    n_perm = np.min(permuted_ES_df.groupby(axis=0, level=0).agg('count')['neg_ES_idx'].values)
    normalization_ES_coef['neg_ES'] = normalization_ES_coef['neg_ES'] / normalization_ES_coef['neg_ES_idx']
    normalization_ES_coef['pos_ES'] = normalization_ES_coef['pos_ES'] / (n_perm - normalization_ES_coef['neg_ES_idx'])
    del normalization_ES_coef['ES'], normalization_ES_coef['neg_ES_idx']


    # Integrate and merge permuted and normal results
    normalized_ES_df = ES_df.merge(
        normalization_ES_coef,
        left_index=True,
        right_index=True,
        suffixes=('', '_perm')
    )
    normalized_ES_df['norm_coef'] = 0
    normalized_ES_df.loc[normalized_ES_df['ES'] < 0, 'norm_coef'] = normalized_ES_df.loc[normalized_ES_df['ES'] < 0, 'neg_ES_perm']
    normalized_ES_df.loc[normalized_ES_df['ES'] > 0, 'norm_coef'] = normalized_ES_df.loc[normalized_ES_df['ES'] > 0, 'pos_ES_perm']
    normalized_ES_df['norm_coef'] = np.abs(normalized_ES_df['norm_coef'])

    # Compute NES by norm_coef division
    normalized_ES_df['NES'] = normalized_ES_df['ES'] / normalized_ES_df['norm_coef']

    # Integrate and correct permuted scores
    normalized_permuted_ES_df = permuted_ES_df.merge(
        normalization_ES_coef,
        left_index=True,
        right_index=True,
        suffixes=('', '_norm')
    )
    normalized_permuted_ES_df['neg_ES_norm'] = np.abs(normalized_permuted_ES_df['neg_ES_norm'])
    normalized_permuted_ES_df = _norm_perm_df(normalized_permuted_ES_df)

    del normalized_permuted_ES_df['neg_ES']
    del normalized_permuted_ES_df['pos_ES']
    del normalized_permuted_ES_df['neg_ES_norm']
    del normalized_permuted_ES_df['pos_ES_norm']
    del normalized_permuted_ES_df['neg_ES_idx']
    del normalized_ES_df['neg_ES']
    del normalized_ES_df['pos_ES']
    del normalized_ES_df['neg_ES_perm']
    del normalized_ES_df['pos_ES_perm']
    del normalized_ES_df['norm_coef']
    gc.collect()

    return normalized_ES_df, normalized_permuted_ES_df
    

def _norm_perm_df(df):
    df.loc[df['neg_ES_idx'], 'NES'] = df.loc[df['neg_ES_idx'], 'neg_ES'] / df.loc[df['neg_ES_idx'],'neg_ES_norm']
    df.loc[~df['neg_ES_idx'], 'NES'] = df.loc[~df['neg_ES_idx'], 'pos_ES'] / df.loc[~df['neg_ES_idx'],'pos_ES_norm']
    return df


def compute_positive_FDR(normalized_ES_df, permuted_normalized_ES_df):
    """
    Given a dataframe of normalized enrichment scores and permutations, compute the FDR as indicated in the
    original GSEA paper.
    """
    positive_pathways_NES_df = normalized_ES_df[normalized_ES_df['NES'] > 0]
    null_model_positive_df = permuted_normalized_ES_df[permuted_normalized_ES_df['NES'] > 0]
    n_positive_null = null_model_positive_df.shape[0]
    n_positive_NES = positive_pathways_NES_df.shape[0]
    
    positive_FDR = {}
    for gene_set in positive_pathways_NES_df.index:
        # Proportion of permutation above 0, but below NES of gene set
        prop_null_sup = np.sum(null_model_positive_df['NES'] >= positive_pathways_NES_df.loc[gene_set]['NES']) 
        prop_null_sup /= n_positive_null

        # Proportion of gene sets above 0 which are also above the NES of the gene set
        prop_NES_sup = np.sum(positive_pathways_NES_df['NES'] >=  positive_pathways_NES_df.loc[gene_set]['NES'])
        prop_NES_sup /= n_positive_NES
        
        positive_FDR[gene_set] = min(prop_null_sup / prop_NES_sup,1)
        
    return positive_FDR

####
# FEATURE-LEVEL PERMUTATION
###

def _gene_permuted_enrichment_scores(interaction_weights_df, gene_set, PV_number, power=1):
    permutations = np.arange(interaction_weights_df.shape[0])
    np.random.shuffle(permutations)
    permutated_interaction_weights_df = pd.DataFrame(
        data=interaction_weights_df.values[permutations],
        index=interaction_weights_df.index,
        columns=interaction_weights_df.columns
    )

    permuted_values_df, permuted_ES_df = compute_gene_set_enrichment_interactions(
        gene_set,
        permutated_interaction_weights_df, 
        PV_number,
        power
    )
    
    return permuted_ES_df


def compute_gene_permuted_normalization_scores(interaction_weights_df, gene_set, PV_number, n_perm=1000, n_jobs=30, power=1):
    """
    Gene-level permutation and computation of the normalization ratio for assessing significance.
    """
    # Calling method with permutation
    permuted_ES_df = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_gene_permuted_enrichment_scores)(
            interaction_weights_df,
            gene_set,
            PV_number,
            power
        ) for _ in range(n_perm))

    # Aggregate results and compute normalization coefficients
    permuted_ES_df = pd.concat(permuted_ES_df)
    return permuted_ES_df

####
# SAMPLE-LEVEL PERMUTATION
###

def _sample_permuted_enrichment_scores(
    interaction_weights_df,
    inverse_product_transform,
    sample_PV_scores,
    gene_set, 
    PV_number, 
    power=1
):
    permuted_interaction_weights = pd.DataFrame(
        np.random.permutation(sample_PV_scores[PV_number]).dot(inverse_product_transform),
        index=interaction_weights_df.index,
        columns=interaction_weights_df.columns
    )
    
    _, perm_ES_df = compute_gene_set_enrichment_interactions(
        gene_set,
        permuted_interaction_weights, 
        PV_number,
        power=power
    )
    return perm_ES_df

def compute_sample_permuted_normalization_scores(
    interaction_weights_df,
    inverse_product_transform,
    sample_PV_scores,
    gene_set, 
    PV_number, 
    n_perm=1000, n_jobs=30, power=1):
    """
    Gene-level permutation and computation of the normalization ratio for assessing significance.
    """
    # Calling method with permutation
    permuted_ES_df = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_sample_permuted_enrichment_scores)(
            interaction_weights_df=interaction_weights_df,
            inverse_product_transform=inverse_product_transform,
            sample_PV_scores=sample_PV_scores,
            gene_set=gene_set,
            PV_number=PV_number,
            power=power
        ) for _ in range(n_perm))

    # Aggregate results and compute normalization coefficients
    permuted_ES_df = pd.concat(permuted_ES_df)
    return permuted_ES_df





