"""
Analysis of interactions using Leiden clustering and EnrichR.
"""

import gc
import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed
import leidenalg
import gseapy as gp


def possible_pathways():
    return gp.utils.DEFAULT_LIBRARY


def compute_interaction_weights_subgraph(subgraph_id, partitions, gene_set, gene_names):
    print(subgraph_id, flush=True)
    subgraph = partitions.subgraphs()[subgraph_id]
    
    # Compute enrichments of genes with EnrichR
    enrichment_df = _enrichr_analysis(subgraph, gene_set, gene_names)

    # Process edge results
    edge_df = _process_edge_df(subgraph)

    # Compute and returns proportion of interactions
    if enrichment_df.results.empty:
        return pd.DataFrame()
    return _compute_interaction_weights_proportion(enrichment_df, edge_df)


def _enrichr_analysis(subgraph, gene_set, gene_names):
    return gp.enrichr(
        gene_list=list(subgraph.get_vertex_dataframe()['name'].values),
        gene_sets=[gene_set] if type(gene_set) is str else gene_set,
        organism='Human', 
        description='test_name', # TO CHANGE
        outdir='test/enrichr_kegg', # TO CHANGE
        background=gene_names,
    #     no_plot=True,
        cutoff=0.5 # test dataset, use lower value from range(0,1)
    )


def _process_edge_df(subgraph):
    """
    Returns an edge DataFrame with the names of the genes instead of their IDs.
    """
    return subgraph.get_edge_dataframe().merge(
        subgraph.get_vertex_dataframe().reset_index(),
        left_on='source',
        right_on='vertex ID',
    ).merge(
        subgraph.get_vertex_dataframe().reset_index(),
        left_on='target',
        right_on='vertex ID',
        suffixes=('_source', '_target')
    )[['name_source', 'name_target', 'weight']]


def _compute_interaction_weights_proportion(enrichment_df, edge_df):
    # Process enrichment results
    sign_enr = enrichment_df.results.sort_values('Adjusted P-value')
    sign_enr = sign_enr[sign_enr['Adjusted P-value'] < 0.05]
    
    interaction_weights_sum_df = {}

    # Iterate over significant gene sets and compute the sum of cross-interactions
    for gs1 in sign_enr.iterrows():
        for gs2 in sign_enr.iterrows():
            if gs1[0] >= gs2[0]:
                continue
            gs1_genes = gs1[1]['Genes'].split(';')
            gs2_genes = gs2[1]['Genes'].split(';')

            inter_subgraph_df = edge_df[
                (edge_df['name_source'].isin(gs1_genes) & edge_df['name_target'].isin(gs2_genes)) | \
                (edge_df['name_target'].isin(gs1_genes) & edge_df['name_source'].isin(gs2_genes))
            ]

            interaction_weights_sum_df[(gs1[1]['Term']), gs2[1]['Term']] = np.sum(inter_subgraph_df['weight'])

    interaction_weights_sum_df = pd.DataFrame(interaction_weights_sum_df, index=['sum_interactions']).T
    interaction_weights_sum_df.sort_values('sum_interactions', ascending=False, inplace=True)
    interaction_weights_sum_df['interaction_ratio'] = interaction_weights_sum_df['sum_interactions'] / np.sum(edge_df['weight'])
    
    return interaction_weights_sum_df