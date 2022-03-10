import pandas as pd
import numpy as np
from anndata import AnnData
import gc

data_folder = '../data/Kinker_Kim/'

def read_data(return_barcodes=False):
    X_combined = pd.read_pickle(
        '%s/combined_count.pkl'%(data_folder),
        compression='gzip'
    )
    X_combined.index.names = ['barcode', 'sample'] + X_combined.index.names[2:]
    combined_annot_df = X_combined.reset_index()[X_combined.index.names]
    gene_names = np.array(X_combined.columns).astype(str)

    X_source = X_combined.loc[(slice(None), slice(None), slice(None), 'CELL_LINE')]
    X_target = X_combined.loc[(slice(None), slice(None), slice(None), 'TUMOR')]

    if X_source.index.nlevels == 4:
        X_source.index = X_source.index.droplevel(3)
    if X_target.index.nlevels == 4:
        X_target.index = X_target.index.droplevel(3)

    assert X_source.shape[1] == X_combined.shape[1]
    assert X_target.shape[1] == X_combined.shape[1]
    assert X_source.shape[0] + X_target.shape[0] == X_combined.shape[0]

    X_source = AnnData(
        X_source.values, 
        obs=pd.DataFrame(np.array([np.array(e) for e in X_source.index]),
                         columns=['UMI', 'sample', 'pool']),
        var=pd.DataFrame(X_source.columns)
    )
    X_target = AnnData(
        X_target.values, 
        obs=pd.DataFrame(np.array([np.array(e) for e in X_target.index]),
                         columns=['UMI', 'sample', 'pool']),
        var=pd.DataFrame(X_target.columns)
    )
    X_input = {
        'source': X_source,
        'target': X_target
    }
    barcodes_df = X_combined.index.to_frame().reset_index(drop=True)
    del X_combined
    gc.collect()


    cell_line_annot_df = combined_annot_df[combined_annot_df['type'] == 'CELL_LINE']
    tumor_annot_df = combined_annot_df[combined_annot_df['type'] == 'TUMOR']
    tumor_annot_df['Index'] = tumor_annot_df['barcode'] + '_' + tumor_annot_df['sample'].str.replace('-', '_')

    if return_barcodes:
        return X_source, X_target, cell_line_annot_df, tumor_annot_df, gene_names, barcodes_df
    return X_source, X_target, cell_line_annot_df, tumor_annot_df, gene_names