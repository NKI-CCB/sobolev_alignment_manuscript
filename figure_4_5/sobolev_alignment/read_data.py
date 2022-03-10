"""
Function to read genomic data.
"""

import pandas as pd
import numpy as np
from anndata import AnnData
import gc

def read_data(data_folder, data_subfolder):
    X_combined = pd.read_csv(
        '%s/%s/combined_count.csv'%(data_folder, data_subfolder),
        sep=',', 
        index_col=[0,1,2,3]
    )

    X_source = X_combined.loc[(slice(None), slice(None), slice(None), 'CELL_LINE')]
    X_target = X_combined.loc[(slice(None), slice(None), slice(None), 'TUMOR')]

    if X_source.index.nlevels == 4:
        X_source.index = X_source.index.droplevel(-1)
    if X_target.index.nlevels == 4:
        X_target.index = X_target.index.droplevel(-1)

    assert X_source.shape[1] == X_combined.shape[1]
    assert X_target.shape[1] == X_combined.shape[1]
    assert X_source.shape[0] + X_target.shape[0] == X_combined.shape[0]

    del X_combined
    gc.collect()

    X_source = AnnData(
        X_source.values, 
        obs=pd.DataFrame(np.array([np.array(e) for e in X_source.index]),
                         columns=['UMI', 'sample', 'pool'])
    )
    X_target = AnnData(
        X_target.values, 
        obs=pd.DataFrame(np.array([np.array(e) for e in X_target.index]),
                         columns=['UMI', 'sample', 'pool'])
    )
    gc.collect()

    return X_source, X_target