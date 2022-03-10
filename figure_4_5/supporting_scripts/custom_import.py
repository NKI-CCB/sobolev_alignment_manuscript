import os, sys, scipy, gc, torch, pylab, umap, re, matplotlib, scanpy
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
import matplotlib.pyplot as plt
from pickle import dump, load
from copy import deepcopy
from functools import reduce
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from time import process_time
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from statannot.statannot import add_stat_annotation

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri