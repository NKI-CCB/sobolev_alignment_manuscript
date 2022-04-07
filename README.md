# Sobolev Alignment

<b>NOTE</b> We here report the different scripts used to produce the figures of the manuscript entitled <a href="https://www.biorxiv.org/content/10.1101/2022.03.08.483431v1">Identifying commonalities between cell lines and tumors at the single cell level using Sobolev Alignment of deep generative models</a>. We are currently putting efforts into automating the scripts to allow easy reproduction of our results. Such automated scripts are ready for downloading and processing the data, and for reproducing Figure 3. We provide the code for Figure 4-5 but the automation is not finished yet.

## Setting up environment

Different Python and R packages are used to produce our results. The environment can be created using the following commands:
```
conda create --name sobolev_alignment_figures python=3.9
pip install -r requirements.txt
```
To install PyTorch, please refer <a href="https://pytorch.org/get-started/locally/">PyTorch's installation website</a>, and select the version suited to your hardware (especially if you have GPUs).
To install Sobolev Alignment, <a href="https://github.com/saroudant/sobolev_alignment">please use our implementation on GitHub</a>.
The R packages we employ are installed in their respective notebooks (specifically for Figures 2 and 3).

## Downloading data
### Kim et al 2020
The tumor dataset (Kim et al 2020, Nature Communications) can be downloaded on GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907
We provide a script to download this dataset in ./data_processing/

### Kinker et al 2020
The treatment-naive cell line (Kinker et al 2020, Nature Genetics) can be downloaded on the Broad Institute portal: https://singlecell.broadinstitute.org/single_cell/study/SCP542/pan-cancer-cell-line-heterogeneity#study-download
Data needs to be copied in ./data/Kinker/raw/ for the rest of the analysis. The scripts we used to process the data are available in ./data_processing.

### McFarland et al 2020
The multiplexed drug perturbation screen (McFarland et al, Nature Communications) can be downloaded on the FigShare: https://figshare.com/s/139f64b495dea9d88c70
Data needs to be copied in ./data/McFarland/raw/ for the rest of the analysis. The scripts we used to process the data are available in ./data_processing.

## Reproducing Figures
### Figure 1

Seurat and LIGER are implemented in R packages and their analysis are performed in two different analysis:
- alignment_all_data.ipynb: R notebook performing (and saving) the alignment.
- alignment_all_data_support.ipynb: Python notebook for UMAP and plotting (consistent with color-scheme of other figures).
LIGER is a Python package and the complete analysis is in HARMONY_alignment_all_data.ipynb

### Figure 3

Each model has its own folder. The complete analysis can be run on the Jupyter notebook "results_analysis_model_.." present in the folder, including:
- Generation of synthetic data.
- Processing of the data.
- Sobolev Alignment.
- Analysis of features and reproduction of Figure 3.

### Figures 4 and 5 (figure_4_5)

<b>NOTE:</B>The scripts supporting this figures have not been fully automated yet and require some minor manual curation.
1. Run launch_hyperopt_search.sh to compute the Hyperopt parameters for Kim, Kinker and the combined dataset.
2. Change the optimal scVI parameters in sobolev_alignment/feature_analysis_params.py
3. Run launch_feature_analysis.sh
The analysis can then be found in the different notebooks.

