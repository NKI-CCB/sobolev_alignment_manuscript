# Sobolev Alignment

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

