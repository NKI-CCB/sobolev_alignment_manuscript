kim_folder=../data/Kim/raw/

# Download
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_Feature_Summary.xlsx
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_cell_annotation.txt.gz
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.rds.gz
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz
wget --directory-prefix=${kim_folder} ftp://ftp.ncbi.nlm.nih.gov:21/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz

# Unzip files
gunzip ${kim_folder}/*